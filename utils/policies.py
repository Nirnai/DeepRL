import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from utils.helper import soft_target_update
from utils.torch_utils import make_mlp


class GaussianPolicy(nn.Module):
    def __init__(self, params, device):
        super(GaussianPolicy, self).__init__()
        self.mean = make_mlp(params, params['ARCHITECTURE'][-1])
        self.log_std = nn.Parameter(torch.ones(params['ARCHITECTURE'][-1]))
        self.optimizer = optim.Adam(self.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
        self.device = device
        self.to(self.device)

    def forward(self, state, deterministic=False):
        if deterministic:
            return self.mean(state)
        else:
            policy = self.policy(state)
            return policy.sample()

    def policy(self, state):
        mean = self.mean(state)
        std = torch.exp(self.log_std.expand_as(mean))
        policy = dist.Normal(mean, std)
        return policy

    def log_prob(self, state, action):
        policy = self.policy(state)
        return policy.log_prob(action).sum(dim=-1)
    
    def entropy(self, state):
        policy = self.policy(state)
        return policy.entropy()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class BoundedGaussianPolicy(GaussianPolicy):
    def __init__(self, params, device):
        super(BoundedGaussianPolicy, self).__init__(params, device)

    def forward(self, state, deterministic=False):
        if deterministic:
            return torch.tanh(self.mean(state))
        else:
            policy = self.policy(state)
            return torch.tanh(policy.sample())
        
    def log_prob(self, state, action):
        policy = self.policy(state)
        return (policy.log_prob(action) - torch.log1p(-torch.tanh(action).pow(2) + 1e-6)).sum(dim=-1)

    def rsample(self, state):
        policy = self.policy(state)
        action = policy.rsample()
        log_prob = (policy.log_prob(action) - torch.log1p(-torch.tanh(action).pow(2) + 1e-6)).sum(dim=-1)
        return torch.tanh(action), log_prob


class DeterministicPolicy(nn.Module):
    def __init__(self, params, device):
        super(DeterministicPolicy, self).__init__()
        self.action = make_mlp(params, params['ARCHITECTURE'][-1])
        self.optimizer = optim.Adam(self.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
        self.device = device
        self.to(self.device)
    
    def forward(self, state):
        return torch.tanh(self.action(state))

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        


class CrossEntropyGuidedPolicy(nn.Module):
    def __init__(self, q_function, params, device):
        super(CrossEntropyGuidedPolicy, self).__init__()
        self.q_function = q_function
        self.action_dim = params['ACTION_DIM']
        self.iterations = params['CEM_ITERATIONS']
        self.batch = params['CEM_BATCH']
        self.topk = params['CEM_TOPK']
        self.device = device
        self.to(self.device)

    def forward(self, state):
        if state.dim() == 2:
            mean = torch.zeros(state.shape[0], self.action_dim).to(self.device)
            std = torch.ones(state.shape[0], self.action_dim).to(self.device)
        else:
            mean = torch.Tensor([0.0] * self.action_dim).to(self.device)
            std = torch.Tensor([1.0] * self.action_dim).to(self.device)
        states = torch.cat(self.batch*[state.unsqueeze(0)], dim=0)
        for i in range(self.iterations):
            p = dist.Normal(mean, std)
            actions = p.sample((self.batch,))
            actions = torch.tanh(actions)
            with torch.no_grad():
                Qs = self.q_function(states, actions)
            Is = Qs.topk(self.topk , dim=0)[1]
            if Is.dim() == 2:
                actions_topk = torch.cat([actions[Is[:,i],i,:].unsqueeze(1) for i in torch.arange(Is.shape[1])], dim=1)
                mean = actions_topk.mean(dim=0)
                std = actions_topk.std(dim=0)
                best_action = actions_topk[0]
            else:
                mean = actions[Is].mean(dim = 0)
                std = actions[Is].std(dim = 0)
                best_action = actions[Is[0]]
        return best_action

