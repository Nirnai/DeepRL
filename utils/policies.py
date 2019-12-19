import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import numpy as np
from utils.helper import soft_target_update
from utils.torch_utils import make_mlp



class GaussianPolicy(nn.Module):
    def __init__(self, params, device):
        super(GaussianPolicy, self).__init__()
        self.mean = make_mlp(params, params['ARCHITECTURE'][-1])
        self.log_std = nn.Parameter(torch.ones(params['ARCHITECTURE'][-1]) * 0.0)
        self.optimizer = optim.Adam(self.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
        self.max_grad = params['MAX_GRAD_NORM']
        self.clipped_action = params['CLIPPED_ACTION']
        self.action_min = torch.tensor([-1]).float()
        self.action_max = torch.tensor([1]).float()
        self.device = device
        self.to(self.device)

    def forward(self, state, deterministic=False):
        if deterministic:
            return self.mean(state)
        else:
            policy = self.policy(state)
            return policy.sample()

    def policy(self, state):
        if self.clipped_action:
            mean = self.mean(state)
            mean = self.action_min + (self.action_max - self.action_min) * torch.sigmoid(mean)
            std_min = torch.tensor(0.01)
            std_max = torch.tensor(1.0 * (self.action_max - self.action_min))
            max_log_std = torch.log(std_max)
            min_log_std = torch.log(std_min)
            std = torch.exp(min_log_std + (max_log_std - min_log_std) * torch.sigmoid(self.log_std.expand_as(mean)))
            # std = torch.exp(self.log_std.expand_as(mean))
        else:    
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
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad)
        pg_norm = self.get_grad_norm()
        self.optimizer.step()
        return pg_norm


class BoundedGaussianPolicy(GaussianPolicy):
    def __init__(self, params, device):
        super(BoundedGaussianPolicy, self).__init__(params, device)
        # self.log_std = nn.Parameter(torch.ones(params['ARCHITECTURE'][-1])*-0.2)

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


class ClippedGaussianPolicy(GaussianPolicy):
    def __init__(self, params, device):
        super(ClippedGaussianPolicy, self).__init__(params, device)
    
    def forward(self, state, deterministic=False):
        if deterministic:
            return torch.clamp(self.mean(state), min=self.action_min.item(), max=self.action_max.item())
        else:
            policy = self.policy(state)
            return torch.clamp(policy.sample(), min=self.action_min.item(), max=self.action_max.item())

    def log_prob(self, state, action):
        policy = self.policy(state)
        unclipped_log_probs = policy.log_prob(action)
        low_log_probs = self.safe_log(policy.cdf(self.action_min))  
        high_log_probs = self.safe_log(1 - policy.cdf(self.action_max))

        elementwise_log_prob = torch.where(
            action <= self.action_min, low_log_probs, 
            torch.where(action >= self.action_max, high_log_probs, 
            unclipped_log_probs))
        return elementwise_log_prob.sum(-1)
    
    def safe_log(self, x):
        return torch.log(torch.where(x.data > 0, x, x.data))


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

    # def forward(self, state):
    #     if state.dim() == 2:
    #         mean = torch.zeros(state.shape[0], self.action_dim).to(self.device)
    #         std = torch.ones(state.shape[0], self.action_dim).to(self.device)
    #     else:
    #         mean = torch.Tensor([0.0] * self.action_dim).to(self.device)
    #         std = torch.Tensor([1.0] * self.action_dim).to(self.device)
    #     states = torch.cat(self.batch*[state.unsqueeze(0)], dim=0)
    #     for i in range(self.iterations):
    #         p = dist.Normal(mean, std)
    #         actions = p.sample((self.batch,))
    #         actions = torch.tanh(actions)
    #         with torch.no_grad():
    #             Qs = self.q_function(states, actions)
    #         Is = Qs.topk(self.topk , dim=0)[1]
    #         if Is.dim() == 2:
    #             actions_topk = torch.cat([actions[Is[:,i],i,:].unsqueeze(1) for i in torch.arange(Is.shape[1])], dim=1)
    #             mean = actions_topk.mean(dim=0)
    #             std = actions_topk.std(dim=0)
    #             best_action = actions_topk[0]
    #         else:
    #             mean = actions[Is].mean(dim = 0)
    #             std = actions[Is].std(dim = 0)
    #             best_action = actions[Is[0]]
    #     return best_action

    def forward(self, state):
        if state.ndim == 2:
            mean = np.zeros((len(state), self.action_dim))
            std = np.ones((len(state), self.action_dim))
            actions = np.random.normal(mean, std, size=(self.batch, len(mean), self.action_dim))
            # state = state.cpu().numpy()
        else:
            mean = np.array([0.0] * self.action_dim)
            std = np.array([1.0] * self.action_dim)
            actions = np.random.normal(mean, std, size=(self.batch, self.action_dim))
        states = np.repeat(np.expand_dims(state,0), self.batch, axis=0)
        for i in range(self.iterations):
            actions = np.tanh(actions)
            with torch.no_grad():
                Qs = self.q_function(torch.from_numpy(states).float().to(self.device), torch.from_numpy(actions).float().to(self.device)).cpu().numpy()
            # Qs = np.random.normal(loc=0., scale=1., size=(64,10000))
            Is = np.argpartition(Qs, self.topk, axis=0)[-self.topk:]
            if Is.ndim == 2:
                _,nC,nR = actions.shape
                Is = nC*nR*Is + nR*np.arange(nR)[:,None] + np.arange(nC)
                actions_topk = np.take(actions,Is)[:,:,np.newaxis]
                mean = actions_topk.mean(axis=0)
                std = actions_topk.std(axis=0)
                best_action = torch.from_numpy(actions_topk[0]).float().to(self.device)
            else:
                mean = actions[Is].mean(axis = 0)
                std = actions[Is].std(axis = 0)
                best_action = actions[Is[0]]
        return best_action