import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from utils.helper import soft_target_update

def activated_layer(in_, out_, activation_):
    return nn.Sequential(
        nn.Linear(in_, out_),
        activation_
        )

def linear_layer(in_, out_):
    return nn.Sequential(
        nn.Linear(in_, out_)
    )

def softmax_layer(in_, out_):
    return nn.Sequential(
        nn.Linear(in_, out_),
        nn.Softmax(dim=-1)
    )

def unwrap_layers(model):
    l = []
    def recursive_wrap(model):
        for m in model.children():
            if isinstance(m, nn.Sequential): recursive_wrap(m)
            else: l.append(m)
    recursive_wrap(model)
    return nn.Sequential(*l)


class GaussianPolicy(nn.Module):
    def __init__(self, params, device):
        super(GaussianPolicy, self).__init__()
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(params['ARCHITECTURE'][:-1], params['ARCHITECTURE'][1:-1])]
        hidden_layers = nn.Sequential(*layers)
        output_layer = linear_layer(params['ARCHITECTURE'][-2], params['ARCHITECTURE'][-1])
        self._mean = unwrap_layers(nn.Sequential(hidden_layers, output_layer))
        self._log_std = nn.Parameter(torch.ones(params['ARCHITECTURE'][-1]))
        self._optim = optim.Adam(self.parameters(), lr=params['LEARNING_RATE'])
        self.device = device
        self.to(self.device)

    def forward(self, state, deterministic=False):
        if deterministic:
            return self._mean(state)
        else:
            policy = self.policy(state)
            return policy.sample()

    def policy(self, state):
        mean = self._mean(state)
        std = torch.exp(self._log_std.expand_as(mean))
        policy = dist.Normal(mean, std)
        return policy

    def log_prob(self, state, action):
        policy = self.policy(state)
        return policy.log_prob(action).sum(dim=-1)
    
    def entropy(self, state):
        policy = self.policy(state)
        return policy.entropy()

    def optimize(self, loss):
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()



class BoundedGaussianPolicy(GaussianPolicy):
    def __init__(self, params, device):
        super(BoundedGaussianPolicy, self).__init__(params, device)

    def forward(self, state, deterministic=False):
        if deterministic:
            return torch.tanh(self._mean(state))
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
        self.device = device
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(params['ARCHITECTURE'][:-1], params['ARCHITECTURE'][1:-1])]
        hidden_layers = nn.Sequential(*layers)
        output_layer = linear_layer(params['ARCHITECTURE'][-2], params['ARCHITECTURE'][-1])
        self._action = unwrap_layers(nn.Sequential(hidden_layers, output_layer))
        self._optim = optim.Adam(self.parameters(), lr=params['LEARNING_RATE'])
        self.to(self.device)
    
    def forward(self, state):
        return torch.tanh(self._action(state))

    def optimize(self, loss):
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        

 

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
        
        for i in range(self.iterations):
            p = dist.Normal(mean, std)
            states = torch.cat(self.batch*[state.unsqueeze(0)], dim=0)
            actions = p.sample((self.batch,))
            with torch.no_grad():
                Qs, _ = self.q_function(states, actions)
            Is = Qs.topk(self.topk , dim=0)[1]
            if Is.dim() == 2:
                mean = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(actions, Is)]).mean(dim = 0)
                std = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(actions, Is)]).std(dim = 0)
                best_action = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(actions, Is)])[0]
            else:
                mean = actions[Is].mean(dim = 0)
                std = actions[Is].std(dim = 0)
                best_action = actions[Is[0]]
        return best_action



class EpsilonGreedyPolicy():
    pass

class DiscretePolicy():
    pass





