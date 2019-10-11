import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from utils.helper import soft_target_update


class GaussianPolicy(nn.Module):
    def __init__(self, params, device):
        super(GaussianPolicy, self).__init__()
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(params['ARCHITECTURE'][:-1], params['ARCHITECTURE'][1:-1])]
        hidden_layers = nn.Sequential(*layers)
        output_layer = linear_layer(params['ARCHITECTURE'][-2], params['ARCHITECTURE'][-1])
        if params['INIT_HIDDEN'] == 'xavier':
            hidden_layers.apply(xavier_init)
        elif params['INIT_HIDDEN'] == 'orthogonal':
            hidden_layers.apply(orthogonal)
        elif params['INIT_HIDDEN'] == 'kaiming':
            hidden_layers.apply(kaiming_init)
        if params['INIT_OUTPUT'] == 'uniform':
            output_layer.apply(init_uniform_small)
        elif params['INIT_OUTPUT'] == 'normal':
            output_layer.apply(init_normal_small)
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
        if params['INIT_HIDDEN'] == 'xavier':
            hidden_layers.apply(xavier_init)
        elif params['INIT_HIDDEN'] == 'orthogonal':
            hidden_layers.apply(orthogonal)
        elif params['INIT_HIDDEN'] == 'kaiming':
            hidden_layers.apply(kaiming_init)
        if params['INIT_OUTPUT'] == 'uniform':
            output_layer.apply(init_uniform_small)
        elif params['INIT_OUTPUT'] == 'normal':
            output_layer.apply(init_normal_small)
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


# class TestCrossEntropyGuidedPolicy(nn.Module):
#     def __init__(self, q_function, action_space, observation_space, params, device):
#         super(TestCrossEntropyGuidedPolicy, self).__init__()
#         self.q_function = q_function
#         self.high = torch.tensor(action_space.high, dtype=torch.float, device=device)
#         self.low = torch.tensor(action_space.low, dtype=torch.float, device=device)
#         self.action_dim = action_space.shape[0]
#         self.observation_dim = observation_space.shape[0]
#         self.iterations = params['CEM_ITERATIONS']
#         self.batch = params['CEM_BATCH']
#         self.topk = params['CEM_TOPK']
#         self.device = device
#         self.to(self.device)

#     def forward(self, state):
#         if len(state.shape) < 2:
#             state = state.unsqueeze(0)
#         obs_batch = state.shape[0]
#         actions = torch.linspace(0, 1, self.batch, device=self.device)
#         actions = actions.reshape(self.batch, -1) * (self.high - self.low) + self.low
#         actions = actions.repeat((obs_batch, 1))
#         states = state.repeat((1, self.batch)).reshape((obs_batch * self.batch, self.observation_dim))
#         for i in range(self.iterations + 1):
#             with torch.no_grad():
#                 Qs, _ = self.q_function(states, actions)
#             if i != self.iterations:
#                 Qs = Qs.reshape((obs_batch, self.batch))
#                 _, idxs = torch.sort(Qs, dim=1, descending=True)  
#                 Is = idxs[:, :self.topk]
#                 Is = Is + torch.arange(0, self.batch * obs_batch, self.batch, device=self.device).reshape((obs_batch, 1))
#                 Is = Is.reshape((self.topk * obs_batch,))
#                 actions_topk =  actions[Is, :]           
#                 actions_topk = actions_topk.reshape((obs_batch, self.topk, self.action_dim))
#                 mean = actions_topk.mean(dim=1)
#                 std = actions_topk.std(dim=1)
#                 actions = dist.Normal(loc=mean, scale=std).rsample(torch.Size((self.batch,)))
#                 actions = actions.transpose(1, 0)
#                 actions = actions.reshape((self.batch * obs_batch, self.action_dim)) 
#                 actions = (actions - self.low) / (self.high - self.low)
#                 actions = torch.clamp(actions, 0, 1) * (self.high - self.low) + self.low
#         Qs = Qs.reshape((obs_batch, self.batch))
#         actions = actions.reshape((obs_batch, self.batch, self.action_dim))
#         Q_max, idx = torch.max(Qs, dim=1)
#         action_max = actions[torch.arange(obs_batch, device=self.device), idx]
#         return action_max


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


def init_uniform_small(m):
    # DDPG initialization
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-3e-3, b=3e-3)
        nn.init.zeros_(m.bias)

def init_normal_small(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)
        nn.init.zeros_(m.bias)

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)
    