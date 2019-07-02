import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def init_policy_weights(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.mul_(0.1)
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.zeros_(m.bias)

def init_output_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -3e-3, 3e-3)
        nn.init.zeros_(m.bias)


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


class Policy(nn.Module):
    def __init__(self, architecture, activation, action_space='discrete'):
        super(Policy, self).__init__()
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]
        self.action_space = action_space

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        affine_layers = nn.Sequential(*layers)
        affine_layers.apply(xavier_init)

        if self.action_space is 'discrete':
            policy_output = softmax_layer(architecture[-2], self.num_outputs)
            policy_output.apply(init_policy_weights)
            policy_layers = nn.Sequential(affine_layers, policy_output)
        elif self.action_space is 'continuous':
            action_mean = linear_layer(architecture[-2], self.num_outputs)
            action_mean.apply(init_policy_weights)
            policy_layers = nn.Sequential(affine_layers, action_mean)
            self.action_log_std = nn.Parameter(torch.zeros(self.num_outputs))
        self._policy = unwrap_layers(policy_layers)
        # self._policy.apply(kaiming_init)
        
    
    def forward(self, state, exploit=False):
        with torch.no_grad():
            policy = self.policy(state)
            if exploit:
                if hasattr(policy, 'probs'):
                    return torch.argmax(policy.probs)
                else:
                    return policy.mean
            else:
                return policy.sample()

    def policy(self, state):
        if self.action_space is 'discrete':
            probs = self._policy(state)
            policy = Categorical(probs)
        if self.action_space is 'continuous':
            mean = self._policy(state)
            std = torch.exp(self.action_log_std.expand_as(mean))
            policy = Normal(mean, std)
        return policy

    def log_probs(self, states, actions):
        policy = self.policy(states)
        return policy.log_prob(actions).squeeze()
    
    def entropy(self, states):
        policy = self.policy(states)
        return policy.entropy()

    def grad(self, loss):
        with torch.no_grad():
            grads = torch.autograd.grad(loss, self.parameters())
        return parameters_to_vector(grads)
    
    # Make properties
    def get_params(self):
        return parameters_to_vector(self.parameters())
    
    def set_params(self, flat_params):
        vector_to_parameters(flat_params, self.parameters())
    
# class SoftPolicy(nn.Module):
#     def __init__(self, architecture, activation, action_min, action_max, action_space='continuous'):
#         super(SoftPolicy, self).__init__()
#         self.num_inputs = architecture[0]
#         self.num_outputs = architecture[-1]
#         self.action_space = action_space
#         self.action_min = action_min
#         self.action_max = action_max

#         activation = getattr(nn.modules.activation, activation)()
#         layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
#         affine_layers = nn.Sequential(*layers)
#         affine_layers.apply(init_hidden_weights)

#         if self.action_space is 'discrete':
#             raise NotImplementedError
#         elif self.action_space is 'continuous':
#             action_params = linear_layer(architecture[-2], 2*self.num_outputs)
#             action_params.apply(init_policy_weights)
#             policy_layers = nn.Sequential(affine_layers, action_params)
#         self.policy = unwrap_layers(policy_layers)
    
#     def forward(self, state):
#         if self.action_space is 'discrete':
#             raise NotImplementedError
#         if self.action_space is 'continuous':
#             mean, std = self.policy(state).chunk(2)
#             std = torch.clamp(std, min=self.action_min, max=self.action_max)
#             dist = Normal(mean, std)
            

#         return dist


class Value(nn.Module):
    def __init__(self, architecture, activation):
        super(Value, self).__init__()
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        affine_layers = nn.Sequential(*layers)
        affine_layers.apply(xavier_init)

        output_layer = linear_layer(architecture[-2], 1)
        output_layer.apply(init_output_weights)
        self.value = unwrap_layers(nn.Sequential(affine_layers, output_layer))
        # self.value.apply(xavier_init)

    def forward(self, state):
        return self.value(state)


class QValue(nn.Module):
    def __init__(self, architecture, activation):
        super(QValue, self).__init__()
        arch_mod = deepcopy(architecture)
        arch_mod[0] = architecture[0] + architecture[-1]
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(arch_mod[:-1], arch_mod[1:-1])]
        affine_layers = nn.Sequential(*layers)
        affine_layers.apply(xavier_init)

        output_layer = linear_layer(architecture[-2], 1)
        output_layer.apply(init_policy_weights)
        
        self.qvalue = unwrap_layers(nn.Sequential(affine_layers, output_layer))

    def forward(self, state, action):
        in_ = torch.cat((state,action), dim=1)
        return self.qvalue(in_)