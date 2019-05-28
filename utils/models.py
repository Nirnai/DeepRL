import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, kl_divergence
from copy import deepcopy

__all__ = ['Value', 'Policy']

def init_policy_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.mul_(0.1)
        # nn.init.normal_(m.weight, mean=0.001, std=0.01)
        nn.init.zeros_(m.bias)

def init_hidden_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
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
        affine_layers.apply(init_hidden_weights)

        if self.action_space is 'discrete':
            policy_output = softmax_layer(architecture[-2], self.num_outputs)
            policy_output.apply(init_policy_weights)
            policy_layers = nn.Sequential(layers, policy_output)
        elif self.action_space is 'continuous':
            action_mean = linear_layer(architecture[-2], self.num_outputs)
            action_mean.apply(init_policy_weights)
            policy_layers = nn.Sequential(affine_layers, action_mean)
        self.policy = unwrap_layers(policy_layers)
        self.action_log_std = nn.Parameter(torch.zeros(self.num_outputs))
    
    def forward(self, state):
        if self.action_space is 'discrete':
            probs = self.policy(state)
            dist = Categorical(probs)
        if self.action_space is 'continuous':
            mean = self.policy(state)
            std = torch.exp(self.action_log_std.expand_as(mean))
            dist = Normal(mean, std)
        return dist

    def get_grads(self, loss):
        grads = torch.autograd.grad(loss, self.parameters(), allow_unused=True)
        # TODO: Missmatch of dim to parameters
        grads = [torch.Tensor([0]) if grad is None else grad for grad in grads]
        grads_flat = torch.cat([grad.view(-1) for grad in grads]).detach() 
        return grads_flat
    
    def get_params(self):
        params = self.state_dict().values()
        params = torch.cat([param.view(-1) for param in params])
        return params


class Value(nn.Module):
    def __init__(self, architecture, activation):
        super(Value, self).__init__()
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        affine_layers = nn.Sequential(*layers)
        affine_layers.apply(init_hidden_weights)

        output_layer = linear_layer(architecture[-2], 1)
        output_layer.apply(init_policy_weights)
        
        self.value = unwrap_layers(nn.Sequential(affine_layers, output_layer))

    def forward(self, state):
        return self.value(state)



class ActorCritic(nn.Module):
    def __init__(self, architecture, activation, action_space='discrete'):
        super(ActorCritic, self).__init__()
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]
        self.action_space = action_space

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        affine_layers = nn.Sequential(*layers)
        affine_layers.apply(init_hidden_weights)

        value = linear_layer(architecture[-2], 1)
        value.apply(init_output_weights)
        critic_layers = nn.Sequential(affine_layers, value)
        self.critic = unwrap_layers(critic_layers)

        if self.action_space is 'discrete':
            actor_output = softmax_layer(architecture[-2], architecture[-1])
            actor_output.apply(init_policy_weights)
            actor_layers = nn.Sequential(layers, actor_output)
        elif self.action_space is 'continuous':
            action_mean = linear_layer(architecture[-2], self.num_outputs)
            action_mean.apply(init_policy_weights)
            actor_layers = nn.Sequential(affine_layers, action_mean)
        self.actor = unwrap_layers(actor_layers)
        
        self.action_std = nn.Parameter(torch.zeros(self.num_outputs))

        # Keep Copy of old Policy for natural policy gradient methods
        self.actor_old = deepcopy(self.actor)

    def forward(self, state, old=False):
        value = self.value(state)
        if old:
            policy = self.policy(state, self.actor_old)
        else:
            policy = self.policy(state, self.actor)
        return policy, value
    
    def value(self, state):
        return self.critic(state)

    def policy(self, state, actor):
        if self.action_space is 'discrete':
            probs = actor(state)
            dist = Categorical(probs)
        elif self.action_space is 'continuous':
            mean = actor(state)
            std = torch.exp(self.action_std.expand_as(mean))
            dist = Normal(mean, std)
        return dist

    def backup(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
