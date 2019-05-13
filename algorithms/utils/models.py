import torch
import torch.nn as nn


def init_hidden_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def init_output_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -3e-3, 3e-3)
        nn.init.zeros_(m.bias)


def activated_layer(in_, out_, activation_):
    return nn.Sequential(
        nn.Linear(in_, out_),
        activation_
        )

def softmax_layer(in_, out_):
    return nn.Sequential(
        nn.Linear(in_, out_),
        nn.Softmax(dim=-1)
    )

def linear_layer(in_, out_):
    return nn.Sequential(
        nn.Linear(in_, out_)
    )



class Policy(nn.Module):
    def __init__(self, architecture, activation, output='discrete'):
        super(Policy, self).__init__()
        activation_ = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation_) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_hidden_weights)

        if output is 'discrete':
            self.output = softmax_layer(architecture[-2], architecture[-1])
        elif output is 'continuous':
            self.output = linear_layer(architecture[-2], architecture[-1])
        else: 
            raise NotImplementedError

    def forward(self, state):
        x = state
        x = self.layers(x)
        policy = self.output(x)
        return policy



class Value(nn.Module):
    def __init__(self, architecture, activation):
        super(Value, self).__init__()
        activation_ = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation_) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_hidden_weights)

        self.output = linear_layer(architecture[-2], architecture[-1])
        self.output.apply(init_output_weights)
        


    def forward(self, state):
        x = state
        x = self.layers(x)
        value = self.output(x)
        return value



class ActionValue(nn.Module):
    def __init__(self, architecture, activation):
        super(ActionValue, self).__init__()
        self.activation = getattr(nn.modules.activation, activation)()
        layers = [self.activated_layer(in_, out_, self.activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        self.layers = nn.Sequential(*layers)
        self.output = self.output_layer(architecture[-2], architecture[-1])
        self.output[-1].weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = state
        x = self.layers(x)
        y = self.output(x)
        return y



class ActorCritic(nn.Module):
    def __init__(self, architecture, activation, output='discrete'):
        super(ActorCritic, self).__init__()
        activation = getattr(nn.modules.activation, activation)()
        layers = [self.activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_hidden_weights)
        
        if output is 'discrete':
            self.actor = self.softmax_layer(architecture[-2], architecture[-1])
        elif output is 'continuous':
            self.actor = self.linear_layer(architecture[-2], architecture[-1])
        else:
            raise NotImplementedError
       
        self.critic = self.linear_layer(architecture[-2])
        self.critic.apply(init_output_weights)

    def forward(self, state):
        x = state
        x = self.layers(x)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value
    
    def policy(self, state):
        x = state
        x = self.layers(x)
        policy = self.actor(x)
        return policy

    def value(self, state):
        x = state
        x = self.layers(x)
        value = self.critic(x)
        return value