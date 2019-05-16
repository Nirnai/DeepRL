import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


def init_weights_underTest(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

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

def continuous_stochatic_layer(in_, out_):
    return nn.ModuleList([nn.Sequential(
        nn.Linear(in_, out_),
        nn.Tanh()
    ), nn.Sequential(
        nn.Linear(in_, out_),
        nn.Softplus()
    )])


def continuous_deterministic_layer(in_, out_):
    return nn.ModuleList([nn.Sequential(nn.Linear(in_, out_))])

class Policy(nn.Module):
    def __init__(self, architecture, activation, action_space='discrete', deterministic=False):
        super(Policy, self).__init__()
        activation_ = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation_) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_hidden_weights)

        if action_space is 'discrete':
            self.output = softmax_layer(architecture[-2], architecture[-1])
            self.dist = torch.distributions.Categorical
        elif action_space is 'continuous':
            if deterministic:
                self.output = continuous_deterministic_layer(architecture[-2], architecture[-1])
                self.dist = lambda mu : mu
            else:
                self.output = continuous_stochatic_layer(architecture[-2], architecture[-1])
                self.dist = torch.distributions.Normal 
            self.output.apply(init_output_weights)
        else: 
            raise NotImplementedError
        

    def forward(self, state):
        x = state
        x = self.layers(x)
        params = []
        for output in self.output:
            params.append(output(x))
        policy = self.dist(*params)
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
        
        self.state_input_layer = activated_layer(architecture[0], architecture[1], self.activation)
        self.state_input_layer.apply(init_hidden_weights)

        self.action_input_layer = activated_layer(architecture[1] + architecture[-1], architecture[2], self.activation)
        self.action_input_layer.apply(init_hidden_weights)

        if len(architecture) > 3:
            layers = [activated_layer(in_, out_, self.activation) for in_, out_ in zip(architecture[2:-1], architecture[3:-1])]
            self.layers = nn.Sequential(*layers)
            self.layers.apply(init_hidden_weights)
            self.output = linear_layer(architecture[-2], architecture[-1])
            self.output.apply(init_output_weights)
        else:
            self.layers = nn.Sequential()
            self.output = nn.Sequential()
    
    def forward(self, state, action):
        x = state
        out = self.state_input_layer(x)
        out = self.action_input_layer(torch.cat([out,action]))
        out = self.layers(out)
        y = self.output(out)
        return y



class ActorCritic(nn.Module):
    def __init__(self, architecture, activation, action_space='discrete', exploration=0.0):
        super(ActorCritic, self).__init__()
        self.action_space = action_space
        self.log_std = nn.Parameter(torch.ones(1, architecture[-1]) * exploration)
        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]

        # Create Actor and Critic Network
        critic_output = [linear_layer(architecture[-2], 1)]
        critic_layers = layers + critic_output
        if self.action_space is 'discrete':
            actor_output = [softmax_layer(architecture[-2], architecture[-1])]
            actor_layers = layers + actor_output
        elif self.action_space is 'continuous':
            actor_output = [linear_layer(architecture[-2], architecture[-1])]
            actor_layers = layers + actor_output
        else:
            raise NotImplementedError
        
        self.critic = nn.Sequential(*critic_layers)
        self.actor = nn.Sequential(*actor_layers)
        self.apply(init_weights_underTest)

    def forward(self, state):
        value = self.critic(state)
        if self.action_space is 'discrete':
            probs = self.actor(state)
            dist = Categorical(probs)
        if self.action_space is 'continuous':
            mu = self.actor(state)
            std = self.log_std.exp().squeeze().expand_as(mu)
            dist = Normal(mu, std)
        return dist, value
    

