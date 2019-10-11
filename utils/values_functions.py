import torch
import torch.nn as nn
from copy import deepcopy


class Value(nn.Module):
    def __init__(self, params, device, init_hidden=None, init_output=None):
        super(Value, self).__init__()
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(params['ARCHITECTURE'][:-1], params['ARCHITECTURE'][1:-1])]
        hidden_layers = nn.Sequential(*layers)
        output_layer = linear_layer(params['ARCHITECTURE'][-2], 1)
        if init_hidden == 'xavier':
            hidden_layers.apply(xavier_init)
        elif init_hidden == 'kaiming':
            hidden_layers.apply(kaiming_init)
        if init_output == 'uniform':
            output_layer.apply(init_uniform_small)
        elif init_output == 'normal':
            output_layer.apply(init_normal_small)
        self.value = unwrap_layers(nn.Sequential(hidden_layers, output_layer))
        self.device = device
        self.to(self.device)

    def forward(self, state):
        return self.value(state).squeeze()


class QValue(nn.Module):
    def __init__(self, params, device, init_hidden=None, init_output=None):
        super(QValue, self).__init__()
        architecture = deepcopy(params['ARCHITECTURE'])
        architecture[0] = params['STATE_DIM'] + params['ACTION_DIM']
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        hidden_layers = nn.Sequential(*layers)
        output_layer = linear_layer(architecture[-2], 1)
        if init_hidden == 'xavier':
            hidden_layers.apply(xavier_init)
        elif init_hidden == 'kaiming':
            hidden_layers.apply(kaiming_init)
        if init_output == 'uniform':
            output_layer.apply(init_uniform_small)
        elif init_output == 'normal':
            output_layer.apply(init_normal_small)
        self.qvalue = unwrap_layers(nn.Sequential(hidden_layers, output_layer))
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        return self.qvalue(xu).squeeze()



def activated_layer(in_, out_, activation_):
    return nn.Sequential(
        nn.Linear(in_, out_),
        activation_
        )

def linear_layer(in_, out_):
    return nn.Sequential(
        nn.Linear(in_, out_)
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