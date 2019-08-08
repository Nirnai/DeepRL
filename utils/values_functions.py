import torch
import torch.nn as nn
from copy import deepcopy

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


class Value(nn.Module):
    def __init__(self, params, device):
        super(Value, self).__init__()
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(params['ARCHITECTURE'][:-1], params['ARCHITECTURE'][1:-1])]
        affine_layers = nn.Sequential(*layers)
        output_layer = linear_layer(params['ARCHITECTURE'][-2], 1)
        self.value = unwrap_layers(nn.Sequential(affine_layers, output_layer))
        self.device = device
        self.to(self.device)

    def forward(self, state):
        return self.value(state).squeeze()


class QValue(nn.Module):
    def __init__(self, params, device):
        super(QValue, self).__init__()
        architecture = deepcopy(params['ARCHITECTURE'])
        architecture[0] = params['STATE_DIM'] + params['ACTION_DIM']
        activation = getattr(nn.modules.activation, params['ACTIVATION'])()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        affine_layers = nn.Sequential(*layers)
        output_layer = linear_layer(architecture[-2], 1)
        self.qvalue = unwrap_layers(nn.Sequential(affine_layers, output_layer))
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        return self.qvalue(xu).squeeze()