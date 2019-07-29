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
    def __init__(self, architecture, activation):
        super(Value, self).__init__()
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        affine_layers = nn.Sequential(*layers)
        # affine_layers.apply(xavier_init)

        output_layer = linear_layer(architecture[-2], 1)
        # output_layer.apply(init_output_weights)
        self.value = unwrap_layers(nn.Sequential(affine_layers, output_layer))
        self.to(self.device)
        # self.value.apply(xavier_init)

    def forward(self, state):
        return self.value(state).squeeze()


class QValue(nn.Module):
    def __init__(self, architecture, activation):
        super(QValue, self).__init__()
        arch_mod = deepcopy(architecture)
        arch_mod[0] = architecture[0] + architecture[-1]
        self.num_inputs = architecture[0]
        self.num_outputs = architecture[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        activation = getattr(nn.modules.activation, activation)()
        layers = [activated_layer(in_, out_, activation) for in_, out_ in zip(arch_mod[:-1], arch_mod[1:-1])]
        affine_layers = nn.Sequential(*layers)
        # affine_layers.apply(xavier_init)

        output_layer = linear_layer(architecture[-2], 1)
        # output_layer.apply(init_output_weights)
        self.qvalue = unwrap_layers(nn.Sequential(affine_layers, output_layer))
        self.to(self.device)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        return self.qvalue(xu).squeeze()