import torch
import torch.nn as nn
import math

def make_mlp(params, output_dim):
    ins = params['ARCHITECTURE'][:-1]
    outs = params['ARCHITECTURE'][1:-1]
    acts = params['ACTIVATION']
    bnorm = params['BATCHNORM']
    dout = params['DROPOUT']
    init = params['INIT']
    layers = [layer(in_, 
                    out_, 
                    activation_=acts,
                    batchnorm=bnorm,
                    dropout=dout).apply(inits[init]) for in_, out_ in zip(ins, outs)]
    layers.append(layer(outs[-1], output_dim).apply(inits[init]))
    return unwrap_layers(nn.Sequential(*layers))


def layer(in_, out_, activation_=None, dropout=False, batchnorm=False):
    l = nn.ModuleList([nn.Linear(in_, out_)])
    if batchnorm:
        l.append(nn.BatchNorm1d(out_))
    if activation_ is not None:
        activation = getattr(nn.modules.activation, activation_)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout()) 
    return l

def unwrap_layers(model):
    l = []
    def recursive_wrap(model):
        for m in model.children():
            if isinstance(m, nn.Sequential): recursive_wrap(m)
            elif isinstance(m, nn.ModuleList): recursive_wrap(m)
            else: l.append(m)
    recursive_wrap(model)
    return nn.Sequential(*l)


def naive(m):
    if isinstance(m, nn.Linear):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / float(fan_in)), b=math.sqrt(1.0 / float(fan_in)))
        nn.init.zeros_(m.bias)

def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

def kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

inits = {
    'naive' : naive,
    'xavier': xavier,
    'kaiming': kaiming,
    'orthogonal': orthogonal
}