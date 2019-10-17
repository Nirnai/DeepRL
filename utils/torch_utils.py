import torch
import torch.nn as nn


def make_mlp(params, output_dim):
    ins = params['ARCHITECTURE'][:-1]
    outs = params['ARCHITECTURE'][1:-1]
    acts = params['ACTIVATION']
    bnorm = params['BATCHNORM']
    dout = params['DROPOUT']
    hinit = params['INIT_HIDDEN']
    oinit = params['INIT_OUTPUT']
    layers = [layer(in_, 
                    out_, 
                    activation_=acts,
                    batchnorm=bnorm,
                    dropout=dout).apply(init[hinit]) for in_, out_ in zip(ins, outs)]
    layers.append(layer(outs[-1], output_dim).apply(init[oinit]))
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


def ddpg_init(m):
    # DDPG initialization
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-3e-3, b=3e-3)
        nn.init.zeros_(m.bias)

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)

def default_init(m):
    return

init = {
    'xavier': xavier_init,
    'kaiming': kaiming_init,
    'orthogonal': orthogonal_init,
    'default' : default_init,
    'ddpg': ddpg_init
}