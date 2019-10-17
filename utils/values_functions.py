import torch
import torch.nn as nn
import torch.optim as optim
from utils.torch_utils import make_mlp


class ActionValueFunction():
    def __init__(self, param, device):
        param['ARCHITECTURE'][0] += param['ARCHITECTURE'][-1]
        self.Q1 = QValue(param, device)
        self.Q2 = QValue(param, device)
        self.Q1_target = QValue(param, device)
        self.Q2_target = QValue(param, device)
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q1.state_dict())
        self.Q1_target.eval()
        self.Q2_target.eval()
        self.optimizer = optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=param['LEARNING_RATE'], weight_decay=param['WEIGHT_DECAY'])
        self._tau = param['TAU']
    
    def __call__(self, state, action):
        return self.Q1(state, action), self.Q2(state, action)

    def target(self, state, action):
        return self.Q1_target(state, action), self.Q2_target(state, action)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_update()
    
    def target_update(self):
        for target_param, local_param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)
        for target_param, local_param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)


class ValueFunction():   
    def __init__(self, param, device):
        self.V = Value(param, device)
        self.optimizer = optim.Adam(self.V.parameters(), lr=param['LEARNING_RATE'], weight_decay=param['WEIGHT_DECAY'])
    
    def parameters(self):
        return self.V.parameters()
    
    def eval(self):
        self.V.eval()
    
    def train(self):
        self.V.train()
    
    def __call__(self, state):
        return self.V(state)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




class Value(nn.Module):
    def __init__(self, params, device):
        super(Value, self).__init__()
        self.value = make_mlp(params, 1)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        return self.value(state).squeeze()


class QValue(nn.Module):
    def __init__(self, params, device):
        super(QValue, self).__init__()
        self.qvalue = make_mlp(params, 1)
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        return self.qvalue(xu).squeeze()


