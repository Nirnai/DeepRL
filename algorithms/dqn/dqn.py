import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from algorithms.utils import ActionValue
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN():
    def __init__(self, env, param):
        self.name = "DQN"
        self.env = env
        self.param = param
        self.rng = random.Random()
        self.timeStep = 0

        if self.param.SEED != None:
            self.seed(self.param.SEED)
        
        self.replay = ReplayBuffer(self.param.MEMORY_SIZE, self.rng)
        self.loss = getattr(nn.modules.loss, self.param.LOSS)()
        self.Q = ActionValue(self.param.NETWORK_ARCHITECTURE, self.param.ACTIVATION)
        self.Q_target = ActionValue(self.param.NETWORK_ARCHITECTURE, self.param.ACTIVATION)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.param.LEARNING_RATE, weight_decay=self.param.WEIGHT_DECAY)
        
        
        self.hard_target_update()
        self.update_epsilon()



    def act(self, state):
        ''' Epsilon Greedy Action selection '''
        self.Q.eval()
        eps = self.rng.random()
        if(eps < self.param.EPSILON):
            action = self.rng.randint(0, self.env.action_space.n - 1)
        else:
            action = torch.argmax(self.Q(torch.from_numpy(state).float().view(1,-1)).detach()).item()
        next_state, reward, done, _ = self.env.step(action) 
        self.replay.push(state, action, reward, next_state, done)
        return next_state, reward, done



    def learn(self):
        ''' '''
        self.Q.train()
        self.update_epsilon()
        if len(self.replay) < self.param.BATCH_SIZE:
            return

        transitions = self.replay.sample(self.param.BATCH_SIZE)
        
        state = torch.FloatTensor(np.float32(transitions.state))
        next_state = torch.FloatTensor(np.float32(transitions.next_state))
        action = torch.LongTensor(transitions.action)
        reward = torch.FloatTensor(transitions.reward)
        done = torch.FloatTensor(transitions.done)

        Q_hat = self.Q(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Target
        # if self.double:
        Q_next = self.Q_target(next_state).gather(1, torch.argmax(self.Q(next_state),1).unsqueeze(1)).squeeze(1)
        # else:
        # Q_next = self.Q_target(next_state).gather(1, torch.argmax(self.Q_target(next_state),1).unsqueeze(1)).squeeze(1)


        Q_target = reward + self.param.GAMMA * Q_next * (1 - done)
        loss = self.loss(Q_hat, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # if self.soft_update:
        self.soft_target_update()
        # else:
        #     if t % self.param.TARGET_UPDATE:
        #         self.update_target()

        return loss

    def hard_target_update(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_target_update(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.param.TAU * local_param.data + (1.0 - self.param.TAU) * target_param.data)

    def update_epsilon(self):
        self.timeStep += 1
        self.param.EPSILON = self.param.EPSILON_MIN + (self.param.EPSILON_MAX - self.param.EPSILON_MIN) * math.exp(-1. * self.timeStep / self.param.EPSILON_DECAY)

    def seed(self, seed):
        self.param.SEED = seed
        self.rng.seed(self.param.SEED)
        torch.manual_seed(self.param.SEED)
        np.random.seed(self.param.SEED)
        
    def reset(self):
        self.__init__(self.env, self.param)




class ReplayBuffer(object):
    def __init__(self, capacity, rng):
        self.buffer = deque(maxlen=capacity)
        self.rng = rng

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append(Transition(state, action, reward, next_state, done))      

    def sample(self, batch_size):
        transitions = Transition(*zip(*self.rng.sample(self.buffer, batch_size)))
        state_batch = np.concatenate(transitions.state)
        action_batch = np.array(transitions.action)
        reward_batch = np.array(transitions.reward)
        next_state_batch = np.concatenate(transitions.next_state)
        done_batch = np.float32(transitions.done)
        transitions = Transition(state_batch, 
                                 action_batch, 
                                 reward_batch, 
                                 next_state_batch, 
                                 done_batch)
        return transitions
        
    def __len__(self):
        return len(self.buffer)




# class ActionValue(nn.Module):
#     def __init__(self, architecture, activation):
#         super(ActionValue, self).__init__()
#         self.activation = getattr(nn.modules.activation, activation)()
#         layers = [self.activated_layer(in_, out_, self.activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
#         self.layers = nn.Sequential(*layers)
#         self.output = self.output_layer(architecture[-2], architecture[-1])
#         # Weight initialization for output layer so that in the beginning values are close to zero
#         self.output[-1].weight.data.uniform_(-3e-3, 3e-3)
    

#     def forward(self, state):
#         x = state
#         x = self.layers(x)
#         y = self.output(x)
#         return y

    
#     def activated_layer(self, in_, out_, activation_):
#         return nn.Sequential(
#             nn.Linear(in_, out_),
#             # nn.BatchNorm1d(out_, affine=False),
#             activation_
#         )
    
#     def output_layer(self, in_, out_):
#         return nn.Sequential(
#             nn.Linear(in_, out_)
#         )

