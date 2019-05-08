import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from itertools import accumulate

class Policy(nn.Module):
    def __init__(self, architecture, activation, is_discrete):
        super(Policy, self).__init__()
        self.activation = getattr(nn.modules.activation, activation)()
        self.is_discrete = is_discrete
        
        layers = [self.activated_layer(in_, out_, self.activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        self.layers = nn.Sequential(*layers)
        self.output = self.output_layer(architecture[-2], architecture[-1], is_discrete)
            
        
    def forward(self, state):
        x = state
        if self.is_discrete:
            x = self.layers(x)
            y = self.output(x)
            return y
            

    def activated_layer(self, in_, out_, activation_):
        return nn.Sequential(
            nn.Linear(in_, out_),
            activation_
        )
        
    def output_layer(self, in_, out_, is_discrete):
        if is_discrete:
            return nn.Sequential(
                nn.Linear(in_, out_),
                nn.Softmax()
            )
        else:
            # Returns mu and sigma
            return 
            nn.Sequential(nn.Linear(in_, out_)), 
            nn.Sequential(
                nn.Linear(in_, out_),
                nn.Softplus()
            )



class REINFORCE():
    def __init__(self, env, param):        
        self.name = "REINFORCE"
        self.env = env
        self.param = param
        if self.param.SEED is None:
            self.rng = random.Random()
        else:
            torch.manual_seed(self.param.SEED)
            np.random.seed(self.param.SEED)
            self.rng = random.Random(self.param.SEED)
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete

        self.policy = Policy(self.param.NETWORK_ARCHITECTURE, self.param.ACTIVATION, self.is_discrete)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.param.LEARNING_RATE)

        self.rewards = []
        self.log_probs = []
        

    def act(self, state):
        '''  '''
        if self.is_discrete:
            # Multinomial Action Distribution
            probs = self.policy(torch.from_numpy(state).float())
            m = dist.Categorical(probs)
        else:
            # Gaussian Action Distribution
            mu, sigma = self.policy(torch.from_numpy(state).float())
            m = dist.Normal(mu, sigma)
        action = m.sample() 
        next_state, reward, done, _ = self.env.step(action.numpy()) 

        self.log_probs.append(m.log_prob(action))
        self.rewards.append(reward)

        return next_state, reward, done


    def learn(self, done):
        if done:
            # Returns with causality
            Qs = torch.Tensor(list(accumulate(self.rewards[::-1], lambda R, r: r + self.param.GAMMA * R))[::-1])

            # Compute Loss L = -log(π(a,s)) * A 
            loss = [-log_prob * Q for log_prob, Q in zip(self.log_probs, Qs)]
            # loss = []            
            # for log_prob, Q in zip(self.log_probs, Qs):
            #     loss.append(-log_prob * Q)    
            loss = torch.stack(loss).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del self.log_probs[:]
            del self.rewards[:]
            return loss
        else:
            return 
        

    def estimate_Qs(self):
        R = 0
        Qs = []
        for r in self.rewards[::-1]:
            R = r + self.param.GAMMA * R
            Qs.insert(0, R)
        Qs = torch.Tensor(Qs)
        return Qs

    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        np.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)