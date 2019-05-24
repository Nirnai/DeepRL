import random
import numpy
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from utils.models import Policy
from utils.env import getEnvInfo
from itertools import accumulate

class PG():
    def __init__(self, env, param):        
        self.name = "PG"
        self.env = env
        self.param = param
        self.rng = random.Random()
        
        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        self.param.NETWORK_ARCHITECTURE[0] = self.state_dim
        self.param.NETWORK_ARCHITECTURE[-1] = self.action_dim

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.policy = Policy(self.param.NETWORK_ARCHITECTURE, self.param.ACTIVATION, action_space=self.action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.param.LEARNING_RATE)

        

        self.rewards = []
        self.log_probs = []
        self.done = False
        

    def act(self, state):
        '''  '''
        policy = self.policy(torch.from_numpy(state).float())
        action = policy.sample() 
        next_state, reward, self.done, _ = self.env.step(action.numpy()) 

        self.log_probs.append(policy.log_prob(action))
        self.rewards.append(reward)

        return next_state, reward, self.done


    def learn(self):
        if self.done:
            # Monte Carlo estimate of returns 
            Qs = self.monte_carlo_estimate(self.rewards)

            # Compute Loss L = -log(π(a,s)) * Q 
            loss = [-log_prob * Q for log_prob, Q in zip(self.log_probs, Qs)]  
            loss = torch.stack(loss).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del self.log_probs[:]
            del self.rewards[:]
        

    def monte_carlo_estimate(self, rewards):
        R = 0
        Qs = []
        for r in rewards[::-1]:
            R = r + self.param.GAMMA * R
            Qs.insert(0, R)
        Qs = torch.Tensor(Qs)
        Qs = Qs - Qs.mean()
        return Qs

    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)