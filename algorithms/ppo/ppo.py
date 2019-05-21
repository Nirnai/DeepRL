import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from algorithms.utils import ActorCritic, Policy, Value
from algorithms.utils import RolloutBuffer
from algorithms.utils import getEnvInfo
from collections import namedtuple

import time

Transition = namedtuple('Transition', ('states', 'actions' ,'rewards', 'next_states', 'mask'))


class PPO():
    def __init__(self, env, param):
        self.name = "A2C"
        self.env = env
        self.param = param
        self.rng = random.Random()

        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        self.param.ACTOR_ARCHITECTURE[0] = self.state_dim
        self.param.ACTOR_ARCHITECTURE[-1] = self.action_dim

        architecture = self.param.ACTOR_ARCHITECTURE
        activation = self.param.ACTIVATION

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.actor = Policy(architecture, activation, action_space=self.action_space)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)

        self.critic = Value(architecture, activation)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.param.LEARNING_RATE)

        self.rolloutBuffer = RolloutBuffer()

        self.steps = 0
        self.done = False
    
    def act(self, state):
        ''' '''
        steps = self.param.BATCH_SIZE
        reward = [0]*steps
        done = [0]*steps
        for i in range(steps):
            policy = self.actor(torch.from_numpy(state).float())
            action = policy.sample() 
            next_state, reward[i], done[i], _ = self.env.step(action.numpy()) 
            mask = 1-done[i]
            self.rolloutBuffer.push(state, action, reward[i], next_state, mask)
            self.steps += 1
            if done[i]:
                state = self.env.reset()
            else:
                state = next_state
        # self.done = done  
        return next_state, reward, done


    def learn(self):
        for _ in range(4):
            batch = self.rolloutBuffer.sample()
            states = torch.Tensor(batch.state)
            actions = torch.stack(batch.action)
            rewards = torch.Tensor(batch.reward)
            next_states = torch.Tensor(batch.next_state)
            mask = torch.Tensor(batch.mask)

            
            advantages, values, returns = self.generaized_advantage_estimation(states, next_states[-1], rewards, mask)
            advantages = (advantages - advantages.mean()) / advantages.std()
            
            # Optimizer Critic
            self.critic_optim.zero_grad()
            # value_loss = (values - returns).pow(2.).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            value_loss.backward()
            self.critic_optim.step()

            curr_policy = self.actor(states)
            curr_log_probs = curr_policy.log_prob(actions).squeeze()
            old_policy = self.actor(states, old=True)
            old_log_probs = old_policy.log_prob(actions).squeeze()
            self.actor.backup()

            # Optimize Policy
            self.actor_optim.zero_grad()
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio*advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.8,  1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
            self.actor_optim.step()

        del self.rolloutBuffer.memory[:]
        

    
    def bootstrap_target_estimations(self, next_states, rewards, mask):
        pass
        # with torch.no_grad():
        #     targets = rewards + self.param.GAMMA * self.critic(next_states).squeeze()
        # return targets


    def generaized_advantage_estimation(self, states, next_state ,rewards, mask):
        '''  Generaized Advantage Estimation '''
        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_value = self.critic(next_state)

        values = torch.cat((values, next_value))
        returns = [0] * (len(rewards)+1)
        advantages = [0] * (len(rewards)+1)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.param.GAMMA * values[t+1] * mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * mask[t] * advantages[t+1]
            returns[t] = rewards[t] + self.param.GAMMA * returns[t+1] * mask[t]
        return torch.stack(advantages[:-1]).detach(), values[:-1], torch.stack(returns[:-1]).squeeze()


    def monte_carlo_advantage_estimation(self, states, rewards, mask):
        pass
        # with torch.no_grad():
        #     values = self.critic(states)
        # Q = 0
        # returns = []
        # for t in reversed(range(len(rewards))):
        #     Q = rewards[t] + self.param.GAMMA * Q * mask[t]
        #     returns.insert(0, Q)   
        # advantages = torch.Tensor(returns) - values * mask
        # return advantages


    def log_prob(self, action, mean, std):
        pi = numpy.pi
        var = std.pow(2).squeeze()
        mean = mean.squeeze()
        return -0.5 * (2 * pi * var).log() - (action - mean).pow(2)/(2*var)  


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)
