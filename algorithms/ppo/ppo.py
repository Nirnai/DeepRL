import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from algorithms.utils import ActorCritic, Policy, Value
from algorithms.utils import RolloutBuffer
from algorithms.utils import getEnvInfo
from collections import namedtuple

import time

Transition = namedtuple('Transition', ('states', 'actions' ,'rewards', 'next_states', 'mask'))


class PPO():
    def __init__(self, env, param):
        self.name = "PPO"
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
    
    def act(self, state, exploit=False):
        ''' '''
        policy = self.actor(torch.from_numpy(state).float())
        if exploit:
            action = policy.mean.detach()
        else:
            action = policy.sample()
        next_state, reward, done, _ = self.env.step(action.numpy()) 
        self.rolloutBuffer.push(state, action, reward, next_state, done)
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done


    def learn(self):
        if self.steps % self.param.BATCH_SIZE == 0:
            for _ in range(4):
                batch = self.rolloutBuffer.sample()
                states = torch.Tensor(batch.state)
                actions = torch.stack(batch.action)
                rewards = torch.Tensor(batch.reward)
                next_states = torch.Tensor(batch.next_state)
                mask = torch.Tensor(batch.mask)

                advantages = self.gae(states, next_states[-1], rewards, mask)
                
                # Optimizer Critic
                self.critic_optim.zero_grad()
                value_loss = advantages.pow(2.).mean()
                value_loss.backward()
                self.critic_optim.step()

                advantages = (advantages - advantages.mean()) / advantages.std()
                curr_policy = self.actor(states)
                curr_log_probs = curr_policy.log_prob(actions).squeeze()
                old_policy = self.actor(states, old=True)
                old_log_probs = old_policy.log_prob(actions).squeeze()
                self.actor.backup()

                # Optimize Actor
                self.actor_optim.zero_grad()
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio*advantages.detach()
                surr2 = torch.clamp(ratio, 1.0 - 0.8,  1.2) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss.backward()
                self.actor_optim.step()

            del self.rolloutBuffer.memory[:]


    def gae(self, states, next_state ,rewards, mask):
        '''  Generaized Advantage Estimation '''
        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_value = self.critic(next_state)
        values = torch.cat((values, next_value))
        advantages = [0] * (len(rewards)+1)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.param.GAMMA * values[t+1] * mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * mask[t] * advantages[t+1]
        return torch.stack(advantages[:-1])


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)
