import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from algorithms.utils import ActorCritic
from algorithms.utils import getEnvInfo
from collections import namedtuple

Transition = namedtuple('Transition', ('log_probs', 'entropy', 'rewards', 'values', 'mask'))


class PPO():
    def __init__(self, env, param):
        self.name = "A2C"
        self.env = env
        self.param = param
        self.rng = random.Random()

        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        self.param.ACTOR_ARCHITECTURE[0] = self.state_dim
        self.param.ACTOR_ARCHITECTURE[-1] = self.action_dim

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.model = ActorCritic(self.param.ACTOR_ARCHITECTURE, self.param.ACTIVATION, action_space=self.action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.param.LEARNING_RATE)

        self.rollout = []

        self.steps = 0
        self.done = False
    
    def act(self, state):
        ''' '''
        policy, value = self.model(torch.from_numpy(state).float())
        action = policy.sample() 
        next_state, reward, done, _ = self.env.step(action.numpy()) 

        self.rollout.append(Transition(policy.log_prob(action), 
                                       policy.entropy(),
                                       reward, 
                                       value, 
                                       (1-done)))
        self.steps += 1
        self.done = done  
        self.next_state = torch.from_numpy(next_state).float()       

        return next_state, reward, done


    def learn(self):
        # if self.steps % self.param.STEPS == 0:
        if self.done:
            rollout = Transition(*zip(*self.rollout))

            rewards = torch.Tensor(rollout.rewards)
            values = torch.stack(rollout.values).squeeze()
            log_probs = torch.stack(rollout.log_probs)
            entropy = torch.stack(rollout.entropy).mean()
            mask = torch.Tensor(rollout.mask)

            # Monte Carlo Advantages
            advantages = self.monte_carlo_advantage_estimation(rewards, values, mask)
            # General Advantages
            # advantages = self.generaized_advantage_estimation(rewards, values, mask)

            critic_loss = advantages.pow(2).mean()
            actor_loss = (- log_probs * advantages.detach()).sum()

            loss = critic_loss + actor_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del self.rollout[:]

    
    def generaized_advantage_estimation(self, rewards, values, mask):
        '''  Generaized Advantage Estimation '''
        with torch.no_grad():
                _, next_value = self.model(self.next_state)

        values = torch.cat((values, next_value))
        advantages = []
        a_gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.param.GAMMA * values[t+1] * mask[t] - values[t]
            a_gae = (self.param.GAMMA * self.param.LAMBDA * mask[t]) * a_gae + delta
            advantages.insert(0, a_gae)
        return torch.Tensor(advantages)


    def monte_carlo_advantage_estimation(self, rewards, values, mask):
        Q = 0
        returns = []
        for t in reversed(range(len(rewards))):
            Q = rewards[t] + self.param.GAMMA * Q * mask[t]
            returns.insert(0, Q)   
        advantages = torch.Tensor(returns) - values * mask
        return advantages


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)
