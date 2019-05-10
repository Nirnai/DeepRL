import random
import numpy
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from itertools import accumulate

class PG():
    def __init__(self, env, param):        
        self.name = "PG"
        self.env = env
        self.param = param
        self.rng = random.Random()
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        self.policy = Policy(self.param.NETWORK_ARCHITECTURE, self.param.ACTIVATION, self.is_discrete)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.param.LEARNING_RATE)

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.rewards = []
        self.log_probs = []
        self.done = False
        

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
        next_state, reward, self.done, _ = self.env.step(action.numpy()) 

        self.log_probs.append(m.log_prob(action))
        self.rewards.append(reward)

        return next_state, reward, self.done


    def learn(self):
        if self.done:
            # Monte Carlo estimate of returns 
            Qs = self.monte_carlo_estimate(self.rewards)

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

    
class Policy(nn.Module):
    def __init__(self, state_space, action_space, architecture, activation):
        super(Policy, self).__init__()
        self.activation = getattr(nn.modules.activation, activation)()
        
        
        layers = [self.activated_layer(in_, out_, self.activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        self.layers = nn.Sequential(*layers)
        self.output = self.output_layer(architecture[-2], action_space)
            
        
    def forward(self, state):
        x = state
        if self.is_discrete:
            x = self.layers(x)
            y = self.output(x)
            return y
            

    def activated_layer(self, in_, out_, activation_):
        return nn.Sequential(
            nn.Linear(in_, out_),
            # nn.BatchNorm1d(out_, affine=False),
            activation_
        )
        
    def output_layer(self, in_, action_space):
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            return nn.Sequential(
                nn.Linear(in_, num_outputs),
                nn.Softmax()
            )
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            return nn.Sequential(nn.Linear(in_, num_outputs))
        else:
            raise NotImplementedError

