import torch
from algorithms import onPolicy, BasePG, HyperParameter
from utils.models import Policy
from utils.env import getEnvInfo
from itertools import accumulate

class VPG(BasePG):
    def __init__(self, env):    
        super().__init__(env)    
        self.name = "VPG"
    
    @onPolicy
    def learn(self):
        rollouts = self.onPolicyData
        # Monte Carlo estimate of returns
        R = self.mc_returns(rollouts.reward, rollouts.mask)
        # Log Probability of current policy
        log_probs = self.policy.log_probs(rollouts.state, rollouts.action)
        # Compute Loss L = -log(π(a,s)) * R
        loss = (-log_probs * R).sum()
        # Optimize actor
        self.pg_step(loss)
        
    def pg_step(self, loss):
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

    def mc_returns(self, rewards, masks):
        returns = [0] * len(rewards)
        for t in reversed(range(len(rewards[:-1]))):
            returns[t] = rewards[t] + self.param.GAMMA * masks[t] * returns[t+1]
        return torch.Tensor(returns)
    