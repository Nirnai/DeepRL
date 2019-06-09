import torch
from algorithms import onPolicy, BaseRL, HyperParameter
from utils.models import Policy
from utils.env import getEnvInfo
from itertools import accumulate

class Test(BaseRL):
    def __init__(self, env):    
        super().__init__(env)    
        self.name = "Test"
    
    def act(self, state):
        action = self.env.action_space.sample()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.push(state, torch.Tensor(action), reward, next_state, done)
        self.steps += 1
        return next_state, reward, done

    @onPolicy
    def learn(self):
        rollouts = self.onPolicyData
        print(rollouts)