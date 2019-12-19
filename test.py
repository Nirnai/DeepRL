# import os
# import numpy as np

# base = 'data/regularization/policySmoothing/'
# # new_base = 'data/regularization/entropy/'

# paths = [
#     ['CGP_cheetahrun_2019-11-26_06-29',  'CGP_cheetahrun_2019-11-26_17-22']
# ]

# files = [
#     '/returns_offline.npz',
#     '/returns_online.npz']

# for path in paths:
#     for f in files:
#         data = []
#         for p in path:
#             data += [array for array in np.load(base + p + f).values()]
#         directory = ('_').join(path[0].split('_')[:2])
#         if not os.path.isdir(base + directory):  
#             os.makedirs(base + directory, exist_ok=True)
#         new_path = base + directory + f
#         np.savez(new_path, *data)

####### Disturbances Test
import os
import sys
import gym 
import torch
import dm_control2gym
import numpy as np
from algorithms import PPO
from evaluator import Evaluator
from algorithms import HyperParameter

envs = [
    # ('cartpole', 'balance'),
    ('cartpole', 'swingup'),
    # ('acrobot', 'swingup'),
    # ('cheetah', 'run'),
    # ('hopper', 'hop'),
    # ('walker', 'run')
]


returns = []
reg_returns = []

for domain, task in envs:
    env = dm_control2gym.make(domain_name=domain, task_name=task)
    agent = PPO(env)
    evl = Evaluator(agent, 'data/testGeneralization')
    for i in range(10):
        path = 'data/adaptive/LRSchedule_KLCutoff/PPO_{}/policies/actor_model_{}.pt'.format(domain+task,i)  
        agent.actor.load_state_dict(torch.load(path)) 
        evl.eval_robustness()
        returns.append(np.mean(evl.robust_returns))
        # evl.save_robust_returns()



for domain, task in envs:
    env = dm_control2gym.make(domain_name=domain, task_name=task)
    agent = PPO(env)
    evl = Evaluator(agent, 'data/testGeneralization')
    for i in range(10):
        path = 'data/regularization/l2/PPO_{}/policies/actor_model_{}.pt'.format(domain+task,i)  
        agent.actor.load_state_dict(torch.load(path)) 
        evl.eval_robustness()
        reg_returns.append(np.mean(evl.robust_returns))

print(returns)