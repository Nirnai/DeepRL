import gym 
import envs
from envs import NormalizeWrapper
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator, plot_dataset

import numpy as np
import torch
import matplotlib.pyplot as plt

env = gym.make('CartpoleSwingup-v0')
alg1 = PPO(env)
alg2 = TD3(env)
alg3 = SAC(env)
alg4 = CGP(env)

states = []
for i in range(1000):
    states.append(env.reset())
states = np.stack(states)
with torch.no_grad():
    actions1 = alg1.actor(torch.from_numpy(states).float()).squeeze()
    log_probs1 = alg1.actor.log_prob(torch.from_numpy(states).float(), actions1)
    entropy1 = torch.mean(-log_probs1)
    actions2 = alg2.actor(torch.from_numpy(states).float()).squeeze()
    actions3 = alg3.actor(torch.from_numpy(states).float()).squeeze()
    log_probs3 = alg3.actor.log_prob(torch.from_numpy(states).float(), actions3)
    entropy3 = torch.mean(-log_probs3)
    actions4 = alg4.actor(torch.from_numpy(states).float()).squeeze()
    

plt.figure()
x = ['ppo', 'sac']
plt.bar(x, [entropy1, entropy3])

actions = [actions1, actions2, actions3, actions4]
for action in actions:
    plt.figure()
    plt.hist(action.numpy(), bins='auto')

# actions = []
# for i in range(1000):
#     actions.append(env.action_space.sample())
# actions = np.stack(actions)
# plt.figure()
# plt.hist(actions, bins='auto')
plt.show()
