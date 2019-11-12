import sys
import os
import time
import torch
import gym
import dm_control2gym
import utils
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from algorithms import TRPO, TD3, SAC, PPO, CGP
from algorithms import HyperParameter

env = dm_control2gym.make(domain_name='cartpole', task_name='swingup')
states = []
actions = []
for _ in range(10):
    states.append(env.reset())
    for _ in range(999):
        a = env.action_space.sample()
        s, _, _, _ = env.step(a)
        states.append(s)
        actions.append(a)
states = np.stack(states)
actions = np.stack(actions)

algs = [CGP]
for alg in algs:
    plt.figure()
    params_path = os.path.abspath(sys.modules[alg.__module__].__file__).split('/')
    params_path[-1] = 'parameters.json'
    params_path = '/'.join(params_path)

    inits = ['naive', 'kaiming', 'orthogonal']
    for init in inits:  
        hist = np.zeros((100,))
        bias = 0
        for i in range(10):
            param = HyperParameter(path=params_path)
            param.policy['INIT'] = init
            param.qvalue['INIT'] = init
            agent = alg(env, param=param)  
            with torch.no_grad():
                if agent.name == 'CGP':
                    start = time.clock()
                    a = agent.actor_cem(torch.from_numpy(states).float())
                    elapsed = time.clock()
                    elapsed = elapsed - start
                    print("Time: {}".format(elapsed))
                else:
                    a = agent.actor(torch.from_numpy(states).float())
                    if agent.name == 'TD3':
                        a += torch.randn(a.shape) * param.POLICY_EXPLORATION_NOISE 
                        a = a.cpu().numpy()
                        a = np.clip(a, -1, 1)
            h, b= np.histogram(a, density=True, bins=100, range=[-1,1])
            bias += a.mean()
            hist += h
        hist = hist/10
        bias = bias/10
        print(bias)
        bins = np.linspace(-1,1,100)
        plt.title('{}'.format(agent.name))
        plt.xlabel('action')
        plt.ylabel('PDF')
        plt.plot(bins, hist, label=init)
    plt.legend()
plt.show()