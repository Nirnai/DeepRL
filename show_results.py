import os
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
from evaluator.plot import plot_dataset, compare_datasets, plot, load_dataset, plot_offline, plot_final_performance

returns = 'data/init/PPO_acrobotswingup_2019-10-20_22-23/returns_online.npz'
returns_offline = 'data/init/PPO_acrobotswingup_2019-10-20_22-23/returns_offline.npz'
std_offline = 'data/init/PPO_acrobotswingup_2019-10-20_22-23/deviation_offline.npz'
entropy = 'data/init/PPO_acrobotswingup_2019-10-20_22-23/entropy.npz'
kl = 'data/init/PPO_acrobotswingup_2019-10-20_22-23/kl.npz'
H = load_dataset(entropy)
std = np.exp(H - 0.5 - 0.5 * math.log(2 * math.pi))
plot_dataset(returns, goal=1000, statistic='normal')
plot_dataset(entropy,statistic='normal')
plot(std)
plot_dataset(kl, goal=0.01, statistic='normal')
plot_offline([returns_offline],[std_offline],1)

# mean_files = [ 'data/PPO_CartpoleSwingup-v0_2019-10-15_15-08/final_returns.npz',
#                'data/PPO_CartpoleSwingup-v0_2019-10-15_19-18/final_returns.npz'
#                ]

# std_files = ['data/PPO_CartpoleSwingup-v0_2019-10-15_15-08/final_deviation.npz',
#              'data/PPO_CartpoleSwingup-v0_2019-10-15_19-18/final_deviation.npz'
#              ]

# x = ['base','init']
# means = []
# stds = []
# n = 10
# for mean, std in zip(mean_files, std_files):
#         mu = load_dataset(mean)
#         sigma = load_dataset(std)
#         n_total = len(mu) * n
#         new_mean =  n/n_total * np.sum(mu[-1])
#         new_sigma = np.sqrt(n/n_total * (np.sum(sigma[-1]**2) + np.sum((mu[-1] - new_mean)**2)))
#         means.append(new_mean)
#         stds.append(new_sigma)      

# fig, ax = plt.subplots()
# ax.bar(x, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)


plt.show()



