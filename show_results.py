import os
import gym
import matplotlib.pyplot as plt
import numpy as np
from evaluator.plot import plot_dataset, compare_datasets, plot, load_dataset, plot_offline, plot_final_performance

data1 = 'data/PPO_CartpoleSwingup-v0_2019-10-16_00-00/returns_online.npz'
data2 = 'data/PPO_CartpoleSwingup-v0_2019-10-15_19-18/returns_online.npz'
data3 = 'data/PPO_cartpoleswingup_2019-10-17_12-18/returns_online.npz'

compare_datasets([data1, data2, data3],goal=1000)

plot_dataset(data3, goal=1000)

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

data4 = 'data/PPO_CartpoleSwingup-v0_2019-10-16_00-00/returns_offline.npz'
data5 = 'data/PPO_CartpoleSwingup-v0_2019-10-16_00-00/deviation_offline.npz'
data6 = 'data/PPO_CartpoleSwingup-v0_2019-10-15_19-18/returns_offline.npz'
data7 = 'data/PPO_CartpoleSwingup-v0_2019-10-15_19-18/deviation_offline.npz'
data8 = 'data/PPO_cartpoleswingup_2019-10-17_12-18/returns_offline.npz'
data9 = 'data/PPO_cartpoleswingup_2019-10-17_12-18/deviation_offline.npz'
plot_offline([data4, data6, data8],[data5, data7, data9],1)


# data13 = 'data/PPO_CartpoleSwingup-v0_2019-10-16_00-00/explained_variance.npz'
# data14 = 'data/PPO_CartpoleSwingup-v0_2019-10-16_12-26/explained_variance.npz'
# compare_datasets([data13,data14],goal=1)

plt.show()



