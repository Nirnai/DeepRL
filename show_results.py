import os
import gym
import matplotlib.pyplot as plt
import numpy as np
from evaluator.plot import plot_dataset, compare_datasets, plot, load_dataset, plot_offline, plot_final_performance

# mean = 'data/PPO_CartpoleSwingup-v0_final_returns.npz'
# std = 'data/PPO_CartpoleSwingup-v0_final_deviation.npz'
# mean = load_dataset(mean)
# std = load_dataset(std)
# print(mean)
# print(std)

# mean = 'data/PPO_CartpoleSwingup-v0_robust_returns.npz'
# std = 'data/PPO_CartpoleSwingup-v0_robust_deviation.npz'
# mean = load_dataset(mean)
# std = load_dataset(std)
# print(mean)
# print(std)

# mean_files = [ 'data/ppo/Cartpole/policyNetwork/hidden8/PPO_CartpoleSwingup-v0_final_returns.npz',
#                'data/ppo/Cartpole/policyNetwork/hidden16/PPO_CartpoleSwingup-v0_final_returns.npz',
#                'data/ppo/Cartpole/policyNetwork/hidden32/PPO_CartpoleSwingup-v0_final_returns.npz',
#                'data/ppo/Cartpole/policyNetwork/hidden64/PPO_CartpoleSwingup-v0_final_returns.npz',
#                'data/ppo/Cartpole/policyNetwork/hidden128/PPO_CartpoleSwingup-v0_final_returns.npz',
#                'data/ppo/Cartpole/policyNetwork/hidden256/PPO_CartpoleSwingup-v0_final_returns.npz'
#                ]

# std_files = ['data/ppo/Cartpole/policyNetwork/hidden8/PPO_CartpoleSwingup-v0_final_deviation.npz',
#              'data/ppo/Cartpole/policyNetwork/hidden16/PPO_CartpoleSwingup-v0_final_deviation.npz',
#              'data/ppo/Cartpole/policyNetwork/hidden32/PPO_CartpoleSwingup-v0_final_deviation.npz',
#              'data/ppo/Cartpole/policyNetwork/hidden64/PPO_CartpoleSwingup-v0_final_deviation.npz',
#              'data/ppo/Cartpole/policyNetwork/hidden128/PPO_CartpoleSwingup-v0_final_deviation.npz',
#              'data/ppo/Cartpole/policyNetwork/hidden256/PPO_CartpoleSwingup-v0_final_deviation.npz'
#              ]

# x = ['8','16','32','64','128','256']
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


# # Late Start
# data = ['data/ppo/Cartpole/benchmark/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         'data/ppo/Cartpole/startFrom50000/PPO_CartpoleSwingup-v0_returns_offline.npz'
# ]

# compare_datasets(data, goal=1000, show=False)


# data = ['data/ppo/Cartpole/policyNetwork/hidden8/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         'data/ppo/Cartpole/policyNetwork/hidden16/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         'data/ppo/Cartpole/policyNetwork/hidden32/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         'data/ppo/Cartpole/policyNetwork/hidden64/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         'data/ppo/Cartpole/policyNetwork/hidden128/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         'data/ppo/Cartpole/policyNetwork/hidden256/PPO_CartpoleSwingup-v0_returns_offline.npz',
#         ]

# compare_datasets(data, goal=1000, show=False)
# data = [ 'data/ppo/Cartpole/policyNetwork/hidden32/PPO_CartpoleSwingup-v0_returns_online.npz',
#          'data/PPO_CartpoleSwingup-v0_returns_online.npz'
#         ]
# compare_datasets(data, goal=1000, show=False)


# data1 = 'data/ppo/CartpoleNewImpl/LRDecay/PPO_CartpoleSwingup-v0_returns_offline.npz'
# data2 = 'data/ppo/CartpoleNewImpl/LRDecay+NormAdv/PPO_CartpoleSwingup-v0_returns_offline.npz'
# data3 = 'data/ppo/CartpoleNewImpl/all/PPO_CartpoleSwingup-v0_returns_offline.npz'
# data4 = 'data/ppo/CartpoleNewImpl/LRDecay+GradNormClip/PPO_CartpoleSwingup-v0_returns_offline.npz'
# compare_datasets([data1, data2, data3, data4], goal=1,show=False)

data1 = 'data/CGP_CartpoleSwingup-v0_returns_online.npz'
data2 = 'data/cgp/CGP_CartpoleSwingup-v0_returns_online.npz'
compare_datasets([data1, data2],goal=1000)
mean = 'data/CGP_CartpoleSwingup-v0_final_returns.npz'
std = 'data/CGP_CartpoleSwingup-v0_final_deviation.npz'
x = [0]
mu = load_dataset(mean)
sigma = load_dataset(std)
print(mu, sigma)
# plot_dataset(data, goal=1000, show=False, statistic='normal')
# plot_dataset(data1, goal=1000, show=False, statistic='normal')
# plot_dataset(data3, goal=1000, show=False, statistic='normal')
# plt.ylim([-1,1])
# # plot_dataset(data2, goal=1000, show=False, statistic='normal')

# returns = 'data/PPO_CartpoleSwingup-v0_returns_offline.npz'
# deviations = 'data/PPO_CartpoleSwingup-v0_deviation_offline.npz'
# plot_offline(returns, deviations, 10)

# returns = ['data/PPO_CartpoleSwingup-v0_final_returns.npz',
#            'data/ppo/Cartpole/startFrom50000/PPO_CartpoleSwingup-v0_final_returns.npz']
# deviations = ['data/PPO_CartpoleSwingup-v0_final_deviation.npz',
#               'data/ppo/Cartpole/startFrom50000/PPO_CartpoleSwingup-v0_final_deviation.npz']
# plot_final_performance(returns, deviations, 10)


# ## Analysing Network sweep
# path =  [
#         'data/value_architecture/trpo/4/TRPO_CartpoleSwingup-v0_explained_variance.npz',
#         'data/value_architecture/trpo/8/TRPO_CartpoleSwingup-v0_explained_variance.npz',
#         'data/value_architecture/trpo/16/TRPO_CartpoleSwingup-v0_explained_variance.npz',
#         'data/value_architecture/trpo/32/TRPO_CartpoleSwingup-v0_explained_variance.npz',
#         # 'data/value_architecture/trpo/64/TRPO_CartpoleSwingup-v0_explained_variance.npz',
#         # 'data/value_architecture/trpo/128/TRPO_CartpoleSwingup-v0_explained_variance.npz',
#         # 'data/value_architecture/trpo/256/TRPO_CartpoleSwingup-v0_explained_variance.npz'
#         ]
# compare_datasets(path, goal=1,show=False)


# returns =   [
#             'data/value_architecture/trpo/4/TRPO_CartpoleSwingup-v0_returns_offline.npz',
#             'data/value_architecture/trpo/8/TRPO_CartpoleSwingup-v0_returns_offline.npz',
#             'data/value_architecture/trpo/16/TRPO_CartpoleSwingup-v0_returns_offline.npz',
#             'data/value_architecture/trpo/32/TRPO_CartpoleSwingup-v0_returns_offline.npz',
#             'data/value_architecture/trpo/64/TRPO_CartpoleSwingup-v0_returns_offline.npz',
#             'data/value_architecture/trpo/128/TRPO_CartpoleSwingup-v0_returns_offline.npz',
#             'data/value_architecture/trpo/256/TRPO_CartpoleSwingup-v0_returns_offline.npz'
#             ]

# deviations =    [
#                 'data/value_architecture/trpo/4/TRPO_CartpoleSwingup-v0_deviation_offline.npz',
#                 'data/value_architecture/trpo/8/TRPO_CartpoleSwingup-v0_deviation_offline.npz',
#                 'data/value_architecture/trpo/16/TRPO_CartpoleSwingup-v0_deviation_offline.npz',
#                 'data/value_architecture/trpo/32/TRPO_CartpoleSwingup-v0_deviation_offline.npz',
#                 'data/value_architecture/trpo/64/TRPO_CartpoleSwingup-v0_deviation_offline.npz',
#                 'data/value_architecture/trpo/128/TRPO_CartpoleSwingup-v0_deviation_offline.npz',
#                 'data/value_architecture/trpo/256/TRPO_CartpoleSwingup-v0_deviation_offline.npz'
#                 ]
# plot_offline(returns, deviations, 10)


plt.show()



