import os
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
# import statsmodels.stats as stats
from evaluator.statistics import power, effect_size, mean_confidance
from evaluator.plot import plot_dataset, plot_learning_curves, load_dataset, plot_final_performance, get_labels, plot_kl

## Test Bootstrap Power Analysis
# alg1 = 'data/init/naive/PPO_cartpolebalance_2019-10-19_21-54/returns_offline.npz'
# alg2 = 'data/init/PPO_cartpolebalance_2019-10-21_19-35//returns_offline.npz'

# alg1 = load_dataset(alg1)[:,-1]
# alg2 = load_dataset(alg2)[:,-1]

# mean1 = alg1.mean()
# mean2 = alg2.mean()

# se_desired = mean1 - mean2

# std1 = alg1.std()
# std2 = alg2.std()

# effect = effect_size(mean1, mean2, std1, std2)
# pow = power(effect, 20)

# n = np.ceil((1.96**2 *0.5**2)/(se_desired * 100/ mean1 /100)**2)
# data =  'data/early Stopping/PPO_cartpoleswingup_2019-10-26_08-21/returns_offline.npz'
# plot_dataset(data)

# returns =  [ 'data/external69/init/naive/TD3_cartpoleswingup_2019-10-23_08-00/returns_offline.npz']

# labels, exp, alg, task = get_labels(returns)

# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='bs')
# # plot_final_performance(data, x=labels)


returns =  [    'data/external69/init/kaiming/TD3_cheetahrun_2019-11-02_21-38/returns_offline.npz',
                'data/external69/init/naive/TD3_cheetahrun_2019-10-31_12-11/returns_offline.npz',
                'data/external69/init/orthogonal/TD3_cheetahrun_2019-11-05_07-07/returns_offline.npz']
data = []
for path in returns:
    data.append(load_dataset(path))
plot_learning_curves(data, interval='bs')
# plot_final_performance(data, x=labels)

# returns =  [    'data/external69/init/naive/TD3_cartpoleswingup/returns_offline.npz',
#                 'data/external69/init/kaiming/TD3_cartpoleswingup/returns_offline.npz',
#                 'data/external69/init/orthogonal/TD3_cartpoleswingup/returns_offline.npz']
# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='t')


# returns =  [    'data/init/naive/TRPO_cartpolebalance_2019-10-29_11-39/kl.npz']
# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='bs')


# returns =  [    'data/init/naive/TRPO_cartpolebalance_2019-10-29_11-39/entropy.npz']
# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='bs')


# returns =  [    'data/init/naive/TRPO_cartpolebalance_2019-10-29_11-39/pg_norm.npz']
# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='bs')

# returns =  [  'data/init/xavier/PPO_cartpoleswingup/returns_offline.npz',
#             'data/early Stopping/PPO_cartpoleswingup_2019-10-26_08-21/returns_offline.npz']
# labels, exp, alg, task = get_labels(returns)
# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='t')
# plot_final_performance(data, x=labels)



# std_offline =     'data/init/naive/PPO_cartpolebalance_2019-10-19_21-54/deviation_offline.npz'
# kl = 'data/early Stopping/PPO_cartpolebalance_2019-10-25_20-38/kl.npz'
# data = load_dataset(kl)
# plot_kl([data])
# H = load_dataset(entropy)
# std = np.exp(H - 0.5 - 0.5 * math.log(2 * math.pi))

# returns2_offline = 'data/init/PPO_cartpolebalance_2019-10-23_10-44/returns_offline.npz'
# std2_offline =     'data/init/PPO_cartpolebalance_2019-10-23_10-44/deviation_offline.npz'


# plot_dataset(returns, goal=1000)
# plot_dataset(entropy,statistic='normal')
# plot(std)
# plot_dataset(kl, goal=0.01, statistic='normal')
# plot_offline([returns_offline, returns2_offline],[std_offline, std2_offline],1)


# import bootstrapped.bootstrap as bs
# import bootstrapped.stats_functions as bs_stats
# returns = 'data/external69/init/TD3_cartpolebalance_2019-10-21_11-25/returns_offline.npz'
# data = load_dataset(returns)

# plot_dataset(returns, goal=1000)
# bs_mean = np.zeros(200)
# bs_low = np.zeros(200)
# bs_high = np.zeros(200)
# for i in range(data.shape[1]):
#     temp = bs.bootstrap(data[:,i], stat_func=bs_stats.mean, is_pivotal=True)
#     bs_mean[i] = temp.value
#     bs_low[i] = temp.lower_bound
#     bs_high[i] = temp.upper_bound
# plt.figure()
# plt.plot(bs_mean)
# plt.fill_between(range(200), bs_low, bs_high, alpha=0.2)

# mean = data.mean(axis=0)
# se_low = data.mean(axis=0) - 1.96 * data.std(axis=0)/np.sqrt(20)
# se_high = data.mean(axis=0) + 1.96 * data.std(axis=0)/np.sqrt(20)
# plt.plot(range(200), data.mean(axis=0))
# plt.fill_between(range(200), se_low, se_high, alpha=0.2)

# mean = data.mean(axis=0)
# std_low = data.mean(axis=0) - data.std(axis=0)
# std_high = data.mean(axis=0) + data.std(axis=0)
# plt.plot(range(200), data.mean(axis=0))
# plt.fill_between(range(200), std_low, std_high, alpha=0.2)

# median = np.median(data,axis=0)
# perc_low = np.percentile(data, 5 ,axis=0)
# perc_high = np.percentile(data, 95 ,axis=0)
# plt.plot(range(200), median)
# plt.fill_between(range(200), perc_low, perc_high, alpha=0.2)

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



