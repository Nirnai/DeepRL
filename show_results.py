import gym
import pybulletgym
import matplotlib.pyplot as plt
from evaluator.plot import plot_dataset, compare_datasets, plot_action

# ## SAC, TD3, CGP
# data = ['data/benchmark/SAC_Pendulum-v0_returns.npz',
#         'data/benchmark/TD3_Pendulum-v0_returns.npz',
#         'data/benchmark/CGP_Pendulum-v0_returns.npz']
# fig1 = compare_datasets(data, goal=-250, show=False)
# fig1.suptitle('SAC vs TD3 vs CGP')

# ## SAC, TRPO, PPO
# data = ['data/benchmark/TRPO_Pendulum-v0_returns.npz',
#         'data/benchmark/PPO_Pendulum-v0_returns.npz',
#         'data/benchmark/SAC_Pendulum-v0_returns_long.npz',
#         'data/benchmark/CGP_Pendulum-v0_returns_long.npz']
# fig2 = compare_datasets(data, goal=-250, show=False)
# fig2.suptitle('TRPO vs PPO vs SAC')

# ## Evaluation: Online vs. Offline
# data = ['data/ppo/online/PPO_Pendulum-v0_returns.npz',
#         'data/ppo/offline/PPO_Pendulum-v0_returns.npz']
# fig3 = compare_datasets(data, goal=-250, show=False)
# fig3.suptitle('PPO Online vs Offline')

# data = ['data/trpo/online/TRPO_Pendulum-v0_returns.npz',
#         'data/trpo/offline/TRPO_Pendulum-v0_returns.npz']
# fig4 = compare_datasets(data, goal=-250, show=False)
# fig4.suptitle('TRPO Online vs Offline')



# ## Normalization vs Non-Normalized
# data = ['data/ppo/online/PPO_Pendulum-v0_returns.npz',
#         'data/ppo/normalized/PPO_Pendulum-v0_returns.npz']
# fig5 = compare_datasets(data, goal=-250, show=False)
# fig5.suptitle('PPO Normalization vs Non-Normalized')

# data = ['data/trpo/online/TRPO_Pendulum-v0_returns.npz',
#         'data/trpo/normalized/TRPO_Pendulum-v0_returns.npz']
# fig6 = compare_datasets(data, goal=-250, show=False)
# fig6.suptitle('TRPO Normalization vs Non-Normalized')

# ## Q-Function: Underestimation vs Overestimation (only SAC, TD3, CGP)



## Action Smoothness
# env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0')
# data1 = 'data/TRPO_InvertedPendulumSwingupPyBulletEnv-v0_actions.npz'
# data2 = 'data/SAC_InvertedPendulumSwingupPyBulletEnv-v0_actions.npz'
# data3 = 'data/TRPO_Pendulum-v0_actions.npz'
# # plot_action(data1, env)
# plot_action(data2, env)

# Benchmark

# data = ['data/fromServer/TRPO_Pendulum-v0.npz',
#         'data/fromServer/PPO_Pendulum-v0.npz',
#         'data/fromServer/SAC_Pendulum-v0.npz',
#         'data/fromServer/TD3_Pendulum-v0.npz',
#         'data/fromServer/CGP_Pendulum-v0.npz']

# compare_datasets(data, goal=-250, show=False)

# # 1, 2 and 5 steps
# data = ['data/fromServer/OffPolicylargerBatches/SAC_Pendulum-v0_1-steps.npz',
#         'data/fromServer/OffPolicylargerBatches/SAC_Pendulum-v0_2-steps.npz',
#         'data/fromServer/OffPolicylargerBatches/SAC_Pendulum-v0_5-steps.npz']

# compare_datasets(data, goal=-250, show=False)

# # Time limit test
# data = ['data/trpo/valueMultiOptim/lambda095/TRPO_Pendulum-v0_returns.npz',
#         'data/trpo/valueSingleOptim/TRPO_Pendulum-v0_returns.npz'
#         ]

# compare_datasets(data, goal=-250, show=False)
# plt.ylim([-1500, 100])

# data = ['data/trpo/valueMultiOptim/lambda095/TRPO_Pendulum-v0_value.npz',
#         'data/trpo/valueSingleOptim/TRPO_Pendulum-v0_value.npz'
#         ]

# compare_datasets(data, goal=-250, show=False)
# plt.ylim([-1500,100])
# data = ['data/sac/SAC_CartpoleSwingup-v0_returns.npz',
#         'data/SAC_CartpoleSwingup-v0_returns.npz'
#         ]

# compare_datasets(data, goal=1000, show=False)

# data = 'data/TD3_CartpoleSwingup-v0_returns.npz'
# plot_dataset(data, goal=1000, show=False, statistic='normal')


# data = 'data/TD3_CartpoleSwingup-v0_value.npz'
# plot_dataset(data, goal=1000, show=False)#, statistic='normal')

data = 'data/CGP_CartpoleSwingup-v0_returns.npz'
plot_dataset(data, goal=1000, show=False)#, statistic='normal')


# data = 'data/CGP_CartpoleSwingup-v0_value.npz'
# plot_dataset(data, goal=1000, show=False)#, statistic='normal')

# data = 'data/SAC_CartpoleSwingup-v0_returns.npz'
# plot_dataset(data, goal=1000, show=False)#, statistic='normal')

# data = 'data/TRPO_CartpoleSwingup-v0_value.npz'
# plot_dataset(data, goal=1000, show=False)#, statistic='normal')



plt.show()



