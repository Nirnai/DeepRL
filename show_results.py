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

returns = [ 'data/init/orthogonal/CGP_cartpoleswingup/returns_offline.npz',
            'data/external5/init/kaiming/CGP_cartpoleswingup_2019-11-29_01-33/returns_offline.npz',
            # 'data/external139/NEWnormalize/TD3_cheetahrun_2019-11-29_15-26/returns_offline.npz',
            # 'data/external5/init/kaiming/CGP_cartpolebalance_2019-11-28_18-16/returns_offline.npz',
            # 'data/external5/adaptive/bgd/none/PPO_cartpoleswingup_2019-11-19_11-29/returns_offline.npz',
            # 'data/adaptive/Only-AdvNorm/PPO_cartpoleswingup_2019-11-15_15-44/returns_online.npz',
            # 'data/external69/adaptive/AdvNorm/PPO_cartpoleswingup_2019-11-21_19-22/returns_offline.npz',
            # 'data/external69/adaptive/AdvNorm_ClipGrad/PPO_cartpoleswingup_2019-11-19_19-34/returns_offline.npz',
            # 'data/external5/adaptive/klstopping/PPO_cartpoleswingup_2019-11-20_17-44/returns_offline.npz',
            # 'data/adaptive/KLCutoff/PPO_cartpoleswingup_2019-11-20_17-43/returns_offline.npz',
            # 'data/init/naive/TRPO_cartpolebalance/returns_offline.npz',
            # 'data/init/orthogonal/TD3_cartpolesactor_model_
]
# data = []   
# for path in returns:
#     data.append(load_dataset(path))
# fig, ax = plot_learning_curves(data, interval='bs')

# plot_dataset('data/normalize/TRPO_cartpolebalance_2019-11-18_17-35/returns_offline.npz')
# plt.ylim([-100,1000])
# returns = [ 'data/normalize/TRPO_cheetahrun_2019-11-11_08-05/returns_online.npz',
#             'data/init/orthogonal/PPO_cheetahrun_2019-11-12_07-46/returns_online.npz',
#             'data/init/orthogonal/PPO_cheetahrun_2019-11-12_08-28/returns_online.npz']
# data = []
# for path in returns:
#     data.append(load_dataset(path))   
# fig, ax = plot_learning_curves(data, interval='bs')

# ##### Init #####
# algs = ['TD3', 'TRPO', 'PPO']
# envs = ['cartpolebalance', 'cartpoleswingup', 'acrobotswingup']
# for alg in algs:
#     for env in envs:
#         returns =  [    'data/init/naive/{}_{}/returns_offline.npz'.format(alg, env),
#                         'data/init/xavier/{}_{}/returns_offline.npz'.format(alg, env),
#                         'data/init/orthogonal/{}_{}/returns_offline.npz'.format(alg, env)]
#         data = []
#         labels = ['naive', 'xavier/kaiming', 'orthogonal']
#         for path in returns:
#             data.append(load_dataset(path))
#         fig, ax = plot_learning_curves(data, interval='t')
#         plt.legend(iter(ax.lines), labels)
#         plt.title('{}-{}'.format(alg,env))



# #### Late Start ####
# returns = [ #'data/init/xavier/PPO_cartpolebalance/returns_offline.npz',
#             'data/delayedStart/PPO_cartpolebalance_2019-11-06_08-24/returns_offline.npz']
# data = []
# for path in returns:
#     data.append(load_dataset(path))
# plot_learning_curves(data, interval='t')


# ### PG-Norm and KL-Div ### 
paths = ['/home/nirnai/Cloud/Uni/TUM/MasterThesis/data/init/naive/TRPO_cartpoleswingup/kl.npz',]
        #  '/home/nirnai/Cloud/Uni/TUM/MasterThesis/data/adaptive/Basic/PPO_cartpoleswingup/kl.npz',]
data = []
for path in paths:
    data.append(load_dataset(path))
plot_kl(data)
plt.ylim([0,0.3])


plt.show()



