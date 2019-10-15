import numpy as np
import torch
import matplotlib.pyplot as plt

import gym 
import envs
from utils.policies import GaussianPolicy
from utils.env import getEnvInfo
from algorithms.hyperparameter import HyperParameter


env = gym.make('CartpoleSwingup-v0')
state_dim, action_dim = getEnvInfo(env)

param = HyperParameter(path='algorithms/ppo/parameters.json')
models = ['value', 'qvalue', 'policy']        
for model in models: 
    if(hasattr(param, model)):
        attr = getattr(param, model)
        attr['STATE_DIM'] = state_dim
        attr['ACTION_DIM'] = action_dim
        if 'ARCHITECTURE' in attr.keys():
            attr['ARCHITECTURE'].insert(0, state_dim)
            attr['ARCHITECTURE'].append(action_dim)
activations = ['Tanh','ReLU']
initializations = ['default','xavier','kaiming','orthogonal']

for f in activations:
    plt.figure()
    for init in initializations:
        param.policy['ACTIVATION'] = f
        param.policy['INIT_HIDDEN'] = init
        param.policy['INIT_OUTPUT'] = init
        actor = GaussianPolicy(param.policy, 'cpu')

        states = []
        for i in range(1000):
            states.append(env.reset())
        states = np.stack(states)
        with torch.no_grad():
            states = torch.from_numpy(states).float()
            actions = actor(states).squeeze()
            log_probs = actor.log_prob(states, actions)
            entropy = torch.mean(-log_probs)
        plt.hist(actions.numpy(), bins='auto', label='{}_{}'.format(f,init))
    plt.legend()
# plt.figure()
# x = ['ppo', 'sac']
# plt.bar(x, [entropy1, entropy3])

# actions = [actions1, actions2, actions3, actions4]
# for action in actions:
#     plt.figure()
#     plt.hist(action.numpy(), bins='auto')

# actions = []
# for i in range(1000):
#     actions.append(env.action_space.sample())
# actions = np.stack(actions)
# plt.figure()
# plt.hist(actions, bins='auto')

plt.show()
