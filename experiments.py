import os
import sys
import gym 
import dm_control2gym
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator
from algorithms import HyperParameter

envs = [
    ('cartpole', 'balance'),
    ('cartpole', 'swingup'),
    ('acrobot', 'swingup'),
    ('cheetah', 'run'),
    ('hopper', 'hop'),
    ('walker', 'run')
]

sparse = [
    ('cartpole', 'balance_sparse'),
    ('cartpole', 'swingup_sparse'),
    ('acrobot', 'swingup_sparse')
]

algs = [TD3, CGP, SAC, TRPO, PPO]

def environments():
    for domain, task in envs:
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        print("Actions Bounds: [{},{}]".format(env.action_space.low, env.action_space.high))

def baseline(alg, directory):
    for domain, task in envs:
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        agent = alg(env)
        evl = Evaluator(agent, 'data/{}'.format(directory))
        evl.run_statistic(samples=5, seed=0)


def init(alg):
    params_path = os.path.abspath(sys.modules[alg.__module__].__file__).split('/')
    params_path[-1] = 'parameters.json'
    params_path = '/'.join(params_path)
    param = HyperParameter(path=params_path)
    if param.policy['ACTIVATION'] == 'Tanh':
        modes = ['orthogonal']
    elif param.policy['ACTIVATION'] == 'ReLU':
        modes = ['kaiming', 'orthogonal']
    for mode in modes:
        param.policy['INIT'] = mode
        if hasattr(param, 'value'): 
            param.value['INIT'] = mode
        elif hasattr(param, 'qvalue'):
            param.qvalue['INIT'] = mode
        for domain, task in envs:
            env = dm_control2gym.make(domain_name=domain, task_name=task)
            agent = alg(env, param=param)
            evl = Evaluator(agent, 'data/init/'+ mode)
            evl.run_statistic(samples=5, seed=0)

def pretraining(alg):
    params_path = os.path.abspath(sys.modules[alg.__module__].__file__).split('/')
    params_path[-1] = 'parameters.json'
    params_path = '/'.join(params_path)
    for domain, task in envs:
        param = HyperParameter(path=params_path)
        param.policy['INIT'] = 'xavier'
        if hasattr(param, 'value'): 
            param.value['INIT'] = 'xavier'
        elif hasattr(param, 'qvalue'):
            param.qvalue['INIT'] = 'xavier'
        param.DELAYED_START = 0
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        agent = alg(env, param=param)
        evl = Evaluator(agent, 'data/delayedStart/')
        evl.run_statistic(samples=20, seed=0)

def normalize(alg):
    import sys
    sys.path.insert(1, '/home/nirnai/Cloud/Uni/TUM/MasterThesis/python/baselines')
    import baselines.common.vec_env as venv
    for domain, task in envs:
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        env.num_envs = 1
        env = venv.VecNormalize(env, ob=True, ret=False)
        agent = alg(env)
        evl = Evaluator(agent, 'data/normalize')
        evl.run_statistic(samples=10, seed=0)

if __name__ == '__main__':
    # environments()
    baseline(TRPO, 'baselines')
    # init(PPO)
    # pretraining(PPO)
    # normalize(TD3)


# def ppo_experiments(env):
#     env = gym.make('CartpoleSwingup-v0')
#     state_dim, action_dim = getEnvInfo(env)

#     param = HyperParameter(path='algorithms/ppo/parameters.json')
#     models = ['value', 'qvalue', 'policy']        
#     for model in models: 
#         if(hasattr(param, model)):
#             attr = getattr(param, model)
#             attr['STATE_DIM'] = state_dim
#             attr['ACTION_DIM'] = action_dim
#             if 'ARCHITECTURE' in attr.keys():
#                 attr['ARCHITECTURE'].insert(0, state_dim)
#                 attr['ARCHITECTURE'].append(action_dim)
#     activations = ['Tanh','ReLU']
#     initializations = ['default','xavier','kaiming','orthogonal']


# def qvalue_architecture(alg):
#     param_path = 'algorithms/{}/parameters.json'.format(alg.name.lower())
#     p = HyperParameter(path=param_path)
#     archs = [50,100,200,300,500]
#     for arch in archs:
#         path = 'data/qvalue_architecture/{}/{}'.format(alg.name.lower(), str(arch))
#         p.qvalue['ARCHITECTURE'] = [arch, arch]
#         if not os.path.isdir(path):  
#             os.mkdir(path)
#         p.save_parameters(path)
#         p.save_parameters(param_path) 
#         e = Evaluator(alg, path)
#         e.run_statistic(samples=5, seed=100)

# def value_architecture(alg):
#     param_path = 'algorithms/{}/parameters.json'.format(alg.name.lower())
#     p = HyperParameter(path=param_path)
#     p.policy['ARCHITECTURE'] = [32, 32]
#     archs = [4,8,16,32,64,128,256]
#     for arch in archs:
#         path = 'data/value_architecture/{}/{}'.format(alg.name.lower(), str(arch))
#         p.value['ARCHITECTURE'] = [arch,arch]
#         if not os.path.isdir(path):  
#             os.makedirs(path)
#         p.save_parameters(path + '/parameters.json')
#         p.save_parameters(param_path) 
#         e = Evaluator(alg, path)
#         e.run_statistic(samples=5, seed=100)

# def policy_architecture(alg):
#     param_path = 'algorithms/{}/parameters.json'.format(alg.name.lower())
#     p = HyperParameter(path=param_path)
#     archs = [4,8,16,32,64,128,256]
#     for arch in archs:
#         path = 'data/policy_architecture/{}/{}'.format(alg.name.lower(), str(arch))
#         p.policy['ARCHITECTURE'] = [arch, arch]
#         if not os.path.isdir(path):  
#             os.mkdir(path)
#         p.save_parameters(path)
#         p.save_parameters(param_path) 
#         e = Evaluator(alg, path)
#         e.run_statistic(samples=5, seed=100)


# def activation_function(alg):
#     pass

# def advantage_normalization(alg):
#     pass

# def early_stopping(alg):
#     pass

# def initialization_PG(alg):
#     cases = ['none', 'xavier', 'kaiming', 10000, 50000, 100000, 200000]

# def initialization_Q(alg):
#     cases = ['none', 'xavier', 'kaiming', 'late_start', 'optimistic']

# def bootstrapping(alg):
#     pass

# def explorationPG(alg):
#     cases = ['none','small_output','max_entropy','reg_entropy']



