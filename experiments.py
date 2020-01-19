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
        evl.run_statistic(samples=10, seed=0)


def init(alg):
    params_path = os.path.abspath(sys.modules[alg.__module__].__file__).split('/')
    params_path[-1] = 'parameters.json'
    params_path = '/'.join(params_path)
    param = HyperParameter(path=params_path)
    if param.policy['ACTIVATION'] == 'Tanh':
        modes = ['naive', 'xavier','orthogonal']
    elif param.policy['ACTIVATION'] == 'ReLU':
        modes = ['naive','kaiming', 'orthogonal']
    for domain, task in envs:
        param = HyperParameter(path=params_path)
        for mode in modes:
            param.policy['INIT'] = mode
            if hasattr(param, 'value'): 
                param.value['INIT'] = mode
            elif hasattr(param, 'qvalue'):
                param.qvalue['INIT'] = mode
            env = dm_control2gym.make(domain_name=domain, task_name=task)
            agent = alg(env, param=param)
            evl = Evaluator(agent, 'data/init/'+ mode)
            evl.run_statistic(samples=10, seed=0)

def pretraining(alg):
    params_path = os.path.abspath(sys.modules[alg.__module__].__file__).split('/')
    params_path[-1] = 'parameters.json'
    params_path = '/'.join(params_path)
    for domain, task in envs:
        param = HyperParameter(path=params_path)
        param.DELAYED_START = 10000
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        agent = alg(env, param=param)
        evl = Evaluator(agent, 'data/delayedStart/')
        evl.run_statistic(samples=10, seed=0)

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
    baseline(PPO, 'testOnlyCutoff')
    # init(TRPO)
    # pretraining(TD3)
    # normalize(TRPO)

