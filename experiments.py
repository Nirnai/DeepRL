import os
import gym 
import envs
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator, plot_dataset
from algorithms import HyperParameter

def qvalue_architecture(alg):
    param_path = 'algorithms/{}/parameters.json'.format(alg.name.lower())
    p = HyperParameter(path=param_path)
    archs = [50,100,200,300,500]
    for arch in archs:
        path = 'data/qvalue_architecture/{}/{}'.format(alg.name.lower(), str(arch))
        p.qvalue['ARCHITECTURE'] = [arch, arch]
        if not os.path.isdir(path):  
            os.mkdir(path)
        p.save_parameters(path)
        p.save_parameters(param_path) 
        e = Evaluator(alg, path)
        e.run_statistic(samples=5, seed=100)

def value_architecture(alg):
    param_path = 'algorithms/{}/parameters.json'.format(alg.name.lower())
    p = HyperParameter(path=param_path)
    p.policy['ARCHITECTURE'] = [32, 32]
    archs = [4,8,16,32,64,128,256]
    for arch in archs:
        path = 'data/value_architecture/{}/{}'.format(alg.name.lower(), str(arch))
        p.value['ARCHITECTURE'] = [arch,arch]
        if not os.path.isdir(path):  
            os.makedirs(path)
        p.save_parameters(path + '/parameters.json')
        p.save_parameters(param_path) 
        e = Evaluator(alg, path)
        e.run_statistic(samples=5, seed=100)

def policy_architecture(alg):
    param_path = 'algorithms/{}/parameters.json'.format(alg.name.lower())
    p = HyperParameter(path=param_path)
    archs = [4,8,16,32,64,128,256]
    for arch in archs:
        path = 'data/policy_architecture/{}/{}'.format(alg.name.lower(), str(arch))
        p.policy['ARCHITECTURE'] = [arch, arch]
        if not os.path.isdir(path):  
            os.mkdir(path)
        p.save_parameters(path)
        p.save_parameters(param_path) 
        e = Evaluator(alg, path)
        e.run_statistic(samples=5, seed=100)


def activation_function(alg):
    pass

def advantage_normalization(alg):
    pass

def early_stopping(alg):
    pass

def initialization_PG(alg):
    cases = ['none', 'xavier', 'kaiming', 10000, 50000, 100000, 200000]

def initialization_Q(alg):
    cases = ['none', 'xavier', 'kaiming', 'late_start', 'optimistic']

def bootstrapping(alg):
    pass

def explorationPG(alg):
    cases = ['none','small_output','max_entropy','reg_entropy']



if __name__ == '__main__':

    env = gym.make('CartpoleSwingup-v0')
    value_architecture(TRPO(env))