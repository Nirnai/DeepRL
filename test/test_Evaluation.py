import os
import pytest
import gym
import numpy as np
from evaluation import Evaluation

def test_generating_results():
    
    x1 = np.arange(200)
    x2 = np.arange(100)
    data1 = np.ones(200) * x1
    data2 = np.ones(100) * 0.5 * x2

    env = gym.make('CartPole-v0')

    class Alg():
        def __init__(self):
            self.name = 'dummy'
            self.param = None

    dummy_alg = Alg() 
    filename = "{}_{}".format(dummy_alg.name,env.spec.id)

    evaluator = Evaluation(env, dummy_alg)
    evaluator.evaluation_rewards = data1
    evaluator.generate_results('test')
    evaluator.evaluation_rewards = data2
    evaluator.results_generated = False
    evaluator.generate_results('test')

    assert(os.path.isfile('test/{}.npz'.format(filename)))
    assert(os.path.isfile('test/{}.png'.format(filename)))

    os.remove('test/{}.npz'.format(filename))
    os.remove('test/{}.png'.format(filename))





