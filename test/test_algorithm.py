import datetime
import os
import gym 
import torch
import random
import numpy as np
from itertools import count
from algorithms.dqn.dqn import DQN
from algorithms.reinforce.reinforce import REINFORCE
from evaluation import Evaluation
from hyperparameter import HyperParameter

def test_dqn():
    # Environment
    env = gym.make('CartPole-v1')
    # Hyperparameters
    filepath = 'algorithms/dqn/parameters.json'
    param = HyperParameter(filepath)
    
    # RL Algorithm
    alg = DQN(env, param, double=False, soft_update=True)

    # Evaluation
    evaluator = Evaluation(env, alg, episodes=20)

    # Test
    output_dir = 'test/output_data/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(output_dir)
    evaluator.evaluate_algorithm(alg, param, env, output_dir, samples=3, seed=0)


def test_ddqn():
    pass