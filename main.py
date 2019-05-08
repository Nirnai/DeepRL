import gym 
import torch
import random
import numpy as np
from itertools import count
from algorithms.dqn.dqn import DQN
from algorithms.reinforce.reinforce import REINFORCE
from evaluation import Evaluation
from hyperparameter import HyperParameter

if __name__ == '__main__':

    # Environment
    env = gym.make('CartPole-v1')
    # Hyperparameters
    filepath = 'dqn/parameters.json'
    param = HyperParameter(filepath)
    
    # RL Algorithm
    # alg = REINFORCE(env, param)
    alg = DQN(env, param, double=True, soft_update=True)

    # Evaluation
    evaluator = Evaluation(env, alg, episodes=1000)
    # evaluator.generate_statistic('results')
    # evaluator.evaluate_algorithm(alg, param, env)

    state = env.reset()

    for t in count():
        # Act
        state, reward, done = alg.act(state)

        # Log
        is_solved = evaluator.process(reward, done)

        # Train/Finish
        if is_solved:
            evaluator.generate_results('results')
            param.save_parameters(filepath)
            env.render()
        else:
            # TODO: done and t --> unify learn interface
            loss = alg.learn(done)

        if done:
            state = env.reset()
            evaluator.show_progress()