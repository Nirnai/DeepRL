import itertools
import gym 
from algorithms import DQN, PG, A2C
from evaluation import Evaluation
from hyperparameter import HyperParameter

if __name__ == '__main__':

    # Test ssh
    # Environment
    env = gym.make('CartPole-v1')
    # Hyperparameters
    filepath = 'algorithms/pg/parameters.json'
    param = HyperParameter(filepath)
    
    # RL Algorithm
    alg = PG(env, param)
    # alg = DQN(env, param)
    # alg = A2C(env, param)

    # Evaluation
    evaluator = Evaluation(env, alg, episodes=1000)
    # evaluator.generate_statistic('results')
    evaluator.evaluate_algorithm(alg, param, env, 'test/output_data')

    # state = env.reset()

    # for t in itertools.count():
    #     # Act
    #     state, reward, done = alg.act(state)

    #     # Log
    #     is_solved = evaluator.process(reward, done)

    #     # Train/Finish
    #     if is_solved:
    #         evaluator.generate_results('results')
    #         param.save_parameters(filepath)
    #         env.render()
    #     else:
    #         # TODO: done and t --> unify learn interface
    #         alg.learn()

    #     if done:
    #         state = env.reset()
    #         evaluator.show_progress()