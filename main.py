import itertools
import gym 
from algorithms import PPO
from evaluation import Evaluation
from hyperparameter import HyperParameter

if __name__ == '__main__':

    # Test ssh
    # Environment
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('MountainCar-v0')
    # Hyperparameters
    filepath = 'algorithms/ppo/parameters.json'
    param = HyperParameter(filepath)
    
    # RL Algorithm
    # alg = PG(env, param)
    # alg = DQN(env, param)
    # alg = A2C(env, param)
    alg = PPO(env, param)
    # alg = DDPG(env, param)

    # Evaluation
    evaluator = Evaluation(env, alg)
    # evaluator.generate_statistic('results')
    # evaluator.evaluate_algorithm(alg, param, env, 'test/output_data')

    state = env.reset()

    for t in itertools.count():
        # Act
        state, reward, done = alg.act(state)
        # Log
        is_solved, episode = evaluator.process(reward, done)

        # Train/Finish
        if is_solved:
            # evaluator.generate_results('results')
            # param.save_parameters(filepath)
            env.render()
            alg.learn()
        else:
            alg.learn()

        if done:
            state = env.reset()
            evaluator.show_progress()