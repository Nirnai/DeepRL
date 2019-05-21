import itertools
import gym 
import torch
from algorithms import PPO
from evaluation import Evaluation
from hyperparameter import HyperParameter



if __name__ == '__main__':

    # Test ssh
    # Environment
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v0')
    # env.spec.reward_threshold = -200
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
    # evaluator.evaluate_algorithm(alg, param, env, 'test/output_data', seed=1)

    state = env.reset()
    for t in itertools.count():
        # Act
        state, reward, done = alg.act(state)
        # Log
        is_solved, episode = evaluator.process(reward, done)
        # Train/Finish
        alg.learn()
        # Visualization
        evaluator.show_progress(interval=1)
        if is_solved:
            evaluator.generate_results('test/output_data')
            break
            # param.save_parameters(filepath)
    state = env.reset()
    while True:
        env.render()
        policy = alg.actor(torch.from_numpy(state).float())
        action = policy.sample() 
        next_state, _, done, _ = env.step(action.numpy())
        if done:
            state = env.reset()
        else:
            state = next_state 