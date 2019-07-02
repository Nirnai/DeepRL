import itertools
import gym 
import torch
from algorithms import VPG, A2C, PPO, TRPO, SAC, Test
from evaluator import Evaluator, plot_dataset

from evaluation import Evaluation

if __name__ == '__main__':

    # Test ssh
    # Environment
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v0')

    # RL Algorithm
    # alg = DQN(env)
    # alg = VPG(env)
    # alg = A2C(env)
    # alg = PPO(env)
    # alg = SAC(env)
    alg = TRPO(env)
    # alg = Test(env)


    ######### New Eval
    evl = Evaluator(env, alg, total_timesteps=1e6, averaging_window=100)
    evl.evaluate('test/output_data', seed=1 , samples=10)
    plot_dataset('test/output_data/TRPO_Pendulum-v0.npz', step = 1)
    # plot_dataset('test/output_data/TRPO_Pendulum-v0.npz', step = 1, statistic='normal')
    #################

    # Evaluation
    # evaluator = Evaluation(env, alg)
    # evaluator.evaluate_algorithm(alg, env, 'test/output_data', episodes=3000, seed=1 , samples=20)
    # datasets = ['test/output_data/MidTermDatasets/PPO_Dataset/5samples/PPO_Pendulum1.npz',
    #             'test/output_data/MidTermDatasets/PPO_Dataset/5samples/PPO_Pendulum2.npz']
    # evaluator.compare_datasets(datasets)
    # evaluator.compare_statistics('test/output_data/MidTermDatasets/TRPO_Dataset/TRPO_Pendulum-v0.npz')
    # evaluator.get_max_difference('test/output_data/MidTermDatasets/PPO_Dataset/10samples/PPO_Pendulum1.npz', 10)
    # results_filename = "test/output_data/{}_{}".format(evaluator.alg_name, evaluator.env_name)
    # evaluator.generate_results(results_filename)
    # evaluator.generate_statistic('test/output_data/MidTermDatasets/PPO_Dataset/PPO_Pendulum-v0', dist='normal')

    # state = env.reset()
    # for t in itertools.count():
    #     # Act
    #     state, reward, done = alg.act(state)
    #     # Train
    #     alg.learn()
    #     # Log
    #     is_solved, episode = evaluator.process(state, reward, done)
    #     # Visualize Progress
    #     if done:
    #         evaluator.show_progress(interval=1)
    #     if is_solved:
    #         evaluator.generate_results('test/output_data')
    #         break

    # Exploit Learned Policy
    # alg.env = gym.wrappers.Monitor(env, "video", force=True)
    # state = alg.env.reset()
    # while True:
    #     alg.env.render()
    #     with torch.no_grad():
    #         state, reward, done = alg.act(state, exploit=True)
    #     is_solved, episode = evaluator.process(state,reward, done)
    #     evaluator.show_progress(interval=1)
    #     if done:
    #         break
    #         # state = env.reset()
    # alg.env.close()