# import os
# os.environ['LD_LIBRARY_PATH'] = "$LD_LIBRARY_PATH:/home/nirnai/.mujoco/mujoco200/bin"


import itertools
import gym 
try:
    import mujoco_py
except ImportError:
    pass
import torch
from algorithms import VPG, A2C, PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator, plot_dataset

# from evaluation import Evaluation

if __name__ == '__main__':

    # Test ssh
    # Environment
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v0')
    # env = gym.make('InvertedPendulum-v2')
    # env = gym.make('HalfCheetah-v2')
    

    # Mujoco
    # saved_state = env.sim.get_state()
    # env.sim.set_state(saved_state)

    # RL Algorithm
    alg = TRPO(env)
    # alg = PPO(env)
    # alg = SAC(env)
    # alg = CGP(env)
    # alg = TD3(env)

    ######### New Eval
    evl = Evaluator(alg, total_timesteps=1e6)
    evl.evaluate('data')

    data= ['data/sac/VFunction_QFunction/SAC_Pendulum-v0_returns.npz', 
           'data/sac/QFunctions/SAC_Pendulum-v0_returns.npz', 
           'data/cgp/CGP_Pendulum-v0_returns.npz']
    # plot_dataset('test/output_data/TRPO_Pendulum-v0.npz', step = 1)
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