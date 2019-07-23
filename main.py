import itertools
import gym 
try:
    import mujoco_py
except ImportError:
    pass
try:
    import pybulletgym
except ImportError:
    pass
import torch
from algorithms import VPG, A2C, PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator, plot_dataset

# from evaluation import Evaluation

if __name__ == '__main__':

    ########## Environment ###########
    # env = gym.make('CartPole-v1')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('InvertedPendulum-v2')
    # env = gym.make('HalfCheetah-v2')
    # env = gym.make('InvertedPendulumPyBulletEnv-v0')
    env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0')


    ########## Algorithm ###########
    alg = TRPO(env)
    # alg = PPO(env)
    # alg = SAC(env)
    # alg = CGP(env)
    # alg = TD3(env)

    # ########### Evaluation ###########
    evl = Evaluator(alg, total_timesteps=1e6)
    # evl.evaluate('data')
    # ##################################


    ########## One Learning Loop #########
    env.render()
    state = env.reset()
    for t in range(int(1e6)):
        # Act
        state, reward, done = alg.act(state)
        # Train
        metrics = alg.learn()
        # Log
        evl._log_metrics(metrics)
        evl._eval_online(reward, done)
    ########## One Learning Loop #########


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