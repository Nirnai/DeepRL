import gym 
# import envs
import dm_control2gym
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator, plot_dataset


# try:
#     import mujoco_py
# except ImportError:
#     pass
# try:
#     import pybulletgym
# except ImportError:
#     pass
# from evaluation import Evaluation

if __name__ == '__main__':
    ########## Environment ###########
    # env = gym.make('CartPole-v1')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('InvertedPendulumstate-v2')
    # env = gym.make('HalfCheetah-v2')
    # env = gym.make('InvertedPendulumPyBulletEnv-v0')
    # env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0')
    # env = gym.make('PendulumSwingup-v0')
    # env = gym.make('CartpoleSwingup-v0')
    # env = gym.make('CartpoleBalance-v0')
    # env = gym.make('WalkerWalk-v0')
    # env = gym.make('HumanoidWalk-v0')
    env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
    # env = gym.make('ManipulatorBring_Ball-v0')
    # env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    ########## Algorithm ###########
    # alg = TRPO(env)
    # alg = PPO(env)
    # alg = SAC(env)
    alg = CGP(env)
    # alg = TD3(env)    

    # ########### Evaluation ###########
    evl = Evaluator(alg, 'data')
    evl.run_statistic()
    # ##################################


    ########## One Learning Loop #########    
    # state = env.reset()
    # for t in range(int(1e6)):
    #     env.render()
    #     # Act
    #     state, reward, done = alg.act(state)
    #     # Train
    #     metrics = alg.learn()
    #     if done:
    #         state = env.reset()
    #     # Log
    #     print("Timestep: {} \nReward: {} \n -----------------------".format(t, reward))
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