# import os
# import mujoco_py


# mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)
# print(sim.data.qpos)
# sim.step()
# print(sim.data.qpos)

# import gym
# import numpy as np
# from itertools import count
# env = gym.make("InvertedPendulum-v2")
# observation = env.reset()
# # q_pos = np.array([0.0,3.0])
# # q_vel = np.array([0.0,0.0])
# # env.set_state(q_pos, q_vel)
# # observation = np.array(q_pos, q_vel)
# env.render()
# for t in count():
#   # env.render()
#   action = env.action_space.sample()
#   observation, reward, done, info = env.step(action)
#   # if done:
#   #   # observation = env.reset()
#   #   env.set_state(q_pos, q_vel)
#   #   observation = np.array(q_pos, q_vel)

# # env.close()


# import torch
# import torch.distributions as dist

# mean = torch.Tensor([1]) 
# std = torch.Tensor([[0.1]])

# torch.manual_seed(0)

# # p1 = dist.Normal(mean, std)
# p2 = dist.MultivariateNormal(mean, std)

# print(p2.sample())

# import gym
# import pybulletgym.envs 

# env = gym.make("HumanoidPyBulletEnv-v0")
# env.render()
# env.reset()

# for _ in range(10000):
#     env.step(env.action_space.sample()) # take a random action
# env.close()

import time
import gym
import pybulletgym

env = gym.make('InvertedPendulumPyBulletEnv-v0')
env.render()
env.reset()
while True:
    time.sleep(0.01)
    env.step(env.action_space.sample())
    