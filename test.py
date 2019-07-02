import gym
from gym import wrappers

env = gym.make("Pendulum-v0")
# env = wrappers.Monitor(env, directory="video", force=True)
observation = env.reset()
# i = 0
# for _ in range(100):
env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     if done:
#         i+=1
#         print(i)
#         env.reset()

# env.close()
# gym.upload("/tmp/gym-results", api_key="YOUR_API_KEY")
env.close()