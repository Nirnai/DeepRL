import gym 
import itertools


env = gym.make('CartPole-v1')
# env = gym.make('Pendulum-v0')
# env = gym.make('MountainCar-v0')
env.reset()

for t in itertools.count():
    env.render()
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        env.reset()