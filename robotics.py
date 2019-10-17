import gym 
import dm_control2gym


# env = gym.make('FetchReach-v1')
# env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
# env = gym.wrappers.Monitor(env, "data/gym-results", force=True)


# make the dm_control environment
env = dm_control2gym.make(domain_name="cartpole", task_name="balance")

# use same syntax as in gym
env.reset()
for t in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    env.render()