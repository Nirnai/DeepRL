import gym
from gym.envs.registration import register
from dm_control import suite
from envs.wrapper import DeepMindControlSuiteWrapper

for domain, task in suite.BENCHMARKING:
    register(
        id=domain.title() + task.title() + '-v0',
        entry_point='envs.wrapper:DeepMindControlSuiteWrapper',
        kwargs={'domain_name': domain, 'task_name': task}
    )

# gym_id_list = []

# def make(domain_name, task_name, task_kwargs=None, visualize_reward=False):

#     if task_kwargs is None:
#         gym_id = domain_name.title() + task_name.title() + '-v0'
#     else: 
#         gym_id = domain_name.title() + task_name.title() + str(task_kwargs) + '-v0'

#     if gym_id not in gym_id_list:
#         register(id=gym_id, 
#                  entry_point='envs.wrapper:DeepMindControlSuiteWrapper',
#                  kwargs={'domain_name': domain_name, 'task_name': task_name, 'task_kwargs': task_kwargs, 'visualize_reward': visualize_reward})

#     gym_id_list.append(gym_id)
#     return gym.make(gym_id)