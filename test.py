import dm_control2gym
from algorithms import *

envs = [
    ('cartpole', 'balance')
]

algs = [TD3, CGP, SAC, TRPO, PPO]

for alg in algs:
  for domain, task in envs:
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        agent = alg(env)