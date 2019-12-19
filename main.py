import gym 
import dm_control2gym
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator
from evaluator.plot import plot_learning_curves, load_dataset

envs = [
    ('cartpole', 'swingup'),
]

algs = [TD3, CGP, SAC, TRPO, PPO]

for alg in algs:
  for domain, task in envs:
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        agent = alg(env)
        evl = Evaluator(agent, 'output_data')
        # evl.run_statistic(samples=1, seed=0)
# returns = load_dataset('path/to/returns_offline.npz')
# fig, ax = plot_learning_curves(returns, interval='bs')