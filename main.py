import gym 
import dm_control2gym
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator
from evaluator.plot import plot_learning_curves, load_dataset


# env = dm_control2gym.make(domain_name='cartpole', task_name='swingup')
env = gym.make('Pendulum-v0')


algs = [TD3, CGP, SAC, TRPO, PPO]

for alg in algs:
  agent = alg(env)
  evl = Evaluator(agent, 'output_data')
  # evl.run_statistic(samples=1, seed=0)
# returns = load_dataset('path/to/returns_offline.npz')
# fig, ax = plot_learning_curves(returns, interval='bs')