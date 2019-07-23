import os
import time
import numpy as np
from copy import deepcopy
from itertools import count
# from plot import plot_dataset

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class Evaluator():
    def __init__(self, algorithm, total_timesteps=1e6, eval_timesteps=1000, averaging_window=20):
        self.alg = algorithm
        self.param = algorithm.param
        self._alg_name = algorithm.name
        self._env_name = algorithm.env.spec.id
        # Parameters
        self._total_timesteps = int(total_timesteps)
        self._eval_timesteps = int(eval_timesteps)
        self._window = averaging_window
        self._desired_average_return = algorithm.env.spec.reward_threshold
        self._log_interval = 1
        # Metrics
        self._curr_episode = 0
        self._returns = [0.0]
        self._average_returns = []
        self._metrics = dict()
        # Checks
        self._solved = False


    def evaluate(self, output, samples=10, seed=100, mode='online'):
        if not os.path.isdir(output):  
            os.mkdir(output)
        output_filename = "{}/{}_{}".format(output, self._alg_name, self._env_name)
        seeds = []
        np.random.seed(seed)
        for i in range(samples):
            seeds.append(np.random.randint(0,100))
            self._seed(seeds[i])
            self.reset()
            self._train(eval_mode=mode)
            self.save_returns(output_filename)
            self.save_metrics(output_filename)

    def save_returns(self, path):
        returns_file = '{}_returns.npz'.format(path)
        samples = []
        if os.path.isfile(returns_file):
            samples = [array for array in np.load(returns_file).values()]
        samples.append(self._average_returns)
        np.savez(returns_file[:-4], *samples)
    
    def save_metrics(self, path):
        for key, values in self._metrics.items():
            key.replace(" ", "")                
            filename = '{}_{}.npz'.format(path, key)
            samples = []
            if os.path.isfile(filename):
                samples = [array for array in np.load(filename).values()]
            samples.append(values)
            np.savez(filename[:-4], *samples)

    def reset(self):
        self._curr_episode = 0
        self._returns = [0.0]
        self._average_returns = []
        self._metrics = dict()
        self._solved = False
        self.alg.reset()


    def _train(self, eval_mode='online'):
        done = True
        t1 = time.time()
        for t in range(self._total_timesteps):
            if done:
                state = self.alg.env.reset()
            state, reward, done = self._step(state)
            if eval_mode is 'online':
                self._eval_online(reward, done)
            elif eval_mode is 'offline':
                self._eval_offline()
            else: 
                NotImplementedError
            if done and self._curr_episode % self._log_interval == 0:
                self._print_progress()
                t2 = time.time()
                print("Time Elapsed since last progress Update: {:.3f}s".format((t2-t1)))
                t1 = deepcopy(t2)

    def _step(self, state):
        # Act
        state, reward, done = self.alg.act(state)
        # Learn
        metrics = self.alg.learn()
        # Collect Metrics
        self._log_metrics(metrics)

        return state, reward, done

    def _eval_online(self, reward, done):
        self._log_reward(reward, done)
        if done:
            self._average_returns.append(np.mean(self._returns[-self._window:-1]))
            # if self._curr_episode % self._log_interval == 0:
            #     self._print_progress()

    def _eval_offline(self):
        if self.alg.steps % self._eval_timesteps == 0:
            eval_alg = deepcopy(self.alg)
            state = eval_alg.env.reset()
            while True:
                state, reward, done = eval_alg.act(state, exploit=True)
                self._log_reward(reward, done)
                if(self._curr_episode == self._window):
                    self._average_returns.append(np.mean(self._returns))
                    # self._print_progress()
                    self._returns = [0.0]
                    self._curr_episode = 0
                    break


    def _log_reward(self, reward, done):
        if done:
            self._curr_episode += 1
            self._returns.append(0.0)
        else:
            self._returns[-1] += reward

    def _log_metrics(self, metrics):
        if type(metrics) is dict:
            for key, value in metrics.items():
                if value is not None:
                    if key in self._metrics:
                        self._metrics[key].append(value)
                    else:
                        self._metrics[key] = [value]


    def _print_progress(self):
        print("Steps: {:,}".format(self.alg.steps))
        print("Average Reward: {:.2f}".format(self._average_returns[-1]))
        print("Goal Average Reward: {}".format(self._desired_average_return))
        for key, value in self._metrics.items():
            print("{}: {:.2f}".format(key, self._metrics[key][-1]))
        print("------------------------------------")
    

    def _seed(self, seed):
        self.alg.env.seed(seed)
        self.alg.seed(seed)


