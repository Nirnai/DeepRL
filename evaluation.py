import os
import random
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

class Evaluation():
    def __init__(self, env, algorithm, episodes=0, averaging_window=100):
        self.alg_name = algorithm.name
        self.env_name = env.spec.id
        self.alg = algorithm
        self.env = env
        self.param = algorithm.param
        
        # Logs
        self.curr_episode = 1
        self.episode_rewards = [0.0]
        self.episode_rewards_est = []
        self.average_rewards = np.array([])

        # Checks
        self.is_solved = False
        self.results_generated = False

        # Parameters
        self.episodes = episodes
        self.avaraging_window = averaging_window
        self.goal_average_reward = env.spec.reward_threshold
        if self.goal_average_reward is None:
            self.goal_average_reward = -250
            

    # TODO: Create Function that does an evaluation on final policy
    def reset(self):
        self.curr_episode = 1
        self.episode_rewards = [0.0]
        self.average_rewards = np.array([])
        self.is_solved = False
        self.results_generated = False

    def show_progress(self, interval=10):
        if self.curr_episode % interval == 0:
            print("Episode: {}".format(self.curr_episode))
            print("Average Reward: {:.2f}".format(self.average_rewards[-1]))
            print("Goal Average Reward: {}".format(self.goal_average_reward))
            print("Steps: {:,}".format(self.alg.steps))
            print("------------------------------------")

    def process(self, state, reward, done):
        if type(reward) != list:
            reward = [reward]
        if type(done) != list:
            done = [done]

        for reward, done in zip(reward, done): 
            if done:
                self.average_rewards = np.append(self.average_rewards, np.mean(self.episode_rewards[-self.avaraging_window:]))
                self.episode_rewards.append(0.0)
                self.episode_rewards_est.append(self.alg.critic(torch.from_numpy(state).float()).item())
                # Check for Termination
                if self.episodes > 0:
                    self.is_solved = self.curr_episode == self.episodes
                else:
                    self.is_solved = self.average_rewards[-1] >= self.goal_average_reward    
                self.curr_episode += 1
            else:
                self.episode_rewards[-1] += reward
        return self.is_solved, self.curr_episode
    
    def generate_results(self, filename):
        if self.results_generated:
            return
        else: 
            # Load existing data
            infile = '{}.npz'.format(filename)
            samples = []
            if os.path.isfile(infile):
                samples = [array for array in np.load(infile).values()]
            samples.append(self.average_rewards)
            # Plot Rewards
            self.plot_dataset(samples, filename, 'pdf')
            np.savez(filename, *samples)
            self.results_generated = True

    def generate_statistic(self, filename, dist='unknown'):
        # load data
        infile = '{}.npz'.format(filename)
        outfile = '{}_stat.pdf'.format(filename)
        if os.path.isfile(infile):
            samples = np.array([array for array in np.load(infile).values()])
        # Compute statistic
        if dist is 'normal':
            mean, low, high = self.normal_statistic(samples)
        else:
            mean, low, high = self.bootstrap_statistic(samples)
        # Plot
        fig = self.plot_statistic(mean, low, high)
        fig.savefig(outfile, format='pdf')

    def evaluate_algorithm(self, alg, env, results_path, episodes=1000, samples=10, seed=0, render=False):
        results_filename = "{}/{}_{}".format(results_path, self.alg_name, self.env_name)
        self.episodes = episodes
        rng = random.Random(seed)
        seeds = []

        for i in range(samples):
            # Setting new seeds
            seeds.append(rng.randint(0,100))
            env.seed(seeds[i])
            alg.seed(seeds[i])
            # reset environment
            self.reset()
            alg.reset()

            state = env.reset()
            for t in count():
                # Act
                state, reward, done = alg.act(state)
                # Eval
                is_solved, episode = self.process(state, reward, done)
                # Learn
                alg.learn()
                if done:
                    self.show_progress(interval=25)
                if is_solved:
                    # TODO: Evaluate the resulting policy
                    self.generate_results(results_filename)
                    break
                    
        self.param.SEED = seeds
        self.param.AVERAGING_WINDOW = self.avaraging_window
        self.param.save_parameters('{}/parameters.json'.format(results_path))
        self.generate_statistic(results_filename)
        # self.generate_video(results_path)
    
    def generate_video(self, path):
        env = gym.wrappers.Monitor(self.alg.env, path)
        state = env.reset()
        for _ in range(10):
            while True:
                state, reward, done = self.alg.act(state, exploit=True)
                if done: break

    def compare_datasets(self, datasets):
        fig = plt.figure()
        plt.title('Pendulum-v0 (TRPO, Different Random Seeds)')
        for i, dataset in enumerate(datasets):
            if os.path.isfile(dataset):
                data = np.load(dataset)
            else:
                raise FileNotFoundError

            data = np.array([array for array in data.values()])
            mean, low, high = self.normal_statistic(data)
            # label = os.path.basename(os.path.normpath(path)).split('_')[0]

            episodes = range(len(mean))
            plt.plot(episodes, mean, label='algo {}'.format(i+1))
            plt.fill_between(episodes,low, high, alpha=0.5)
            plt.xlabel('Episodes')
            plt.ylabel('Average Reward')
        plt.plot(episodes,[self.goal_average_reward] * len(episodes), 'k--', label = 'goal reward')
        plt.legend(loc=4)
        plt.show()

    def normal_statistic(self, data):
        n = len(data)
        mean = np.mean(data, axis = 0)        
        std = np.std(data, axis = 0)
        low = mean -  2.08596 * std/np.sqrt(n)
        high = mean + 2.08596 * std/np.sqrt(n)

        return mean, low, high

    def interpercentile_statistic(self,data):
        mean = np.mean(data, axis = 0)
        low = np.percentile(data, 10, axis = 0)
        high = np.percentile(data, 90, axis = 0)
        return mean, low, high
        
    def plot_statistic(self, mean, low, high):
        episodes = range(len(mean))
        fig = plt.figure()
        plt.plot(episodes, mean, label = 'mean')
        plt.plot(episodes,[self.goal_average_reward] * len(episodes), 'k--', label = 'goal reward')
        plt.fill_between(episodes,low, high, facecolor='lightblue', label='std')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        return fig

    def plot_dataset(self, data, outfile , outformat):
        plt.figure()
        for i, rewards in enumerate(data):
            plt.plot(rewards, label="run_{}".format(i))
        # plt.plot(self.episode_rewards_est, label="values")
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        # Save data and plot
        plt.savefig('{}.{}'.format(outfile, outformat), format=outformat)


    def bootstrap_statistic(self, data, n=1000, func=np.mean):
        """
        Generate `n` bootstrap samples, evaluating `func`
        at each resampling. `bootstrap` returns a function,
        which can be called to obtain confidence intervals
        of interest.
        """
        mean = func(data, axis = 0)
        idx = np.random.choice(data.shape[0], (data.shape[0],n))
        bootstrap_resample = data[idx, :]
        means = func(bootstrap_resample, axis=0)
        # means.sort(axis=1)
        low = np.percentile(means, 2.5, axis=0)
        high = np.percentile(means, 97.5, axis=0)
        return mean, low, high


    def compare_statistics(self, data):
        fig = plt.figure()
        statistics = [self.interpercentile_statistic, self.normal_statistic, self.bootstrap_statistic]
        labels = ['interpercentile', 'normal', 'bootstrapped']
        if os.path.isfile(data):
            data = np.load(data)
        else:
            raise FileNotFoundError

        data = np.array([array for array in data.values()])

        for i, statistic in enumerate(statistics): 
            
            mean, low, high = statistic(data)
            # label = os.path.basename(os.path.normpath(path)).split('_')[0]

            episodes = range(len(mean))
            plt.plot(episodes, mean, label = labels[i])
            plt.fill_between(episodes,low, high, alpha=0.8)
            plt.xlabel('Episodes')
            plt.ylabel('Average Reward')
        plt.plot(episodes,[self.goal_average_reward] * len(episodes), 'k--', label = 'goal reward')
        plt.legend()
        plt.show()