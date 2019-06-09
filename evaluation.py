import os
import random
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
        self.average_rewards = np.array([])
        
        # Checks
        self.is_solved = False
        self.results_generated = False

        # Parameters
        self.episodes = episodes
        self.avaraging_window = averaging_window
        self.goal_average_reward = env.spec.reward_threshold
        if self.goal_average_reward is None:
            self.goal_average_reward = -200
            

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

    def process(self, reward, done):
        if type(reward) != list:
            reward = [reward]
        if type(done) != list:
            done = [done]

        for reward, done in zip(reward, done): 
            if done:
                self.average_rewards = np.append(self.average_rewards, np.mean(self.episode_rewards[-self.avaraging_window:]))
                self.episode_rewards.append(0.0)
                # Check for Termination
                if self.episodes > 0:
                    self.is_solved = self.curr_episode == self.episodes
                else:
                    self.is_solved = self.average_rewards[-1] >= self.goal_average_reward    
                self.curr_episode += 1
            else:
                self.episode_rewards[-1] += reward
        return self.is_solved, self.curr_episode
    
    def generate_results(self, path):
        # TODO: Generated Files should hold paramter info and unique identifier
        if self.results_generated:
            return
        else: 
            # TODO: Label data in plot and generate second plot with avarge and std

            # Load existing data
            filename = "{}_{}".format(self.alg_name, self.env_name)
            data = []
            if os.path.isfile('{}/{}.npz'.format(path,filename)):
                old_results = np.load('{}/{}.npz'.format(path,filename))
                for key in old_results.files:
                    data.append(old_results[key])
            data.append(self.average_rewards)

            # Plot
            plt.figure()
            for run, eval_rewards in enumerate(data):
                plt.plot(eval_rewards, label="run_{}".format(run))
            plt.legend()

            # Save data and plot
            plt.savefig('{}/{}.png'.format(path,filename))
            np.savez('{}/{}'.format(path, filename), *data)
            self.results_generated = True
        
    def generate_statistic(self, path):
        # load data
        filename = "{}_{}".format(self.alg_name, self.env_name)
        data = []
        if os.path.isfile('{}/{}.npz'.format(path,filename)):
            old_results = np.load('{}/{}.npz'.format(path,filename))
            for key in old_results.files:
                data.append(old_results[key])
        # Compute statistic
        stat = np.array([eval_rewards for eval_rewards in data])
        _, x = stat.shape
        mean = np.mean(stat, axis=0)
        std = np.std(stat, axis=0)
        # Plot
        plt.figure()
        plt.plot(range(x),mean, label = 'mean')
        plt.plot(range(x),[self.goal_average_reward]*x, 'k--', label = 'goal reward')
        plt.fill_between(range(x) ,mean - std, mean + std, facecolor='lightblue', label='std')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.savefig('{}/{}_stat.png'.format(path,filename))

    def evaluate_algorithm(self, alg, env, results_path, episodes=1000, samples=10, seed=0, render=False):
        self.episodes = episodes
        rng = random.Random(seed)
        seeds = []

        for i in range(samples):
            # reset environment
            self.reset()
            alg.reset()

            # Setting new seeds
            seeds.append(rng.randint(0,100))
            env.seed(seeds[i])
            alg.seed(seeds[i])
            

            state = env.reset()
            for t in count():
                # Act
                state, reward, done = alg.act(state)
                # Eval
                is_solved, episode = self.process(reward, done)
                # Learn
                loss = alg.learn()

                if done:
                    self.show_progress(interval=25)
                if is_solved:
                    # TODO: Evaluate the resulting policy
                    self.generate_results(results_path)
                    break
                    
        self.param.SEED = seeds
        self.param.AVERAGING_WINDOW = self.avaraging_window
        self.param.save_parameters('{}/parameters.json'.format(results_path))
        self.generate_statistic(results_path)
