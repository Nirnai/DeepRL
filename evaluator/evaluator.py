import os
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import count
# from plot import plot_dataset


class Evaluator():
    def __init__(self, env, algorithm, total_timesteps=0, averaging_window=20):
        self.alg_name = algorithm.name
        self.env_name = env.spec.id
        self.alg = algorithm
        self.env = env
        self.param = algorithm.param

        # Parameters
        self.total_timesteps = int(total_timesteps)
        self.eval_timesteps = self.alg.param.BATCH_SIZE
        self.eval_episodes = averaging_window
        self.desired_average_return = env.spec.reward_threshold
        if self.desired_average_return is None:
            self.desired_average_return = 0
        self.log_interval = 10
        
        # Metrics
        self.curr_episode = 0
        self.returns = [0.0]
        self.average_returns = []
        self.policy_entropy = []
        self.value_loss = []

        # # Checks
        self.solved = False

    def evaluate(self, output, episodes=2000, samples=10, seed=0):
        self.episodes = episodes
        output_filename = "{}/{}_{}".format(output, self.alg_name, self.env_name)
        seeds = []
        np.random.seed(seed)
        
        for i in range(samples):
            seeds.append(np.random.randint(0,100))
            self.seed(seeds[i])
            self.reset()
            self.run()
            self.save_returns(output_filename)
            self.save_value_loss(output_filename)
            self.save_policy_entropy(output_filename)
        # plot_dataset(output_filename + '.npz')


    def run(self):
        state = self.env.reset()
        for t in range(self.total_timesteps):
            state = self.train(state)
            if t % self.eval_timesteps == 0:
                self.eval()
                state = self.env.reset()


    def train(self, state):
        # Act
        state, reward, done = self.alg.act(state, exploit=False)
        # Learn
        # loss, entropy = 
        # TODO: Sum of entropy (trajectory) and mse loss of value
        loss, entropy = self.alg.learn()

        # Eval
        self.process_value_loss(loss)
        self.process_policy_entropy(entropy)
        return state

    
    def eval(self):
        eval_alg = deepcopy(self.alg)
        state = eval_alg.env.reset()
        while True:
            # Act
            state, reward, done = eval_alg.act(state, exploit=True)
            # Eval
            self.process_rewards(reward, done)
            if(self.curr_episode == self.eval_episodes):
                self.average_returns.append(np.mean(self.returns))
                self.print_progress()
                self.returns = [0.0]
                self.curr_episode = 0
                break


    def process(self, reward, done, value_loss=None, policy_entropy=None):
        self.process_rewards(reward, done)
        if value_loss is not None:
            self.process_value_loss(value_loss)
        if policy_entropy is not None:
            self.process_policy_entropy(policy_entropy)


    def process_rewards(self, reward, done):
        if type(reward) != list:
            reward = [reward]
        if type(done) != list:
            done = [done]
        for reward, done in zip(reward, done): 
            if done:
                self.curr_episode += 1
                # self.average_returns = np.append(self.average_returns, np.mean(self.returns[-self.avaraging_window:]))
                if self.curr_episode < self.eval_episodes:
                    self.returns.append(0.0)
                # if self.curr_episode % self.log_interval == 0:
                #     self.progress()
                
            else:
                self.returns[-1] += reward
            # self.solved = self.alg.steps == self.total_timesteps
    
    def process_value_loss(self, loss):
        if loss is not None:
            self.value_loss.append(loss)
    
    def process_policy_entropy(self, entropy):
        if entropy is not None:
            self.policy_entropy.append(entropy)

    def save_returns(self, path):
        returns_file = '{}_returns.npz'.format(path)
        samples = []
        if os.path.isfile(returns_file):
            samples = [array for array in np.load(returns_file).values()]
        samples.append(self.average_returns)
        np.savez(path, *samples)
    
    def save_value_loss(self, path):
        loss_file = '{}_loss.npz'.format(path)
        samples = []
        if os.path.isfile(loss_file):
            samples = [array for array in np.load(loss_file).values()]
        samples.append(self.value_loss)
        np.savez(path, *samples)

    def save_policy_entropy(self, path):
        entropy_file = '{}_entropy.npz'.format(path)
        samples = []
        if os.path.isfile(entropy_file):
            samples = [array for array in np.load(entropy_file).values()]
        samples.append(self.policy_entropy)
        np.savez(path, *samples)


    def print_progress(self):
        print("Steps: {:,}".format(self.alg.steps))
        print("Average Reward: {:.2f}".format(self.average_returns[-1]))
        print("Goal Average Reward: {}".format(self.desired_average_return))
        if self.value_loss:
            print("Critic Loss: {:.2f}".format(self.value_loss[-1]))
        if self.policy_entropy:
            print("Policy Entropy: {:.2f}".format(self.policy_entropy[-1]))
        print("------------------------------------")
    

    def seed(self, seed):
        self.env.seed(seed)
        self.alg.seed(seed)


    def reset(self):
        self.curr_episode = 0
        self.returns = [0.0]
        self.average_returns = []
        self.value_loss = []
        self.policy_entropy = []
        self.solved = False
        self.alg.reset()