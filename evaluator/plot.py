
import os
import numpy as np 

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

import scipy.stats as stats
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


plt.style.use('seaborn-notebook')
width = 422.52348  # Get this from LaTeX using \showthe\textwidth
nice_fonts = {
              # Use LaTeX to write all text
              "text.usetex": True,
              "font.family": "Latin Modern Roman",
              # Use 10pt font in plots, to match 10pt font in document
              "axes.labelsize": 11,
              "font.size": 11,
              # Make the legend/label fonts a little smaller
              "legend.fontsize": 8,
              "xtick.labelsize": 8,
              "ytick.labelsize": 8,
              }
mpl.rcParams.update(nice_fonts)
path = '/home/nirnai/Cloud/Uni/TUM/MasterThesis/LaTex/figures'


def load_dataset(path):
    if os.path.isfile(path):
            data = np.array([array for array in np.load(path).values()])
            return data
    else:
        raise FileNotFoundError

def get_labels(paths):
    labels = []
    for path in paths:
        l = path.split('/')
        v = l[-2].split('_')
        alg = v[0]
        task = v[1]
        experiment = l[1]
        labels.append(l[2])
    return labels, experiment, alg, task

def plot_learning_curves(datas, labels=None, interval='t'):
    fig, ax = plt.subplots(1,1,figsize=set_size(width))
    for data in datas:
        x = np.linspace(0,1000, data.shape[1])
        n = data.shape[0]    
        if interval == 't':
            means = data.mean(axis=0)
            se = stats.sem(data, axis=0)
            low, high = stats.t.interval(0.95, n-1 ,loc=means, scale=se)
        elif interval == 'bs':
            means = np.zeros(data.shape[1])
            low = np.zeros(data.shape[1])
            high = np.zeros(data.shape[1])
            for i in range(data.shape[1]):
                temp = bs.bootstrap(data[:,i], stat_func=bs_stats.median, alpha=0.05, is_pivotal=False)
                means[i] = temp.value
                low[i] = temp.lower_bound
                high[i] = temp.upper_bound
         
        ax.plot(x,means)
        ax.fill_between(x, low, high, alpha=0.2)
        ax.set_ylim([0,1200])
    return fig, ax


def plot_final_performance(datas,x=None,interval='t'):
    fig, ax = plt.subplots(1,1,figsize=set_size(width))
    means = []
    ci = []
    for data in datas:
        n = data.shape[0] 
        if interval == 't':
            means.append(data[:,-1].mean(axis=0))
            se = stats.sem(data[:,-1], axis=0)
            ci.append(stats.t.interval(0.95, n-1 ,loc=means[-1], scale=se))
    ci = np.array(ci)
    yerr = np.c_[means-ci[:,0],ci[:,1]-means ].T
    ax.bar(x, means, yerr=yerr, align='center', alpha=0.5, ecolor='black', capsize=10)    

def plot_kl(datas, interval='t'):
    fig, ax = plt.subplots(1,1,figsize=set_size(width))
    for data in datas:
        x = np.linspace(0,1000, data.shape[1])
        n = data.shape[0]    
        if interval == 't':
            means = data.mean(axis=0)
            se = stats.sem(data, axis=0)
            low, high = stats.t.interval(0.95, n-1 ,loc=means, scale=se)
        elif interval == 'bs':
            means = np.zeros(data.shape[1])
            low = np.zeros(data.shape[1])
            high = np.zeros(data.shape[1])
            for i in range(data.shape[1]):
                temp = bs.bootstrap(data[:,i], stat_func=bs_stats.mean, alpha=0.05, is_pivotal=False)
                means[i] = temp.value
                low[i] = temp.lower_bound
                high[i] = temp.upper_bound
         
        ax.plot(x,means)
        # ax.fill_between(x, low, high, alpha=0.2)
        ax.set_ylim([0,0.05])


def set_size(width, fraction=1):
  """ 
  Set aesthetic figure dimensions to avoid scaling in latex.
  Parameters
  ----------
  width: float
        Width in pts
  fraction: float
        Fraction of the width which you wish the figure to occupy

  Returns
  -------
  fig_dim: tuple Dimensions of figure in inches
  """
  # Width of figure
  fig_width_pt = width * fraction
  # Convert from pt to inches
  inches_per_pt = 1 / 72.27
  # Golden ratio to set aesthetic figure height
  golden_ratio = (5**.5 - 1) / 2
  # Figure width in inches
  fig_width_in = fig_width_pt * inches_per_pt
  # Figure height in inches
  fig_height_in = fig_width_in * golden_ratio
  fig_dim = (fig_width_in, fig_height_in)
  return fig_dim





########################################################
########################################################
########################################################




def plot_dataset(path):
    data = load_dataset(path)
    episodes = range(len(data[0]))
    fig = plt.figure()
    for sample in data:
        plt.plot(episodes, sample)
    return fig

# def plot_offline(path_returns, path_deviations, n):
#     fig = plt.figure()
#     for path_return, path_deviation in zip(path_returns, path_deviations):
#         means = load_dataset(path_return)
#         stds = load_dataset(path_deviation)
#         n_total = len(means) * n
#         new_mean = np.mean(means, axis=0)
#         new_std = np.sqrt(n/n_total * (np.sum(stds**2, axis=0) + np.sum((means - new_mean)**2, axis=0)))
#         new_se = new_std / np.sqrt(10)
#         episodes = np.linspace(0, 1000 , len(new_mean))
#         plt.plot(episodes, [1000] * len(episodes), 'k--', label = 'goal reward')
#         plt.plot(episodes, new_mean)
#         plt.fill_between(episodes, new_mean - new_std, new_mean + new_std, alpha=0.2)
#     # plt.fill_between(range(len(new_mean)),new_mean - 1.96 * new_se, new_mean + 1.96 * new_se, alpha=0.5)


# def plot_final_performance(paths_returns, paths_deviations, n):
#     x = []
#     means = []
#     stds = []
#     for path_returns, path_deviations in zip(paths_returns, paths_deviations):
#         label = path_returns.split('/')[-2]
#         mean = load_dataset(path_returns)
#         std = load_dataset(path_deviations)
#         n_total = len(mean) * n
#         new_mean =  n/n_total * np.sum(mean)
#         new_std = np.sqrt(n/n_total * (np.sum(std**2) + np.sum((mean - new_mean)**2)))
#         x.append(label)
#         means.append(new_mean)
#         stds.append(new_std)
#     fig, ax = plt.subplots()
#     ax.bar(x, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)


# def compare_datasets(paths, goal=0, show=False):
#     fig = plt.figure()
#     for path in paths:
#         name = os.path.basename(path)[:-4]
#         alg_name = name.split('_')[0]
#         env_name = name.split('_')[1]
#         if len(name.split('_')) == 3:
#             exp_name = name.split('_')[2]
#         else:
#             exp_name = ''
#         dataset = load_dataset(path)
#         episodes = np.linspace(0,1000,len(dataset[0])) #range(len(dataset[0]))
#         mean, low, high = mean_confidance(dataset)
#         plt.plot(episodes, mean, label='{} {}'.format(alg_name, exp_name))
#         plt.fill_between(episodes, low, high, alpha=0.2)
#         plt.legend()
#         plt.xlabel('episodes')
#         plt.ylabel('Average Reward')
#         plt.title('{}'.format(env_name))
#     plt.plot(episodes, [goal] * len(episodes), 'k--', label = 'goal reward')
#     if show:
#         plt.show()
#     return fig



################################################################
########################## Utilities ###########################
################################################################

def combine_statistics(means, stds, n):
    n_total = len(means) * n
    new_mean = np.mean(means)
    new_std = np.sqrt(n/n_total * (np.sum(stds**2) + np.sum((means - new_mean)**2)))
    return new_mean, new_std
        


if __name__ == '__main__':
    fig = plot_dataset('example.npz', statistic='normal')
    plt.show()
