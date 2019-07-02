import numpy as np
import scipy.stats

def mean_confidance(data, confidence=0.99): 
    n = len(data)
    mean = np.mean(data, axis = 0)        
    se = np.std(data, axis = 0)/np.sqrt(n)
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    else:
        raise Exception('confidance can only have the values 0.90, 0.95 or 0.99')
    low  = mean - z * se
    high = mean + z * se 
    return mean, low, high


def bootstrap_confidance(self, data, n=1000, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    m = func(data, axis = 0)
    idx = np.random.choice(data.shape[0], (data.shape[0],n))
    bootstrap_resample = data[idx, :]
    means = func(bootstrap_resample, axis=0)
    # means.sort(axis=1)
    low = np.percentile(means, 2.5, axis=0)
    high = np.percentile(means, 97.5, axis=0)
    return m, low, high