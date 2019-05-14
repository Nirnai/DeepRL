def getEnvInfo(env):
    state_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        action_space = 'discrete'
    else:
        action_dim = env.action_space.shape[0]
        action_space = 'continuous'

    return state_dim, action_dim, action_space