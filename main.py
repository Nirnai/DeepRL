import itertools
import gym 
from algorithms import PPO, TRPO, SAC
from evaluation import Evaluation


if __name__ == '__main__':

    # Test ssh
    # Environment
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v0')
    # env = gym.make('Acrobot-v1')
    # env.spec.reward_threshold = -200
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('MountainCar-v0')
    # Hyperparameters
    # filepath = 'algorithms/ppo/parameters.json'
    # param = HyperParameter(filepath)
    
    # RL Algorithm
    # alg = PG(env, param)
    # alg = DQN(env, param)
    # alg = A2C(env, param)
    # alg = PPO(env)
    alg = SAC(env)
    # alg = TRPO(env)

    # Evaluation
    evaluator = Evaluation(env, alg)
    # evaluator.evaluate_algorithm(alg, env, 'test/output_data', episodes=2000, seed=1)


    state = env.reset()
    for t in itertools.count():
        # Act
        state, reward, done = alg.act(state)
        # Log
        is_solved, episode = evaluator.process(reward, done)
        # Train
        alg.learn()
        # Visualize Progress
        if done:
            evaluator.show_progress(interval=1)
        if is_solved:
            evaluator.generate_results('test/output_data')
            break

    # Exploit Learned Policy
    state = env.reset()
    while True:
        env.render()
        state, reward, done = alg.act(state, exploit=True)
        is_solved, episode = evaluator.process(reward, done)
        evaluator.show_progress(interval=1)
        if done:
            state = env.reset()
