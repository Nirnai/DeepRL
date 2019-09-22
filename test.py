import gym 
import envs
import numpy as np
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator, plot_dataset

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


high = np.ones(2) * 100
low = -high
dist_space = gym.spaces.Box(low, high)


env = gym.make('CartpoleSwingup-v0')
# env = gym.make('InvertedPendulum-v2')
env.reset()

for t in range(1000):
    env.render()
    action = env.action_space.sample()
    env.env.physics.data.xfrc_applied[2][0] = 5.
    observation, reward, done, info = env.step(0) # take a random action
env.close()




# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
# fig = plt.figure()
# im = []
# for frame in frames:
#     im.append([plt.imshow(frame)])
# ani = animation.ArtistAnimation(fig, im, interval=10, blit=True, repeat_delay=1000)
# ani.save('dynamic_images.mp4')
# plt.show()

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


# fig1 = plt.figure()

# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                    interval=50, blit=True)
# line_ani.save('lines.mp4', writer=writer)

# fig2 = plt.figure()

# x = np.arange(-9, 10)
# y = np.arange(-9, 10).reshape(-1, 1)
# base = np.hypot(x, y)
# ims = []
# for add in np.arange(15):
#     ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

# im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
#                                    blit=True)
# im_ani.save('im.mp4', writer=writer)
