import matplotlib.pyplot as plt
import numpy as np
from math import ceil

def plot_sim(sim, envprobe, errorprobe):
    '''
    network plots
    '''
    plt.figure()
    plt.subplot(311)
    plt.plot(sim.trange(), sim.data[errorprobe])
    plt.title("Error Signal")
    plt.subplot(312)
    plt.plot(sim.trange(), sim.data[envprobe][:,:-1])
    plt.ylim([-1.0, 1.0])
    plt.legend(["xloc", "yloc", "reward", "done"])
    plt.subplot(313)
    plt.plot(sim.trange(), sim.data[envprobe][:,2])
    plt.title("Reward")
    plt.show()


def plot_trajectories(sim, env, envprobe):
    '''
    trajectory plots
    '''
    episode_indices = np.where(sim.data[envprobe][:,3] == 1.0)
    episode_indices = np.append(episode_indices[0], max(sim.trange()) / env.timestep)

    fig = plt.figure()
    ax = plt.gca()

    last_episode = 0
    for episode in episode_indices:
        vx = sim.data[envprobe][int(last_episode):int(episode),0]
        vy = sim.data[envprobe][int(last_episode):int(episode),1]
        colour = np.random.random(3)
        ax.plot(vx, vy, '-', alpha=0.6, c=colour, label="%d-%d" % (int(last_episode), int(episode))) # plot all points
        ax.plot(vx[0], vy[0], 'o', alpha=0.6, c=colour) # plot start point as o
        ax.plot(vx[-1], vy[-1], 'x', alpha=0.6, c=colour) # plot end point as x
        last_episode = episode + 1

    arena = plt.Circle((0,0), 1, color='k', fill=False)
    platform = plt.Circle(env.platform_loc, env.platformsize/2, fill=True, color='k')
    ax.add_artist(platform)
    ax.add_artist(arena)
    ax.axis('equal')
    ax.legend()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.title("trajectory")
    plt.show()
