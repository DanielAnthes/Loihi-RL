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
    plot trajectories
    TODO: known to break
    '''
    
    episode_indices = np.where(sim.data[envprobe][:,3] == 1.0)
    episode_indices = np.append(episode_indices[0], max(sim.trange()) / env.timestep)

    plot_columns = 2
    plot_rows = ceil(len(episode_indices) / plot_columns)

    prior = 0
    n = 1

    plt.figure()

    for index in episode_indices:
        vx = sim.data[envprobe][int(prior):int(index),0]
        vy = sim.data[envprobe][int(prior):int(index),1]
        sub = int(plot_rows * 100 + plot_columns * 10 + n)

        ax = plt.subplot(sub)
        ax.plot(vx, vy, ".-")
        arena = plt.Circle((0,0), 1, color='k', fill=False)
        platform = plt.Circle(env.platform_loc, env.platformsize/2, fill=True, color='k')
        ax.add_artist(platform)
        ax.add_artist(arena)
        ax = plt.gca()
        ax.axis('equal')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title("%d-%dms" % (int(prior), int(index)))

        prior = index + 1
        n += 1

    plt.show()
