import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from math import ceil
import nengo

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

def plot_value_func(model, agent, env, backend, eval_points=50, len_presentation=0.1):
    '''
    simulate the output of the agents Critic net for positions all over the arena and plot
    an approximation of the value surface
    
    INPUTS:
        model               -   model used for original simulation
        agent               -   a (trained) agent model containing the critic net
        env                 -   environment the agent was trained in, needed to extract information about the arena
        simclass            -   the desired simulator to use
        eval_points         -   number of points in each dimension at which to evaluate the agent's value function
        len_presentation    -   how long toi present each location (may influence accuracy, will influence runtime)
    '''

    diameter = env.diameter
    x = np.linspace(-(diameter/2),diameter/2,eval_points)
    y = np.linspace(-1,1,eval_points)
    xx,yy = np.meshgrid(x,y)
    idx = np.sqrt(xx**2 + yy**2) < env.diameter / 2 # indices of all locations that fall inside the maze
    xx = xx[idx]
    yy = yy[idx]
    locs = np.array([xx.flatten(), yy.flatten()])
    locs = locs.T

    # set up model
    with model:
        eval_node = nengo.Node(nengo.processes.PresentInput(locs, len_presentation)) # present each position for 0.1 seconds
        place_cells = agent.PlaceCells.net
        value_func = agent.Critic.net

        nengo.Connection(eval_node, place_cells.placecells)
        nengo.Connection(place_cells.placecells, value_func.input)

        value_probe = nengo.Probe(value_func.output)
        
    sim = simulate_with_backend(backend, model, len(locs)*len_presentation, env.timestep)
    
    n_timepoints_presentation = int(len_presentation/env.timestep)
    print(f"each location was presented for {n_timepoints_presentation} steps")
    values = sim.data[value_probe]
    value_evaluations = values[n_timepoints_presentation-1::n_timepoints_presentation]
    print(f"number of sampled locations: {len(value_evaluations)}")
    print(f"presented locations: {locs.shape}")

    print(f"x: {xx.shape}, y: {yy.shape}, values:{value_evaluations.shape}")

    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xx,yy,value_evaluations.flatten(), cmap=cm.summer)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")

def simulate_with_backend(backend, model, duration, timestep):
    if backend == 'CPU':
        sim = nengo.Simulator(model, dt=timestep)

    elif backend == 'GPU':
        import nengo_ocl
        import pyopencl as cl
        # set device to avoid being prompted every time
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        sim = nengo_ocl.Simulator(model, context=ctx, dt=timestep)

    elif backend == 'LOIHI':
        import nengo_loihi
        sim = nengo_loihi.Simulator(model, dt=timestep)

    with sim:
        sim.run(duration)
    
    return sim
