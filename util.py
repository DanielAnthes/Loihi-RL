import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from math import ceil
import nengo

def plot_sim(sim, envprobe, errorprobe, switchprobe):
    '''
    network plots
    '''
    plt.figure()
    plt.subplot(411)
    plt.plot(sim.trange(), sim.data[errorprobe])
    plt.title("Error Signal")
    plt.subplot(412)
    plt.plot(sim.trange(), sim.data[envprobe][:,:-1])
    plt.ylim([-1.0, 1.0])
    plt.legend(["xloc", "yloc", "reward", "done"])
    plt.subplot(413)
    plt.plot(sim.trange(), sim.data[envprobe][:,2])
    plt.title("Reward")
    plt.subplot(414)
    plt.plot(sim.trange(), sim.data[switchprobe])
    plt.title("Learning")
    plt.ylim([0,1.5])

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

    
def plot_value_func(model, agent, env, backend, eval_points=50, len_presentation=0.1):
    '''
    TODO: add smoothing over presentation of each state
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

        model.switch.state=0 # switch off learning for plotting

        nengo.Connection(eval_node, place_cells.placecells)
        nengo.Connection(place_cells.placecells, value_func.input)

        value_probe = nengo.Probe(value_func.output, synapse=len_presentation)
        
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

