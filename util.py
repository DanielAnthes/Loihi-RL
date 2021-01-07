import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from math import ceil
import nengo
from nengo.utils.ensemble import tuning_curves  # another option is response_curves 

def plot_sim(sim, envprobe, errorprobe, switchprobe):
    '''
    network plots
    '''
    t = sim.trange()

    fix, ax = plt.subplots(3,1, sharex=True)
    a = ax[0]
    a.plot(t, sim.data[errorprobe])
    a.set_title("Error Signal")
    
    a = ax[1]
    data = sim.data[envprobe][:,:-1]
    xloc = data[:,0]
    yloc = data[:,1]
    reward = data[:,2]
    done = data[:,3]
    
    a.set_ylim([-1.0, 1.0])
    a.plot(t, xloc, label="xloc")
    a.plot(t, yloc, label="yloc")
    a.vlines(t[np.where(reward==1)], 0, 1, 
        colors="green", linestyles="dashed", zorder=3,
        transform=a.get_xaxis_transform(), label="reward")
    a.vlines(t[np.where((reward==0) & (done==1))], 0, 1, 
        colors="red", linestyles="dashed", zorder=3,
        transform=a.get_xaxis_transform(), label="done")
    a.legend()

    a = ax[2]
    a.plot(t, sim.data[switchprobe])
    a.set_title("Learning")
    a.set_ylim([0,1.5])
    a.set_xlabel("Time")
    return fix

def plot_trajectories(sim, env, envprobe, cdat, labels=False, timestamps=True):
    '''
    trajectory plots
    '''
    episode_indices = np.where(sim.data[envprobe][:,3] == 1.0)
    episode_indices = np.append(episode_indices[0], max(sim.trange()) / env.timestep)

    fig = plt.figure()
    ax = plt.gca()

    last_episode = 0
    for episode in episode_indices:
        if episode == last_episode:
            continue
        vx = sim.data[envprobe][int(last_episode):int(episode),0]
        vy = sim.data[envprobe][int(last_episode):int(episode),1]
        ax.plot(vx, vy, '-', alpha=0.6, label="%d-%d" % (int(last_episode), int(episode)), color='black') # plot all points w labels
        ax.plot(vx[0], vy[0], 'o', alpha=0.6)
        if timestamps is True:
            ax.text(vx[0], vy[0], str(int(round(last_episode * env.timestep))), alpha=0.6, fontsize=8) # plot start point beginning t in s
        ax.plot(vx[-1], vy[-1], '*', alpha=0.6) # plot end point as x
        last_episode = episode + 1

    arena = plt.Circle((0,0), 1, color='k', fill=False)
    platform = plt.Circle(env.platform_loc, env.platformsize/2, fill=True, color='k')
    ax.add_artist(platform)
    ax.add_artist(arena)
    ax.axis('equal')
    if labels is True:
        ax.legend()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.title("Trajectory")
    return fig

def plot_weight_evolution_3d(sim, weights_probe, title="3D Weight evolution (undefined)"):
    '''
    3d weight evolution plots
    use of weights.shape[1] > 1
    '''
    weights = sim.data[weights_probe]
    conns = list(range(weights.shape[1]))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    for conn in conns:
        ax.plot(np.ones(weights.shape[0]) * conn, range(weights.shape[0]), weights[:,conn])
    plt.title(title)
    ax.set_xlabel("Connection")
    ax.set_ylabel("Time")
    ax.set_zlabel("Value")

def plot_weight_evolution_2d(sim, weights_probe, title="2D Weight evolution (undefined)"):
    '''
    2d weight evolution plots
    use of weights.shape[1] == 1
    '''
    weights = sim.data[weights_probe]

    plt.figure()
    plt.plot(range(weights.shape[0]), weights[:,0])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")

def plot_actions_by_activation(env, agent):
    '''
    action plots by activation of ensemble
    '''
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    actions = list(range(len(env.actions)))
    t = range(agent.DecisionMaker.activation.shape[0])

    for a in actions:
        ax.plot(np.ones(agent.DecisionMaker.activation.shape[0]) * a, t, agent.DecisionMaker.activation[:,a])
    plt.title("Action activations as a function of time")
    ax.set_xlabel("Action")
    ax.set_ylabel("Time")
    ax.set_zlabel("Activation")

def plot_actions_by_probability(env, agent, descriptives=True):
    '''
    action plots by probability derived from activation
    '''
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    actions = list(range(len(env.actions)))
    t = range(agent.DecisionMaker.probability.shape[0])

    for a in actions:
        ax.plot(np.ones(agent.DecisionMaker.probability.shape[0]) * a, t, agent.DecisionMaker.probability[:,a])
    plt.title("Action probability as a function of time")
    ax.set_xlabel("Action")
    ax.set_ylabel("Time")
    ax.set_zlabel("Probability")

    if descriptives:
        print("Total actions chosen=%d" % (agent.DecisionMaker.activation.shape[0]))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (0, env.actionmemory.count(0), np.sum(agent.DecisionMaker.probability[:,0]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,0])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (1, env.actionmemory.count(1), np.sum(agent.DecisionMaker.probability[:,1]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,1])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (2, env.actionmemory.count(2), np.sum(agent.DecisionMaker.probability[:,2]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,2])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (3, env.actionmemory.count(3), np.sum(agent.DecisionMaker.probability[:,3]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,3])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (4, env.actionmemory.count(4), np.sum(agent.DecisionMaker.probability[:,4]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,4])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (5, env.actionmemory.count(5), np.sum(agent.DecisionMaker.probability[:,5]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,5])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (6, env.actionmemory.count(6), np.sum(agent.DecisionMaker.probability[:,6]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,6])))
        print("Action %d: Chosen %d times with average probability of %.2f and median p of %.2f." % (7, env.actionmemory.count(7), np.sum(agent.DecisionMaker.probability[:,7]) / agent.DecisionMaker.probability.shape[0] * 100, np.median(agent.DecisionMaker.probability[:,7])))

def plot_actions_by_decision(env):
    '''
    action plots by the decisions made upon probabilities
    '''
    plt.figure()
    edges = np.array(range(len(env.actions)+1)) - 0.5
    plt.hist(env.actionmemory, align='mid', bins=edges)
    plt.title("Action distribution")

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
    locs = np.array([xx.flatten(), yy.flatten()]).T

    # set up model
    with model:
        eval_node = nengo.Node(nengo.processes.PresentInput(locs, len_presentation)) # present each position for 0.1 seconds
        place_cells = agent.PlaceCells.net
        value_func = agent.Critic.net

        model.switch.state=0 # switch off learning for plotting

        nengo.Connection(eval_node, place_cells.placecells)
        nengo.Connection(place_cells.placecells, agent.net.input)

        value_probe = nengo.Probe(value_func.output, synapse=len_presentation)

    sim = simulate_with_backend(backend, model, len(locs)*len_presentation, env.timestep)

    n_timepoints_presentation = int(len_presentation/env.timestep)
    print("each location was presented for {} steps".format(n_timepoints_presentation))
    values = sim.data[value_probe]
    value_evaluations = values[n_timepoints_presentation-1::n_timepoints_presentation]
    print("number of sampled locations: {}".format(len(value_evaluations)))
    print("presented locations: {}".format(locs.shape))

    print("x: {}, y: {}, values:{}".format(xx.shape, yy.shape, value_evaluations.shape))

    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xx,yy,value_evaluations.flatten(), cmap=cm.summer)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")

def plot_place_cell(model, agent, env, backend, loc, len_presentation=0.1):
    '''
    plot a place cell, idealised vs what we simulate
    N.B. does not work anymore with the Guassian encoders
    '''
    
    # get idealised outputs
    activations = agent.PlaceCells.place_cell_activation(loc)

    # get agent.input outputs
    with model:
        eval_node = nengo.Node(nengo.processes.PresentInput([loc], len_presentation)) # present each position for 0.1 seconds
        place_cells = agent.PlaceCells.net
        model.switch.state = 0
        nengo.Connection(eval_node, place_cells.placecells)

        value_probe = nengo.Probe(agent.net.input, synapse=len_presentation)

    sim = simulate_with_backend(backend, model, len([loc]) * len_presentation, env.timestep)
    values = sim.data[value_probe][-1,:]

    # plotting
    fig = plt.figure()

    ax_a = fig.add_subplot(1, 2, 1, projection='3d')
    ax_a.plot_trisurf(agent.PlaceCells.cell_locs[0,:], agent.PlaceCells.cell_locs[1,:], activations, cmap=cm.summer)
    ax_a.set_xlabel("X")
    ax_a.set_ylabel("Y")
    ax_a.set_zlabel("Activation")
    plt.title('Idealised Place Cell')

    ax_b = fig.add_subplot(1, 2, 2, projection='3d')
    ax_b.plot_trisurf(agent.PlaceCells.cell_locs[0,:], agent.PlaceCells.cell_locs[1,:], values, cmap=cm.summer)
    ax_b.set_xlabel("X")
    ax_b.set_ylabel("Y")
    ax_b.set_zlabel("Activation")
    plt.title('Simulated Place Cell')

def plot_tuning_curves(model, ensemble):
    with nengo.Simulator(model) as sim:
        eval_points, activities = tuning_curves(ensemble, sim)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="3d")
    ax.set_title("Tuning curves")
    for i in range(ensemble.n_neurons):
        ax.plot_surface(
            eval_points.T[0], eval_points.T[1], activities.T[i], cmap=plt.cm.autumn
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Firing rate (Hz)")
    plt.show()

def simulate_with_backend(backend, model, duration, timestep):
    sim = create_simulator(backend, model, timestep)

    with sim:
        sim.run(duration)

    return sim

def create_simulator(backend:str, model:nengo.Network, timestep:float) -> nengo.Simulator:
    if backend == 'GPU':
        import nengo_ocl
        import pyopencl as cl
        # set device to avoid being prompted every time
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        sim = nengo_ocl.Simulator(model, context=ctx, dt=timestep)

    elif backend == 'LOIHI':
        import nengo_loihi
        sim = nengo_loihi.Simulator(model, dt=timestep, target='loihi')

    else: # backend == 'CPU':
        sim = nengo.Simulator(model, dt=timestep)

    return sim