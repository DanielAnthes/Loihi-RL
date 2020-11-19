import matplotlib.pyplot as plt
import nengo


from Environment import Maze
from Agent import Mouse

BACKEND = 'CPU' # choice of CPU, GPU and LOIHI
PLOT_TRAJECTORIES = True # True to plot the trajectories the mouse took


# set up simulation, connect networks
env = Maze()

with nengo.Network() as model:
    agent = Mouse(env, 23, 23, act_lr=0.01, crit_lr=0.01)

    # TODO add error node
    # environment node, step function expects integer so need to cast from float
    envstate = nengo.Node(lambda time, action: env.step(int(action)), size_in=1, size_out=5)

    # compute place cell activations
    nengo.Connection(envstate[:2], agent.PlaceCells.net.placecells)

    # place cells give input to actor and critic
    nengo.Connection(agent.PlaceCells.net.placecells, agent.Critic.net.input)
    nengo.Connection(agent.PlaceCells.net.placecells, agent.Actor.net.input)

    # take actor net as input to decision node
    nengo.Connection(agent.Actor.net.output, agent.DecisionMaker.net.choicenode)

    # execute action in environment
    nengo.Connection(agent.DecisionMaker.net.choicenode, envstate)

    # connect error node
    nengo.Connection(envstate[2], agent.Error.net.errornode[0])
    nengo.Connection(agent.Critic.net.output, agent.Error.net.errornode[1])
    nengo.Connection(agent.Error.net.errornode, agent.Critic.net.conn.learning_rule)
    nengo.Connection(agent.Error.net.errornode, agent.Actor.net.conn.learning_rule)

    # add Probes
    errorprobe = nengo.Probe(agent.Error.net.errornode)
    envprobe = nengo.Probe(envstate)

if BACKEND == 'CPU':
    sim = nengo.Simulator(model, dt=env.timestep)

elif BACKEND == 'GPU':
    import nengo_ocl
    import pyopencl as cl
    # set device to avoid being prompted every time
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    sim = nengo_ocl.Simulator(model, context=ctx, dt=env.timestep)

elif BACKEND == 'LOIHI':
    import nengo_loihi
    sim = nengo_loihi.Simulator(model, dt=env.timestep)


with sim:
    sim.run(1000)

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

'''
plot trajectories
'''
if PLOT_TRAJECTORIES:
    import numpy as np
    from math import ceil

    episode_indices = np.where(sim.data[envprobe][:,3] == 1.0)
    episode_indices = np.append(episode_indices[0], max(sim.trange()) / env.timestep)

    plot_columns = 3
    plot_rows = ceil(len(episode_indices) / plot_columns)

    prior = 0
    n = 1

    plt.figure()

    for index in episode_indices:
        vx = sim.data[envprobe][int(prior):int(index),1]
        vy = sim.data[envprobe][int(prior):int(index),2]
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
        plt.title("Trajectory %d-%dms" % (int(prior), int(index)))

        prior = index + 1
        n += 1

    plt.show()
