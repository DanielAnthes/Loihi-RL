import matplotlib.pyplot as plt
import nengo

from Environment import Maze
from Agent import Mouse

BACKEND = 'GPU' # choice of CPU, GPU and LOIHI



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

    # add Probes
    errorprobe = nengo.Probe(agent.Error.net.errornode)
    envprobe = nengo.Probe(envstate)

if BACKEND == 'CPU':
    sim = nengo.Simulator(model)

elif BACKEND == 'GPU':
    import nengo_ocl
    import pyopencl as cl
    # set device to avoid being prompted every time
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    sim = nengo_ocl.Simulator(model, context=ctx)

elif BACKEND == 'LOIHI':
    import nengo_loihi
    sim = nengo_loihi.Simulator(model)


with sim:
    sim.run(100)

plt.figure()
plt.subplot(311)
plt.plot(sim.trange(), sim.data[errorprobe])
plt.title("Error Signal")
plt.subplot(312)
plt.plot(sim.trange(), sim.data[envprobe][:,:-1])
plt.legend(["xloc", "yloc", "reward", "done"])
plt.subplot(313)
plt.plot(sim.trange(), sim.data[envprobe][:,2])
plt.title("Reward")
plt.show()