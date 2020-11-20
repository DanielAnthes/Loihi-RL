import nengo
import util

import matplotlib.pyplot as plt

from Environment import Maze
from Agent import Mouse
from Networks import Switch

BACKEND = 'GPU' # choice of CPU, GPU and LOIHI
PLOT_TRAJECTORIES = True # True to plot the trajectories the mouse took


# set up simulation, connect networks
env = Maze()

with nengo.Network() as model:
    agent = Mouse(env, 23, 23, act_lr=0.01, crit_lr=0.01)

    # TODO add error node
    # environment node, step function expects integer so need to cast from float
    envstate = nengo.Node(lambda time, action: env.step(int(action)), size_in=1, size_out=5)

    # add node to control learning
    model.switch = Switch(state=1)

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
    nengo.Connection(model.switch.net.switch, agent.Error.net.errornode[2])
    nengo.Connection(agent.Error.net.errornode, agent.Critic.net.conn.learning_rule)
    nengo.Connection(agent.Error.net.errornode, agent.Actor.net.conn.learning_rule)

    # add Probes
    errorprobe = nengo.Probe(agent.Error.net.errornode)
    envprobe = nengo.Probe(envstate)
    switchprobe = nengo.Probe(model.switch.net.switch)

sim = util.simulate_with_backend(BACKEND, model, duration=100, timestep=env.timestep)

util.plot_sim(sim, envprobe, errorprobe, switchprobe)
util.plot_value_func(model, agent, env, BACKEND)
util.plot_trajectories(sim, env, envprobe)
plt.show()