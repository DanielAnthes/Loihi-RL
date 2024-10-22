import numpy as np
import matplotlib.pyplot as plt
import nengo
import util

from Environment import Maze
from Agent import Mouse
from Networks import Switch

BACKEND = 'GPU' # choice of CPU, GPU and LOIHI
STEPS = 120  # each trial is max 30 seconds
N_PCX = 23  # N place cells in X direction
N_PCY = 23  # ibid. in Y direction
PLOT_TUNING = False  # Be aware that producing this plot is quite slow

# set up simulation, connect networks
env = Maze()

with nengo.Network() as model:
    agent = Mouse(env, N_PCX, N_PCY, act_lr=1e-6, crit_lr=1e-6)

    envstate = nengo.Node(lambda time, action: env.step(action), size_in=1, size_out=6)

    # add node to control learning
    model.switch = Switch(state=1)

    # compute place cell activations
    # nengo.Connection(envstate[:2], agent.PlaceCells.net.placecells)

    # place cells give input to actor and critic
    nengo.Connection(envstate[:2], agent.net.input)

    # take actor net as input to decision node
    nengo.Connection(agent.Actor.net.output, agent.DecisionMaker.net.choicenode)

    # execute action in environment
    nengo.Connection(agent.DecisionMaker.net.choicenode, envstate, synapse=0)

    # connect error node
    nengo.Connection(envstate[2], agent.Error.net.errornode[0])
    nengo.Connection(agent.Critic.net.output, agent.Error.net.errornode[1])
    nengo.Connection(model.switch.net.switch, agent.Error.net.errornode[2])
    nengo.Connection(agent.Error.net.errornode[1], agent.Error.net.errornode[3]) # recurrent connection to save last state; TODO: synapse=0 if transmission too bad
    nengo.Connection(envstate[5], agent.Error.net.errornode[4]) # pass reset flag
    nengo.Connection(agent.Error.net.errornode[0], agent.Critic.net.conn.learning_rule)
    nengo.Connection(agent.Error.net.errornode[0], agent.Actor.net.conn.learning_rule)

    # add Probes
    errorprobe = nengo.Probe(agent.Error.net.errornode[0])
    envprobe = nengo.Probe(envstate)
    switchprobe = nengo.Probe(model.switch.net.switch)
    actorwprobe = nengo.Probe(agent.Actor.net.conn)
    criticwprobe = nengo.Probe(agent.Critic.net.conn)

    # Plot tuning curves
    if PLOT_TUNING: util.plot_tuning_curves(model, agent.net.input)

# CPU Fallback
try:
    sim = util.simulate_with_backend(BACKEND, model, duration=STEPS, timestep=env.timestep)
except Exception as e:
    print(e)
    print("WARNING: Falling back to CPU backend")
    BACKEND='CPU'
    sim = util.simulate_with_backend(BACKEND, model, duration=STEPS, timestep=env.timestep)

util.plot_sim(sim, envprobe, errorprobe, switchprobe)
#util.plot_value_func(model, agent, env, BACKEND)
util.plot_trajectories(sim, env, envprobe)
#util.plot_actions_by_activation(env, agent)
#util.plot_actions_by_probability(env, agent)
#util.plot_actions_by_decision(env)
#util.plot_weight_evolution_3d(sim, actorwprobe, title="Weight evolution of place cells to actor")
#util.plot_weight_evolution_2d(sim, criticwprobe, title="Weight evolution of place cells to critic")
#util.plot_place_cell(model, agent, env, BACKEND, [0.0, 0.0])
plt.show()
