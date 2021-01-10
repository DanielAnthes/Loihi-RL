import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nengo

from env1d import Marathon as ENV
from Learning import TDL
from Networks import ErrorNode, Switch

STEPS = 800
LR = 1e-6
# point were added noise should be ~0, using a logistic decay
NOISE_LENGTH = 700
env = ENV()

def critic_value(t, pos):
    beta = 0.66
    x = float(env.goal) - float(pos)
    # magic numbers taken from looking at plots
    f = np.exp(-x**2 / (2 * 0.4**2))
    g = 0.9**float(((np.abs(x) / 0.002) / 10))

    return beta * f + (1-beta) * g

with nengo.Network() as model:
    # set up ENV and position ensemble
    envnode = nengo.Node(
        lambda t,action: env.step(action),
        size_in=1,
        size_out=3
    )
    # position encoder
    pos = nengo.Ensemble(
        n_neurons=500,
        dimensions=1,
        radius=1.3
    )

    # actor ensemble to learn
    actor = nengo.Ensemble(
        n_neurons=300, 
        dimensions=1, 
        radius= 1.3
    )
    # noise to enforce exploration in early stages 
    # before actor diverges stronger from 0
    noise = nengo.Node(
        lambda t: (
            (1 - 1 / ( 1 + np.exp(-.04*(t-NOISE_LENGTH/2)) ))
            * np.random.normal(0,1.2,(1,)) )
    )
    nengo.Connection(noise, actor)

    # perfect critic node
    critic = nengo.Node(
        critic_value,
        size_in=1, size_out=1
    )
    
    # feed position into ensemble
    nengo.Connection(envnode[0], pos)
    # feed posiiton into critic
    nengo.Connection(envnode[0], critic)
    # connect actor to envnode
    nengo.Connection(actor, envnode)
    # learning connection to move to goal
    conn = nengo.Connection(
        pos, actor,
        function=lambda x: 1,
        solver=nengo.solvers.NoSolver(weights=False),
        # learning_rule_type=TDL(learning_rate=1e-6)
        learning_rule_type=nengo.PES(LR)
    )

    # Error related
    switch = Switch(1)
    err = ErrorNode(discount=0.9)
    # reward into error
    nengo.Connection(envnode[1], err.net.errornode[0], synapse=0)
    # critic into error
    nengo.Connection(critic, err.net.errornode[1], synapse=0.2)
    # switch learning on or off
    nengo.Connection(switch.net.switch, err.net.errornode[2])
    # feedback connection
    nengo.Connection(err.net.errornode[1], err.net.errornode[3], synapse=0)
    # reset signal
    nengo.Connection(envnode[2], err.net.errornode[4], synapse=0)

    # error 2 agent
    nengo.Connection(err.net.errornode[0], conn.learning_rule[0], transform=-1)

    pEnv = nengo.Probe(envnode, synapse=0)
    pActor = nengo.Probe(actor, synapse=0)
    pCritic = nengo.Probe(critic, synapse=0)
    pError = nengo.Probe(err.net.errornode, synapse=0)

with nengo.Simulator(model) as sim:
    sim.run(STEPS)



t = sim.trange()
error = sim.data[pError][:,0]
state = sim.data[pEnv][:,0]
reward = sim.data[pEnv][:,1]
actorout = sim.data[pActor]
criticout = sim.data[pCritic]

plt.figure(figsize=(12,10))
plt.subplot(411)
plt.plot(t, actorout, label='action', lw=.6, alpha=.666)
success = np.where(reward>0)
plt.vlines(t[success], -1, 1,
            linestyles='dotted', label='reward', colors='green')
plt.plot(t, state, label='position')
plt.hlines(0, 0, STEPS, colors='black', alpha=0.6)
# plt.plot(t, criticout, label='value')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(t, error, label='delta')
# plt.plot(env.actionmem, env.pmem, label='p against action')
# plt.plot(t, learnswitch, label='learning')
plt.legend(loc='upper left')

plt.subplot(413)
plt.title('error node inputs')
plt.vlines(t[success], -1.5, 1.5,
            linestyles='dotted', label='reward', colors='green')
# plt.plot(t, criticout, label='value')
plt.plot(t, pd.Series(err.valuemem).rolling(5).mean(), 
            label="value", alpha=.5)
plt.plot(t, error, label='delta')
# plt.plot(t, sim.data[errorprobe][:,1], label='state')
plt.plot(t, pd.Series(err.statemem).rolling(5).mean(),
            label="state", alpha=.5)
# plt.plot(t, sim.data[envprobe][:,2], label='reset')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(t, criticout, label='Critic Output')
plt.plot(t, state, label='position')
plt.title("Critic Value Prediction")
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()