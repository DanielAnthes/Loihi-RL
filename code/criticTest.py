import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nengo
from Networks import CriticNet, ErrorNode, Switch
from util import simulate_with_backend
from Environment import TestEnv
import pathlib
import nengo_loihi
nengo_loihi.set_defaults()

'''
Note: if reward delay in combination with resetting leads to no learning try staying at goal for multiple steps before resetting
'''

env = TestEnv(invert=False)
BACKEND = 'LOIHI'
dt = 0.001
duration = 400
discount = 0.9

with nengo.Network() as net:
    nengo_loihi.add_params(net)
    envnode = nengo.Node(lambda t: env.step(), size_out=3)
    in_ens = nengo.Ensemble(n_neurons=1000, radius=2, dimensions=1)  # encodes position
    critic = CriticNet(in_ens, n_neuron_out=1000, lr=1e-5)
    error =  ErrorNode(discount=discount)  # seems like a reasonable value to have a reward gradient over the entire episode
    switch =  Switch(state=1, switch_off=False, switchtime=duration/2)  # needed for compatibility with error implementation

    nengo.Connection(envnode[0], in_ens)

    # move ensembles to chip
    net.config[in_ens].on_chip=True


    # error node connections
    # reward = input[0] value = input[1] switch = input[2] state = input[3] reset = input[4].astype(int)
    nengo.Connection(envnode[1], error.net.errornode[0], synapse=0) # reward connection
    nengo.Connection(critic.net.output, error.net.errornode[1], synapse=0) # value prediction
    nengo.Connection(switch.net.switch, error.net.errornode[2], synapse=0) # learning switch
    nengo.Connection(error.net.errornode[1], error.net.errornode[3], synapse=0) # feed value into next step
    nengo.Connection(envnode[2], error.net.errornode[4], synapse=0) # propagate reset signal

    # error to critic
    nengo.Connection(error.net.errornode[0], critic.net.conn.learning_rule, transform=-1)

    # Probes
    envprobe = nengo.Probe(envnode, sample_every=dt)
    criticprobe = nengo.Probe(critic.net.output, sample_every=dt)
    errorprobe = nengo.Probe(error.net.errornode, sample_every=dt)
    switchprobe = nengo.Probe(switch.net.switch, sample_every=dt)

try:
    sim = simulate_with_backend(BACKEND, net, duration, dt) # use default dt
except Exception as e:
    print(e)
    print("WARNING: Falling back to CPU backend")
    sim = simulate_with_backend('CPU', net, duration, dt) # use default dt



t = sim.trange()
sim_error = sim.data[errorprobe][:,0]
state = sim.data[envprobe][:,0]
reward = sim.data[envprobe][:,1]
criticout = sim.data[criticprobe]
learnswitch = sim.data[switchprobe]

dump = pathlib.Path('../dumps/')
dump.mkdir(exist_ok=True)

np.savetxt(dump / "trange.csv", t, delimiter=",")
np.savetxt(dump / "sim_error.csv", sim_error, delimiter=",")
np.savetxt(dump / "state.csv", state, delimiter=",")
np.savetxt(dump / "reward.csv", reward, delimiter=",")
np.savetxt(dump / "criticout.csv", criticout, delimiter=",")
np.savetxt(dump / "learnswitch.csv", learnswitch, delimiter=",")

