from Environment import TestEnv
from Agent import Mouse
from Networks import CriticNet, ErrorNode, Switch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import nengo
import nxsdk
import nengo_loihi
nengo_loihi.set_defaults()

env = TestEnv(invert=False)
discount = 0.9
duration=400

with nengo.Network() as net:
    nengo_loihi.add_params(net)
    envnode = nengo.Node(lambda t: env.step(), size_out=3)

    inputens = nengo.Ensemble(n_neurons=1000, dimensions=2, radius=1)
    inprobe = nengo.Probe(inputens)

    critic = CriticNet(inputens, n_neuron_out=1000, lr=1e-5)
    net.config[inputens].on_chip=True

    error =  ErrorNode(discount=discount)  # seems like a reasonable value to have a reward gradient over the entire episode
    switch =  Switch(state=1, switch_off=False, switchtime=duration/2)  # needed for compatibility with error implementation

    # connections
    nengo.Connection(envnode[1], error.net.errornode[0], synapse=0) # reward connection
    nengo.Connection(critic.net.output, error.net.errornode[1], synapse=0) # value prediction
    nengo.Connection(switch.net.switch, error.net.errornode[2], synapse=0) # learning switch
    nengo.Connection(error.net.errornode[1], error.net.errornode[3], synapse=0) # feed value into next step
    nengo.Connection(envnode[2], error.net.errornode[4], synapse=0) # propagate reset signal

    nengo.Connection(error.net.errornode[0], critic.net.conn.learning_rule, transform=-1)


with nengo_loihi.Simulator(net) as sim:
    sim.run(duration)

