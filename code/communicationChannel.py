import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nengo
import nengo_loihi
from nengo.processes import WhiteSignal
nengo_loihi.set_defaults()

with nengo.Network(label="PES learning") as model:
    # Randomly varying input signal
    stim = nengo.Node(WhiteSignal(60, high=5), size_out=1)

    # Connect pre to the input signal
    pre = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(stim, pre, synapse=0)
    post = nengo.Ensemble(100, dimensions=1)

    # When connecting pre to post,
    # create the connection such that initially it will
    # always output 0. Usually this results in connection
    # weights that are also all 0.
    conn = nengo.Connection(
        pre,
        post,
        function=lambda x: [0],
        learning_rule_type=nengo.PES(learning_rate=2e-4),
    )

    # Calculate the error signal with another ensemble
    error = nengo.Ensemble(100, dimensions=1)

    # Error = actual - target = post - pre
    nengo.Connection(post, error)
    nengo.Connection(pre, error, transform=-1)

    # Connect the error into the learning rule
    nengo.Connection(error, conn.learning_rule)

    stim_p = nengo.Probe(stim)
    pre_p = nengo.Probe(pre, synapse=0)
    post_p = nengo.Probe(post, synapse=0)
    error_p = nengo.Probe(error, synapse=0)


with nengo_loihi.Simulator(model) as sim:
    sim.run(10)
t = sim.trange()


def plot_decoded(t, data):
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(t, data[stim_p].T[0], label="Input")
    plt.plot(t, data[pre_p].T[0], label="pre")
    plt.plot(t, data[post_p].T[0], label="post")
    plt.ylabel("Decoded output")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(t, data[error_p])
    plt.ylim(-1, 1)
    plt.legend(("Error signal",), loc="best")
    return fig

fig = plot_decoded(t, sim.data)
plt.savefig("pestest.png")
