import nengo
import nengo_loihi

with nengo.Network() as net:
    nengo_loihi.add_params(net)
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    net.config[ens].on_chip = True

with nengo_loihi.Simulator(net) as sim:
    sim.run(30)

print("done")
