import nengo
import nengo_loihi

with nengo.Network() as net:
    net.ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    

with nengo.Simulator(net) as sim:
    sim.run(30)

print("done")
