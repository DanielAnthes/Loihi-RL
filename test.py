import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import nengo
import nxsdk
import nengo_loihi
nengo_loihi.set_defaults()

with nengo.Network() as net:
	nengo_loihi.add_params(net)
	a = nengo.Ensemble(n_neurons=1000, dimensions=1)
	net.config[a].on_chip=True
	probe = nengo.Probe(a)

with nengo_loihi.Simulator(net) as sim:
	sim.run(400)

pd = sim.data[probe]
t = sim.trange()


fig = plt.figure()
plt.plot(t, pd)

fig.savefig("test.png")
