import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import nengo
import nengo_loihi
nengo_loihi.set_defaults()


with nengo.Network() as net:
	a = nengo.Ensemble(n_neurons=1000, dimensions=1)
	b = nengo.Ensemble(n_neurons=1000, dimensions=1)
	conn = nengo.Connection(a,b, function=lambda x: [0])
	conn.learning_rule_type=nengo.PES()
	aprobe = nengo.Probe(a)
	bprobe = nengo.Probe(b)

with nengo_loihi.Simulator(net, precompute=False) as sim:
	sim.run(1000)

t = sim.trange()
adata = sim.data[aprobe]
bdata = sim.data[bprobe]

fig = plt.figure()
plt.plot(t, adata)
plt.plot(t, bdata)
fig.savefig('speedtest.png')
