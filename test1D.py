import matplotlib.pyplot as plt
import numpy as np
import nengo
from Networks import CriticNet, ErrorNode, Switch
from util import simulate_with_backend
from Environment import TestEnvActor
import Learning

'''
Note: if reward delay in combination with resetting leads to no learning try staying at goal for multiple steps before resetting
'''

BACKEND = 'CPU'
dt = 0.001
duration = 200
discount = 0.9995
env = TestEnvActor(dt=dt, trial_length=40)

with nengo.Network() as net:
    envnode = nengo.Node(lambda t, v: env.step(v), size_in=1, size_out=3)
    in_ens = nengo.Ensemble(n_neurons=1000, radius=2, dimensions=1)  # encodes position
    actor = nengo.Ensemble(n_neurons=1000, radius=1, dimensions=1)
    critic = CriticNet(in_ens, n_neuron_out=1000, lr=1e-5)
    error =  ErrorNode(discount=discount)  # seems like a reasonable value to have a reward gradient over the entire episode
    switch =  Switch(state=1, switch_off=False, switchtime=duration/2)  # needed for compatibility with error implementation

    nengo.Connection(envnode[0], in_ens)
    conn = nengo.Connection(in_ens, actor, function=lambda x: [0], solver=nengo.solvers.LstsqL2(weights=True), learning_rule_type=Learning.TDL(learning_rate=1e-8))
    nengo.Connection(actor, envnode)


    # error node connections
    # reward = input[0] value = input[1] switch = input[2] state = input[3] reset = input[4].astype(int)
    nengo.Connection(envnode[1], error.net.errornode[0], synapse=0) # reward connection
    nengo.Connection(critic.net.output, error.net.errornode[1], synapse=0) # value prediction
    nengo.Connection(switch.net.switch, error.net.errornode[2], synapse=0) # learning switch
    nengo.Connection(error.net.errornode[1], error.net.errornode[3], synapse=0) # feed value into next step
    nengo.Connection(envnode[2], error.net.errornode[4], synapse=0) # propagate reset signal

    # error to critic
    nengo.Connection(error.net.errornode[0], critic.net.conn.learning_rule, transform=-1)
    nengo.Connection(error.net.errornode[0], conn.learning_rule)

    # Probes
    envprobe = nengo.Probe(envnode)#, sample_every=0.5)
    criticprobe = nengo.Probe(critic.net.output)#, sample_every=0.5)
    actorprobe = nengo.Probe(actor)#, sample_every=0.5)
    errorprobe = nengo.Probe(error.net.errornode)#, sample_every=0.5)

try:
    sim = simulate_with_backend(BACKEND, net, duration, dt) # use default dt
except Exception as e:
    print(e)
    print("WARNING: Falling back to CPU backend")
    sim = simulate_with_backend('CPU', net, duration, dt) # use default dt


t = sim.trange()
#t = np.arange(np.floor(duration / 0.5))
p_pos = sim.data[envprobe][:,0]
p_reward = sim.data[envprobe][:,1]
p_delta = sim.data[errorprobe][:,0]
p_delta_positive = np.where(p_delta > 0, p_delta, 0)
p_delta_negative = np.where(p_delta < 0, p_delta, 0)
p_delta_naught = np.where(p_delta == .0, p_delta, 1.0)
p_prediction = sim.data[criticprobe]
p_activity = sim.data[actorprobe]

plt.figure()
plt.subplot(311)
plt.plot(t, p_pos, label="Position")
plt.plot(t, p_delta, label="Delta")
plt.plot(t, p_prediction, label="Critic")
plt.plot(t, p_reward, label="Reward")
plt.legend()
plt.subplot(312)
plt.plot(t, p_prediction, label="Critic")
plt.plot(t, p_activity, label="Actor")
plt.legend()
plt.subplot(313)
axes = plt.gca()
plt.scatter(t, p_delta_positive, s=1, marker='x', label="Positive Delta")
plt.scatter(t, p_delta_negative, s=1, marker='x', label="Negative Delta")
plt.scatter(t, p_delta_naught, s=1, marker='x', label="Naught Delta")
axes.set_ylim([-5e-2, 5e-2])
plt.legend()
plt.show()
