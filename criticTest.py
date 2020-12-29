import matplotlib.pyplot as plt
import numpy as np
import nengo
from Networks import CriticNet, ErrorNode, Switch
from util import simulate_with_backend
import nengo_loihi
nengo_loihi.set_defaults()  # performance benefits

'''
Note: if reward delay in combination with resetting leads to no learning try staying at goal for multiple steps before resetting
'''

class TestEnv:

    def __init__(self):
        self.stepsize = 0.0002  # with standard dt and track length 2 this
                                # leads to episode length of 10 seconds
        self.pathlen = 2
        self.agentpos = 0
        self.reward = 1
        self.goalcounter = 0
        self.reset = 500

    def step(self):
        reset = 0
        self.agentpos += self.stepsize
        if self.goalReached():
            reward = self.reward
            if self.goalcounter == self.reset:
                self.agentpos = 0
                self.goalcounter = 0
            else:
                self.goalcounter += 1
                self.agentpos = self.pathlen

                if self.goalcounter == self.reset:
                    reset = 1
        else:
            reward = 0
            reset = 0
        return np.array([self.agentpos, reward, reset])

    def goalReached(self):
        return abs(self.agentpos - self.pathlen) < self.stepsize


env = TestEnv()
BACKEND = 'LOIHI'
dt = 0.001
duration = 80
discount = 0.9995

with nengo.Network() as net:
    envnode = nengo.Node(lambda t: env.step(), size_out=3)
    in_ens = nengo.Ensemble(n_neurons=1000, radius=2, dimensions=1)  # encodes position
    critic = CriticNet(in_ens, n_neuron_out=1000, lr=1e-5)
    error =  ErrorNode(discount=discount)  # seems like a reasonable value to have a reward gradient over the entire episode
    switch =  Switch(1)  # needed for compatibility with error implementation

    nengo.Connection(envnode[0], in_ens)


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
    envprobe = nengo.Probe(envnode)
    criticprobe = nengo.Probe(critic.net.output)
    errorprobe = nengo.Probe(error.net.errornode)
    learnprobe = nengo.Probe(critic.net.conn)

try:
    sim = simulate_with_backend(BACKEND, net, duration, dt) # use default dt
except Exception as e:
    print(e)
    print("WARNING: Falling back to CPU backend")
    sim = simulate_with_backend('CPU', net, duration, dt) # use default dt


import pandas as pd
t = sim.trange()
sim_error = sim.data[errorprobe][:,0]
state = sim.data[envprobe][:,0]
reward = sim.data[envprobe][:,1]
criticout = sim.data[criticprobe]
conndata = sim.data[learnprobe]

# plt.figure(figsize=(12,10))
# plt.subplot(411)
# plt.plot(t, state, label='position')
# plt.plot(t, reward, label='reward')
# plt.plot(t, criticout, label='value')
# plt.plot(t, conndata, label='delta')
# plt.legend()
# plt.subplot(412)
# plt.plot(t, sim_error, label='error')
# plt.legend()

# plt.subplot(413)
# plt.title('error node inputs')
# plt.plot(t, reward, label='reward')
# # plt.plot(t, criticout, label='value')
# plt.plot(t, pd.Series(error.valuemem).rolling(5).mean(), label="value", alpha=.5)
# plt.plot(t, sim_error, label='error')
# # plt.plot(t, sim.data[errorprobe][:,1], label='state')
# plt.plot(t, pd.Series(error.statemem).rolling(5).mean(), label="state", alpha=.5)
# # plt.plot(t, sim.data[envprobe][:,2], label='reset')
# plt.legend()

# plt.subplot(414)
# plt.plot(t, criticout)
# plt.title("Critic Value Prediction")

# plt.tight_layout()
# plt.show()
