import matplotlib.pyplot as plt
import numpy as np
import nengo
from Networks import CriticNet, ErrorNode, Switch
from util import simulate_with_backend

class Env2D:
    '''
    2D Environment for our actor tests
    '''

    def __init__(self):
        '''
        environment
        '''
        self.dt = 2e-4
        self.s = 0.3
        self.t = 0
        self.T_m = 30
        self.T_r = 1000
        self.T_c = 0
        self.d = 2
        self.R = 1

        '''
        platform
        '''
        self.platformsize = 0.1
        self.platformloc = np.array([0,0], dtype='float')

        '''
        mouse
        '''
        self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]], dtype='float')
        self.a = self.actions[2]
        self.l = np.array([-1, 0], dtype='float')
        self.mousepos = self.reset()

    def step(self):
        reset = 0
        reward = 0

        self.t += self.dt

        if self.goalReached():
            reward = self.R

            if self.T_c == self.T_r:
                self.mousepos = self.reset()
                self.T_c = 0
                reset = 1
            else:
                self.T_c += 1
        else:
            self.mousepos += (self.a / np.sqrt(np.sum(self.a**2))) * (self.s * self.dt) # take optimal step self.a

        return np.array([self.mousepos[0], self.mousepos[1], reward, reset])

    def reset(self):
        self.t = 0
        return np.copy(self.l) # copy because python sucks and we need a copy of the array, not reference (yes, this is a call out)

    def goalReached(self):
        goalDist = abs(np.sqrt(np.sum((self.platformloc - self.mousepos)**2)))
        return goalDist < self.platformsize / 2

env = Env2D()
BACKEND = 'GPU'
dt = 1e-3
T = 160
discount = 0.9995

with nengo.Network() as net:
    envnode = nengo.Node(lambda t: env.step(), size_out=4)
    in_ens = nengo.Ensemble(n_neurons=1000, dimensions=2, radius=2)
    critic = CriticNet(in_ens, n_neuron_out=1000, lr=1e-5)
    error = ErrorNode(discount=discount)
    switch = Switch(1)

    nengo.Connection(envnode[:2], in_ens)
    nengo.Connection(envnode[2], error.net.errornode[0], synapse=0)
    nengo.Connection(critic.net.output, error.net.errornode[1], synapse=0)
    nengo.Connection(switch.net.switch, error.net.errornode[2], synapse=0)
    nengo.Connection(error.net.errornode[1], error.net.errornode[3], synapse=0)
    nengo.Connection(envnode[3], error.net.errornode[4], synapse=0)
    nengo.Connection(error.net.errornode[0], critic.net.conn.learning_rule, transform=-1)

    envprobe = nengo.Probe(envnode)
    criticprobe = nengo.Probe(critic.net.output)
    errorprobe = nengo.Probe(error.net.errornode)
    learnprobe = nengo.Probe(critic.net.conn)

try:
    sim = simulate_with_backend(BACKEND, net, T, dt)
except Exception as e:
    print(e)
    print("WARNING: Falling back to CPU backend.")
    sim = simulate_with_backend('CPU', net, T, dt)

t = sim.trange()

import pandas as pd
t = sim.trange()
sim_error = sim.data[errorprobe][:,0]
state = sim.data[envprobe][:,0]
reward = sim.data[envprobe][:,1]
criticout = sim.data[criticprobe]
conndata = sim.data[learnprobe]

plt.figure(figsize=(12,10))

plt.subplot(511)
plt.plot(t, sim.data[envprobe][:,0], label='x-pos')
plt.plot(t, sim.data[envprobe][:,1], label='y-pos')
plt.legend()

plt.subplot(512)
plt.plot(t, state, label='position')
plt.plot(t, reward, label='reward')
plt.plot(t, criticout, label='value')
plt.plot(t, conndata, label='delta')
plt.legend()

plt.subplot(513)
plt.plot(t, sim_error, label='error')
plt.legend()

plt.subplot(514)
plt.title('error node inputs')
plt.plot(t, reward, label='reward')
# plt.plot(t, criticout, label='value')
plt.plot(t, pd.Series(error.valuemem).rolling(5).mean(), label="value", alpha=.5)
plt.plot(t, sim_error, label='error')
# plt.plot(t, sim.data[errorprobe][:,1], label='state')
plt.plot(t, pd.Series(error.statemem).rolling(5).mean(), label="state", alpha=.5)
# plt.plot(t, sim.data[envprobe][:,2], label='reset')
plt.legend()

plt.subplot(515)
plt.plot(t, criticout)
plt.title("Critic Value Prediction")

plt.tight_layout()
plt.show()
