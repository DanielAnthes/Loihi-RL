# %%
import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_ocl
from Networks import CriticNet, ErrorNode, Switch
from util import simulate_with_backend

'''
Note: if reward delay in combination with resetting leads to no learning try staying at goal for multiple steps before resetting
'''

class TestEnv:

    def __init__(self):
        self.stepsize = 0.05
        self.pathlen = 2
        self.agentpos = 0
        self.reward = 1
        self.goalcounter = 0
        self.reset = 10

    def step(self):
        self.agentpos += self.stepsize
        if self.goalReached():
            reward = self.reward
            if self.goalcounter == self.reset:
                self.agentpos = 0
                self.goalcounter = 0
            else:
                self.goalcounter += 1
                self.agentpos = self.pathlen
        else:
            reward = 0
        return np.array([self.agentpos, reward])
    
    def goalReached(self):
        return abs(self.agentpos - self.pathlen) < self.stepsize


env = TestEnv()

with nengo.Network() as net:
    envnode = nengo.Node(lambda t: env.step(), size_out=2)
    in_ens = nengo.Ensemble(n_neurons=1000, radius=2, dimensions=1)  # encodes position
    critic = CriticNet(in_ens, n_neuron_out=100, lr=1e-4)
    error =  ErrorNode(discount=0.9)
    switch =  Switch(1)  # needed for compatibility with error implementation

    nengo.Connection(envnode[0], in_ens)


    # error node connections
    nengo.Connection(envnode[1], error.net.errornode[0])
    nengo.Connection(critic.net.output, error.net.errornode[1])
    nengo.Connection(switch.net.switch, error.net.errornode[2])
    nengo.Connection(error.net.errornode[1], error.net.errornode[3])
    
    # error to critic
    nengo.Connection(error.net.errornode[0], critic.net.conn.learning_rule)

    # Probes
    envprobe = nengo.Probe(envnode)
    criticprobe = nengo.Probe(critic.net.output)
    errorprobe = nengo.Probe(error.net.errornode)
    learnprobe = nengo.Probe(critic.net.conn)

sim = simulate_with_backend('GPU', net, 40, 0.1)


# %%
t = sim.trange()
sim_error = sim.data[errorprobe][:,0]
state = sim.data[envprobe][:,0]
reward = sim.data[envprobe][:,1]
criticout = sim.data[criticprobe]
conndata = sim.data[learnprobe]

# %%
plt.figure(figsize=(12,8))
plt.subplot(211)
plt.plot(t, state, label='position')
plt.plot(t, reward, label='reward')
plt.plot(t, criticout, label='value')
plt.plot(t, conndata, label='delta')
plt.legend()
plt.subplot(212)
plt.plot(t, sim_error, label='error')
plt.legend()

plt.figure()
plt.title('error node inputs')
plt.plot(t, reward, label='reward')
plt.plot(t, criticout, label='value')
plt.plot(t, sim_error, label='error')
plt.plot(t, sim.data[errorprobe][:,1], label='state')
plt.legend()

plt.figure()
plt.plot(t, criticout)
plt.title("Critic Value Prediction")
plt.show()


