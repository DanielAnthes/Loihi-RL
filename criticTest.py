import numpy as np
import nengo
from Networks import CriticNet, ErrorNode, Switch

'''
Note: if reward delay in combination with resetting leads to no learning try staying at goal for multiple steps before resetting
'''

class TestEnv:

    def __init__(self):
        self.stepsize = 0.1
        self.pathlen = 1
        self.agentpos = 0
        self.reward = 1

    def step(self):
        self.agentpos += self.stepsize
        if self.goalReached():
            reward = self.reward
            self.agentpos = 0
        else:
            reward = 0
        return np.array([self.agentpos, reward])
    
    def goalReached(self):
        return abs(self.agentpos - self.pathlen) < self.stepsize


env = TestEnv()

with nengo.Network() as net:
    envnode = nengo.Node(function = lambda t: env.step(), n_out=2)
    in_ens = nengo.Ensemble(n_neurons=100, radius=1, dimensions=1)  # encodes position
    critic = CriticNet(in_ens, n_neuron_out=100, lr=1e-4)
    error =  ErrorNode(discount=0.9)
    switch =  Switch(1)  # needed for compatibility with error implementation

    nengo.Connection(envnode[0], in_ens)

    # error node connections
    nengo.Connection(envnode[1], error.net.errornode[0])
    nengo.Connection(critic.net.output, error.net.errornode[1])
    nengo.Connection(switch.net.switch, error.net.errornode[2])
    nengo.Connection(error.net.errornode[1], error.net.errornode[4])
    
