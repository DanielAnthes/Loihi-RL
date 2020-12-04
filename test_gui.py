import numpy as np
import nengo

class CriticNet:
    '''
    TODO: INSERT INSIGHTFUL DESCRIPTION HERE
    '''

    def __init__(self, input_node, n_neuron_out, lr):
        '''
        initialize critic net as a nengo network object

        PARAMS:
            n_pc            -   number of place cells
            n_neuron_in     -   number of neurons in Ensemble encoding input
            n_neuron_out    -   number of neurons in Ensemble encoding output
        '''
        with nengo.Network(label="critic") as net:
            net.output = nengo.Ensemble(n_neurons=n_neuron_out, dimensions=1)
            net.conn = nengo.Connection(input_node, net.output, function=lambda x: [0])
            net.conn.learning_rule_type = nengo.PES(learning_rate=lr)
        self.net = net


class ErrorNode:
    '''
    Computes delta as described in Foster et al.
    To avoid recomputation of the states and to enable error feedback during simulation
    To do this we use a self loop feeding back the previous value prediction V and  use
    it with the new prediction V_t+1 to compute delta as:

        delta = Reward + discount*V_t+1 - V_t

    Probably, this will cause the error signal to be delayed and applied at the wrong time.
    TODO check whether this still works or is complete nonsense
    '''
    def __init__(self, discount):
        self.discount = discount
 
        with nengo.Network(label="error") as net:
            net.errornode = nengo.Node(lambda t, input: self.update(input), size_in=4, size_out=2)

        self.net = net

    def update(self, input):
        reward = input[0]
        value = input[1]
        switch = input[2]
        state = input[3]

        if state is None:
            return [0, value]  # no error without prediction

        delta = reward if reward > 0 else self.discount*value - state
        # value = 0 if reward > 0 else value  # fix bleeding into novel episodes

        return [delta*switch, value]


class Switch:
    def __init__(self, state=1):
        self.state = state

        with nengo.Network(label="switch") as net:
            net.switch = nengo.Node(lambda t: self.state, size_out=1)
        self.net = net
        
        
class TestEnv:

    def __init__(self):
        self.stepsize = 0.0002  # with standard dt and track length 2m 
                                # this comes out to a runtime of 10 seconds
        self.pathlen = 2
        self.agentpos = 0
        self.reward = 1
        self.goalcounter = 0
        self.reset = 1000

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

with nengo.Network() as model:
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


    

