import nengo
import numpy as np

class ActorNet:
    '''
    TODO: INSERT INSIGHTFUL DESCRIPTION HERE
    '''

def __init__(self, n_pc, n_neuron_in, n_neuron_out):
    '''
    Initialize actor net as a nengo network object
    
        PARAMS:
            n_pc            -   number of place cells
            n_neuron_in     -   number of neurons in Ensemble encoding input
            n_neuron_out    -   number of neurons in Ensemble encoding output
    '''
    with nengo.Network() as net:
        net.input = nengo.Ensemble(n_neurons=n_neuron_in, dimensions=n_pc, radius=np.sqrt(n_pc))
        net.output = nengo.Ensemble(n_neurons=n_neuron_out, dimensions=8, radius=np.sqrt(8))
        net.conn = nengo.Connection(net.input, net.output) # TODO: add learning here
    self.net = net


class CriticNet:
    '''
    TODO: INSERT INSIGHTFUL DESCRIPTION HERE
    '''

    def __init__(self, n_pc, n_neuron_in, n_neuron_out):
        '''
        initialize critic net as a nengo network object

        PARAMS:
            n_pc            -   number of place cells
            n_neuron_in     -   number of neurons in Ensemble encoding input
            n_neuron_out    -   number of neurons in Ensemble encoding output

        Note:   Radius of input population is set to sqrt(n_pc). The reasoning behind this is that
                the maximum activation of each place cell is 1 (exp(0)=1). However, this becomes a
                large value if many place cells are used. Since it is impossible for all place cells to
                give input value 1 at the same time this may be wasteful, choose smaller radius?
        '''
        with nengo.Network() as net:
            net.input = nengo.Ensemble(n_neurons=n_neuron_in, dimensions=n_pc, radius=np.sqrt(n_pc))
            net.output = nengo.Ensemble(n_neurons=n_neuron_out, dimensions=1)
            net.conn = nengo.Connection(net.input, net.output) # TODO add learning here
        self.net = net
