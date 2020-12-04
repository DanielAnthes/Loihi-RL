import nengo
import numpy as np
from numpy.random import choice, random
import Learning

class ActorNet:
    '''
    TODO: INSERT INSIGHTFUL DESCRIPTION HERE
    '''

    def __init__(self, n_pc, input_node, n_neuron_out, lr):
        '''
        Initialize actor net as a nengo network object

            PARAMS:
                n_pc            -   number of place cells
                n_neuron_in     -   number of neurons in Ensemble encoding input
                n_neuron_out    -   number of neurons in Ensemble encoding output
        '''
        with nengo.Network() as net:
            net.output = nengo.Ensemble(n_neurons=n_neuron_out, dimensions=8, radius=np.sqrt(8))
            net.conn = nengo.Connection(input_node, net.output,
                                        function=lambda x: [0]*8,
                                        solver=nengo.solvers.LstsqL2(weights=True),
                                        learning_rule_type=Learning.TDL(learning_rate=lr))
        self.net = net


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
        with nengo.Network() as net:
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
 
        with nengo.Network() as net:
            net.errornode = nengo.Node(lambda t, input: self.update(input), size_in=5, size_out=2)

        self.net = net

    def update(self, input):
        reward = input[0]
        value = input[1]
        switch = input[2]
        state = input[3]
        reset = input[4]

        if state is None:
            return [0, value]  # no error without prediction

        delta = reward + self.discount*value - state
        if reset:
            return [0,0]
        else:
            return [delta*switch, value]


class DecisionNode:
    '''
    TODO: combine with actor? model properly?
    Node for use within a Nengo network.
    Takes as input a vector of neural activities and
    stochastically makes a choice for an action with probability proportional
    to the relative activations of the neural input
    '''
    def __init__(self, actions):
        '''
        INPUTS:
            actions     -   a vector of possible actions to choose from
        '''
        self.actions = actions
        self.lastaction = None
        self.activation = np.array([np.zeros(8)])
        self.probability = np.array([np.zeros(8)])

        with nengo.Network() as net:
            net.choicenode = nengo.Node(lambda t,x: self.chooseAction(x), size_in=len(actions), size_out=1)
        self.net = net

    def chooseAction(self, activation_in):
        '''
        makes a random choice given neuronal activities for choices as input

        INPUT:
            activation_in   - neuronal acitivies (numpy array)
        RETURNS:
            decision    -   integer index of choice made
        '''

        self.activation = np.append(self.activation, [activation_in], axis=0)

        coin = random()
        if coin > .25 and self.lastaction is not None:  # repeat last action
                decision = self.lastaction
        else:
            pos_activation = np.maximum(activation_in, 0)
            if np.sum(pos_activation) == 0:  # avoid division by zero
                decision = choice(self.actions)
            else:
                probs = pos_activation / np.sum(pos_activation)
                self.probability = np.append(self.probability, [probs], axis=0)
                decision = choice(self.actions, p=probs)
        self.lastaction = decision
        return decision


# DEPRECATED
class PlaceCells:
    '''
    TODO: currently assumed "given" and implemented as Node, opportunity to extend the simulation here
    Simple implementation of place cells as 2d gaussians uniformly distributed over a grid
    activation is a function of the distance of the position to be encoded from the preferred location (mean) of the gaussian
    max activation is 1
    '''

    def __init__(self, nx, ny, diameter, sigma):
        '''
        compute locations of place cells, for convenience, place cells are laid out on a square overlapping the arena
        this means that some place cells lay outside of the arena. However, this ensures that place cells are evenly spaced
        alternatively, initialize place cells given polar coordinates and arrange on concentric circles
        but I do not know how to make them evenly spaced that way
        for now the environment is assumed to be square and centered at (0,0)

        INPUTS:
            nx          -   number of place cells in x direction
            ny          -   number of place cells in y direction
            diameter    -   diameter of the environment
        '''
        self.nx = nx
        self.ny = ny
        self.n_place_cells = nx * ny
        self.sigma = sigma
        x_coords = np.linspace(-diameter/2,diameter/2, nx)
        y_coords = np.linspace(-diameter/2, diameter/2, ny)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.cell_locs = np.array([xx.flatten(), yy.flatten()])

        with nengo.Network() as net:
            net.placecells = nengo.Node(lambda t,x: self.place_cell_activation(x), size_in=2, size_out=self.n_place_cells)

        self.net = net

    def place_cell_activation(self, pos):
        '''
        compute activity of all place cells for a given position,
        firing rate for each place cell is given by a gaussian centered around it's preferred
        location with width sigma
        returns a vector of activation of all place cells

        INPUTS:
            pos         -   position to be encoded

        RETURNS:
            activations -   vector encoding activities of all place cells
        '''
        pos = np.expand_dims(pos,1)
        enumerator = np.sqrt(np.sum( (pos - self.cell_locs)**2 ,axis=0))**2
        denominator = 2 * self.sigma**2
        activations = np.exp(-(enumerator / denominator))
        return activations


class Switch:
    def __init__(self, state=1):
        self.state = state

        with nengo.Network() as net:
            net.switch = nengo.Node(lambda t: self.state, size_out=1)
        self.net = net
