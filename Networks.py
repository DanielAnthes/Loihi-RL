import nengo
import numpy as np
from numpy.random import choice, random
import Learning

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
            net.conn = nengo.Connection(net.input, net.output,
                                        function=lambda x: random(8),
                                        solver=nengo.solvers.LstsqL2(weights=True),
                                        learning_rule_type=Learning.TDL(learning_rate=1e-6))    # TODO write the actual TDL rule, not Oja
                                        # learning_rule_type=nengo.Oja())                                                       # TODO if something goes wrong here take a good long look at the initial connection function
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
            net.conn = nengo.Connection(net.input, net.output, function=lambda x: random(1)) # TODO add learning here
            net.conn.learning_rule_type = nengo.PES()
        self.net = net


class ErrorNode:
    '''
    Computes delta as described in Foster et al.
    To avoid recomputation of the states and to enable error feedback during simulation
    To do this keep value prediction at time t: V_t as the state of the Error Node
    on receiving the reward and new prediction V_t+1 computes delta as:

        delta = Reward + discount*V_t+1 - V_t

    Probably, this will cause the error signal to be delayed and applied at the wrong time.
    TODO check whether this still works or is complete nonsense
    '''
    def __init__(self, discount):
        self.state = None
        self.discount = discount

        with nengo.Network() as net:
            net.errornode = nengo.Node(lambda t,input: self.update(input), size_in=2, size_out=1)

        self.net = net

    def update(self, input):
        reward = input[0]
        value = input[1]
        if self.state is None:
            self.state = value
            return 0 # no error without prediction
        delta = reward + self.discount*value - self.state
        self.state = value
        return delta


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
        pos_activation = np.sqrt(activation_in**2) # ensure positive activations TODO: ugly hack, this will probably cause problems later
        if np.sum(activation_in) == 0: # avoid division by zero
            decision = choice(self.actions)
        else:
            probs = pos_activation / np.sum(pos_activation)
            decision = choice(self.actions, p=probs)
        return decision


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