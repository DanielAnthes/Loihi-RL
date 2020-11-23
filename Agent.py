import nengo
import numpy as np

from Networks import ActorNet, CriticNet, PlaceCells, DecisionNode, ErrorNode


class Mouse:
    '''
    Actor class, has an environment in which actions are performed
    Learns to take optimal actions in the environment by applying Actor Critic Learning
    '''

    def __init__(self, env, n1, n2, sigma=0.16, act_lr=6e-8, crit_lr=1e-4):
        '''
        TODO: agent needs error node for learning
        env - the environment to act in
        n1 - number of place cells in x direction
        n2 - number place cells in y direction
        total number of place cells will be n1 x n2
        '''
        n_place_cells = n1*n2
        n_neuron_in = 2000

        action_indices = list(range(len(env.actions)))
        self.env = env
        self.gamma = 0.95 # TODO given in paper? could not find it, discount factor

        # Create shared input node
        self.net = nengo.Network()
        self.net.input = nengo.Ensemble(
            n_neurons=n_neuron_in,
            dimensions=n_place_cells,
            radius=np.sqrt(n_place_cells)
        )

        # initialize neural net for actor
        self.Actor = ActorNet(
            n_pc=n_place_cells,
            input_node=self.net.input,
            n_neuron_out=200,
            lr=act_lr
        )
        # initialize neural net for critic
        self.Critic = CriticNet(
            n_pc=n_place_cells,
            input_node=self.net.input,
            n_neuron_out=100,
            lr=crit_lr
        )
        self.PlaceCells = PlaceCells(n1, n2, env.diameter, sigma)
        self.DecisionMaker = DecisionNode(action_indices)
        self.Error = ErrorNode(self.gamma)
