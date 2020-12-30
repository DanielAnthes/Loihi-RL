import nengo
import numpy as np

from Networks import ActorNet, PlaceCells, DecisionNode, ErrorNode, DeterministicCritic


class Mouse:
    '''
    Actor class, has an environment in which actions are performed
    Learns to take optimal actions in the environment 
    by applying Actor Critic Learning
    '''

    def __init__(
            self, 
            env,
            n1, n2,
            discount_factor=0.9995,
            actor_neurons=200,
            act_lr=6e-8,
            critic_neurons=100,
            crit_lr=1e-4):
        '''
        TODO: agent needs error node for learning
        env - the environment to act in
        n1 - number of place cells in x direction
        n2 - number place cells in y direction
        total number of place cells will be n1 x n2
        '''
        n_place_cells = n1*n2

        action_indices = list(range(len(env.actions)))
        self.env = env
        self.gamma = discount_factor  # discount factor

        # Create shared input node
        self.net = nengo.Network()
        self.net.input = nengo.Ensemble(
            n_neurons=n_place_cells,
            dimensions=2,
            radius=2
        )

        # initialize neural net for actor
        self.Actor = ActorNet(
            n_pc=n_place_cells,
            input_node=self.net.input,
            n_neuron_out=actor_neurons,
            lr=act_lr
        )
        # initialize neural net for critic
        self.Critic = DeterministicCritic(
            n_pc=n_place_cells,
            input_node=self.net.input,
            n_neuron_out=critic_neurons,
            lr=crit_lr
        )
        self.DecisionMaker = DecisionNode(action_indices)
        self.Error = ErrorNode(self.gamma)
