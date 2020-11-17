import nengo
import numpy as np

from Networks import ActorNet, CriticNet


class Mouse:
    '''
    Actor class, has an environment in which actions are performed
    Learns to take optimal actions in the environment by applying Actor Critic Learning
    '''

    def __init__(self, env, n1, n2, sigma=0.16, act_lr=0.01, crit_lr=0.01):
        '''
        constructor
        env - the environment to act in
        n1 - number of place cells in x direction
        n2 - number place cells in y direction
        total number of place cells will be n1 x n2
        '''
        self.env = env
        env_diameter = env.diameter
        # compute locations of place cells, for convenience, place cells are laid out on a square overlapping the arena
        # this means that some place cells lay outside of the arena. However, this ensures that place cells are evenly spaced
        # alternatively, initialize place cells given polar coordinates and arrange on concentric circles
        # but I do not know how to make them evenly spaced that way
        self.n_place_cells = n1 * n2
        x_coords = np.linspace(-env_diameter/2,env_diameter/2, n1)
        y_coords = np.linspace(-env_diameter/2, env_diameter/2, n2)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.cell_locs = np.array([xx.flatten(), yy.flatten()])
        self.sigma = sigma # radius of circle giving 61% firing rate for place cells
        self.gamma = 0.95 # TODO given in paper? could not find it, discount factor

        self.Actor = ActorNet(n_pc=self.n_place_cells, n_neuron_in=2000, n_neuron_out=200) # initialize neural net for actor
        self.Critic = CriticNet(n_pcself.n_place_cells, n_neuron_in=2000, n_neuron_out=100) # initialize neural net for critic


    def _place_cell_activation(self, pos):
        '''
        compute activity of all place cells for a given position,
        firing rate for each place cell is given by a gaussian centered around it's preferred
        location with width sigma
        returns a vector of activation of all place cells
        '''
        pos = np.expand_dims(pos,1)
        enumerator = np.sqrt(np.sum( (pos - self.cell_locs)**2 ,axis=0))**2
        denominator = 2 * self.sigma**2
        activations = np.exp(-(enumerator / denominator))
        return activations

    def _compute_grad_critic(self, delta, state):
        '''
        TODO implement learning
        '''
        pass

    def _compute_grad_actor(self, delta, state, action):
        '''
        TODO implement learning
        '''

    def _compute_delta(self, rewards, values):
        '''
        computes difference between discounted future reward prediction and actual reward at time t
        d = R_t + gamma*C_t+1 - C_t
        additional constraint is that predicted future reward (C_t+1) at goal states is 0
        takes a list of rewards from 0:t and a list of evaluations given by the critic from 0:t+1
        returns a vector of delta values d in range 0:t
        '''
        R = np.array(rewards)
        C_t = np.array(values[:-1])
        C_next = np.array(values[1:])
        if R[-1] == 1:
            C_next[-1] = 0 # if last state is final, enforce prediction of future reward to be 0
        deltas = R + self.gamma*C_next - C_t
        return deltas


    def simulate(self):
        '''
        simulate one iteration of the experiment and return:
        - states, actions, rewards, values) as lists with entries for each timestep
        - a vector with time (for plotting /statistics)
        '''
        states = list()
        actions = list()
        rewards = list()
        values = list()
        path = list()

        time = [0]
        starting_pos = np.array([[0,1],[1,0],[0,-1],[-1,0]], dtype='float')


        env_state = self.env.reset(starting_pos[np.random.choice(4),:]) # random starting position north, south, east, or west
        path.append(env_state.tolist())
        #env_state = self.env.reset(starting_pos[1, :])  # fix starting position for testing
        done = False
        while not done: # keep taking steps until the environment signals done
            state = self._place_cell_activation(env_state)
            states.append(state)
            # flip a coin to simulate momentum
            coin = np.random.random()
            if coin > 0.25 and not len(actions) == 0: # repeat last action
                action = actions[-1]
            else: # generate new action
                probs = self.Actor(state)
                action = np.random.choice(8, p=probs)
            value = self.Critic(state)
            values.append(value[0])
            actions.append(action)
            env_state, reward, done, t = self.env.step(action) # take a step in the environment
            path.append(env_state.tolist())
            rewards.append(reward)
            time.append(t)
        # also add evaluation of last state: (because we may stop early and need to bootstrap with remaining expected future reward)
        state = self._place_cell_activation(env_state)
        states.append(state)
        value = self.Critic(state)
        values.append(value[0])
        return states, actions, rewards, values, time, path

    def eval_value_landscape(self, locs):
        n_locs = locs.shape[1]
        pc_activations = np.zeros((self.n_place_cells, n_locs))
        for i in range(n_locs):
            pc_activations[:,i] = self._place_cell_activation(locs[:,i])
        vals = self.Critic(pc_activations)
        return vals

    def train(self, iter, plot=False, plot_locs=None, plot_inter=10):
        '''
        simulate *iter* experiments and update the agent after each
        if plotting is enabled, a plot of the value function will be generated every plot_inter steps
        the value function will be evaluated at the locations specified in plot_locs
        '''
        ep_lengths = list()
        actor_grads = list()
        critic_grads = list()
        rewards = list()
        landscapes = list()
        paths = list()

        for i in range(iter):
            print(f"Iteration {i}", end="\r")
            if plot and i % plot_inter == 0:
                landscapes.append(self.eval_value_landscape(plot_locs))

            s,a,r,v,t,p = self.simulate_iteration()
            d = self._compute_delta(r, v)
            d_w_act = self._compute_grad_actor(d,s,a)
            d_w_crit = self._compute_grad_critic(d,s)

            self.Actor.update(d_w_act)
            self.Critic.update(d_w_crit)

            ep_lengths.append(t[-1])
            actor_grads.append(np.sum(abs(d_w_act)))
            critic_grads.append(np.sum(abs(d_w_crit)))
            rewards.append(np.sum(r))
            paths.append(p)

        return ep_lengths, actor_grads, critic_grads, rewards, landscapes, paths
