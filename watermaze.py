import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import gif
# from pygifsicle import optimize as optim_gif
from matplotlib import cm

np.seterr(over='raise') # raise errors for overflow warnings

class Maze:
    '''
    Environment Class, interactions are defined by reset and step
    '''
    def __init__(self):
        # coordinates given in meters, (0,0) marks center of the maze
        self.speed = 0.3 # m/s speed of the mouse
        self.timestep = 0.1 # seconds, discrete time steps
        self.diameter = 2 # meters, diameter of the maze
        self.platformsize = 0.1 # meters, diameter of platform
        # self.platform_loc = np.array([0.6,-0.6], dtype='float') # location of platform x,y coordinates in meters
        self.platform_loc = np.array([0,0], dtype='float') # location of platform x,y coordinates in meters
        # self.platform_loc = np.array([0.2,-0.3], dtype='float') # location of platform x,y coordinates in meters
        self.mousepos = np.array([0,1], dtype='float') # starting position of the mouse, north by default
        self.done = False # whether mouse has reached platform
        self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]], dtype='float') # mapping from action (index) to direction vector
        self.max_time = 120 # maximum trial duration in seconds
        self.time = 0

    def _outOfBounds(self, loc):
        '''
        takes a location and returns true if the position lies outside of the maze
        '''
        distanceFromCenter = np.sqrt(np.sum(loc**2)) #euclidean distance
        return distanceFromCenter > (self.diameter / 2)

    def _goalReached(self):
        goalDist = abs(np.sqrt(np.sum((self.platform_loc - self.mousepos)**2)))
        return goalDist < self.platformsize / 2

    def step(self, action):
        '''
        takes an action and returns a new position, a reward, a done flag and
        a time stamp
        '''
        self.time += self.timestep
        if self.done: # check whether simulation has ended
            print("mouse has reached the platform, undefined behaviour")
            return (None,None), -1, True
        else:
            direction = self.actions[action]
            len_step = self.speed * self.timestep
            delta_pos = (direction / np.sqrt(np.sum(direction**2))) * len_step
            if not self._outOfBounds(self.mousepos + delta_pos): # if mouse would go out of bounds bounce back
                self.mousepos += delta_pos

            if self._outOfBounds(self.mousepos):
                print("out of bounds!")


            self.done = self._goalReached()
            reward = 1 if self.done else 0

            if not self.done and self.time > self.max_time: # if timelimit is exceeded and goal is not reached, stop without reward
                self.done = True

            return self.mousepos, reward, self.done, self.time

    def reset(self, mousepos=np.array([0,1], dtype='float')):
        '''
        reset the environment, returns initial position
        '''
        self.time = 0
        self.done = False
        self.mousepos = mousepos
        return self.mousepos


class CriticNet:
    '''
    A network that learns a function mapping input from N placecells to a single critic cell
    function: PC^N -> V
    '''
    def __init__(self, n, lr=.01):
        # self.w = np.random.random((1,n))*2-1 # weights from n inputs to a single output
        self.w = np.zeros((1,n))
        self.lr = lr

    def __call__(self, input):
        '''
        forward pass, takes an activation vector (of placecells) and returns a value according to weights
        shape must be n x m with m states (activation vectors) each of length n
        returns either a single value (for one input vector) or a vector of m values
        '''
        out = np.dot(self.w, input)
        if out.shape == (1,1):
            return out[0,0]
        else:
             return out

    def update(self, delta_w):
        '''
        updates the network with supplied weight update
        delta_w is an array of shape 1 x n
        '''
        self.w += delta_w * self.lr


class ActorNet:
    '''
    A network that learns to map input vectors from place cells to appropriate actions
    function: PC^N -> A^8
    output are the logit action probabilities for eight possible actions
    '''
    def __init__(self, n, lr=0.01):
        '''
        n is the number of place cells
        '''
        # self.w = np.random.random((8,n))*2-1
        self.w = np.zeros((8,n))
        self.lr = lr

    def __call__(self, input):
        '''
        forward pass: takes an activation vector (of placecells) and returns a value according to weights
        shape must be n x m with m states (activation vectors) each of length n
        returns a vector of probabilities to move in each of 8 possible directions
        '''
        logit_out = np.dot(self.w, input)
        # apply scaled softmax

        out = np.exp(2*logit_out)
        out /= np.sum(out) # normalize probabilities

        return out

    def update(self, delta_w):
        '''
        updates the network with supplied weight update
        delta_w is an array of shape 8 x n
        '''
        self.w += delta_w * self.lr


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

        self.Actor = ActorNet(self.n_place_cells, lr=act_lr) # initialize neural net for actor
        self.Critic = CriticNet(self.n_place_cells, lr=crit_lr) # initialize neural net for critic


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
        computes the weight gradients (unscaled) for each weight in the critic net given
        a vector of prediction errors delta and a list of states
        returns weight gradient vector
        '''
        delta_w = np.zeros(self.Critic.w.shape)
        for d,s in zip(delta, state):
            delta_w += d*s
        return delta_w

    def _compute_grad_actor(self, delta, state, action):
        '''
        computes the weight gradients (unscaled) for each weight in the actor net given
        a vector of prediction errors delta a list of states and a list of actions
        returns weight gradient matrix
        '''
        delta_w = np.zeros(self.Actor.w.shape)
        for d,s,a in zip(delta,state,action):
            delta_w[a,:] += d * s # only update weights responsible for the action that was taken
        return delta_w

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


    def simulate_iteration(self):
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
            print(f"Iteration {i}")
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
#%% setup

env = Maze()
actor = Mouse(env, 23, 23, act_lr=0.01, crit_lr=0.01) # use slightly more place cells than experiment since rectangular layout will cause some of them to lie out of bounds
n_iter = 300

# define a grid at which to evaluate value function for plotting
x = np.linspace(-1,1,50)
y = np.linspace(-1,1,50)
xx,yy = np.meshgrid(x,y)
idx = np.sqrt(xx**2 + yy**2) < env.diameter / 2 # indices of all locations that fall inside the maze
xx = xx[idx]
yy = yy[idx]
locs = np.array([xx.flatten(), yy.flatten()])


#%% train

ep_lengths, actor_grads, critic_grads, rewards, landscapes, paths = actor.train(n_iter, plot=True, plot_locs=locs, plot_inter=1)

# plot some statistics

plt.figure()

plt.subplot(311)
plt.plot(range(n_iter),ep_lengths)
plt.xlabel("Iteration")
plt.ylabel("end time in seconds")

plt.subplot(312)
plt.plot(range(n_iter), actor_grads)
plt.plot(range(n_iter), critic_grads, '-.')
plt.xlabel("Iteration")
plt.ylabel("size of weight gradient")
plt.legend(["actor", "critic"])

plt.subplot(313)
plt.scatter(range(n_iter), rewards)
plt.xlabel("Iteration")
plt.ylabel("reward")

plt.show()

# plot value function

fig = plt.figure()
ax = fig.gca(projection='3d')
landscape = landscapes[-1].flatten()
ax.plot_trisurf(xx,yy,landscape, cmap=cm.summer)
plt.title("Final value surface")
plt.show()

#%% plot some paths
def plot_traj(traj, env):
    ax = plt.subplot(111)
    ax.plot(traj[:,0], traj[:,1], '.-')
    arena = plt.Circle((0,0), 1, color='k', fill=False)
    platform = plt.Circle(env.platform_loc, env.platformsize/2, fill=True, color='k')
    ax.add_artist(platform)
    ax.add_artist(arena)

rewards = np.array(rewards)
wins = np.where(rewards == 1)[0]

first_win = wins[0] # first win
first_traj = paths[first_win] # first winning trajectory
first_traj = np.array(first_traj)

plt.figure()
plot_traj(first_traj, env)
ax = plt.gca()
ax.axis('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.title("trajectory")
plt.show()

last_win = wins[-1]
last_traj = paths[last_win] # last winning trajectory
last_traj = np.array(last_traj)

plt.figure()
plot_traj(last_traj, env)
ax = plt.gca()
ax.axis('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.title("trajectory")
plt.show()

'''
#%% create gif of value function iteration
@gif.frame
def valuefunction_frame(landscape, xx, yy):
    fig = plt.figure()
    ax = plt.subplot(111,projection='3d')
    ax.plot_trisurf(xx, yy, landscape, cmap=cm.summer)

frames = list()
for i in range(len(landscapes)):
    frames.append(valuefunction_frame(landscapes[i].flatten(), xx, yy))
gif.save(frames, "valuefunction.gif", duration=100)

#%%

optim_gif("valuefunction.gif")


#%%

fig = plt.figure(figsize=(30,20))
for i in range(30):
    ax = plt.subplot(5,6,i+1, projection='3d')
    landscape = landscapes[i*40]
    ax.plot_surface(xx, yy, np.reshape(landscape, xx.shape))
plt.show()

fig.savefig('valuesurface.png')
'''
