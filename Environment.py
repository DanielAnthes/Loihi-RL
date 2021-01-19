import numpy as np
import nengo


class Maze:
    '''
    Environment Class, interactions are defined by reset and step
    '''
    def __init__(self):
        # coordinates given in meters, (0,0) marks center of the maze
        self.speed = 0.3  # m/s speed of the mouse
        self.timestep = 0.01  # seconds, discrete time steps
        self.diameter = 2  # meters, diameter of the maze TODO: if timesteps are small agent gets to make many choices -> wiggling ensues
        self.platformsize = 0.1 # meters, diameter of platform
        self.platform_loc = np.array([0,0], dtype='float')  # location of platform x,y coordinates in meters
        self.mousepos = self._get_random_start()
        self.done = False  # whether mouse has reached platform
        self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]], dtype='float') # mapping from action (index) to direction vector
        self.starting_pos = np.array([[0,1],[1,0],[0,-1],[-1,0]], dtype='float') # possible starting locations TODO: this should probably be relted to diameter to avoid invalid starting pos
        self.max_time = 30  # maximum trial duration in seconds # originally 120 seconds
        self.time = 0
        self.actionmemory = list()

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
        a time stamp encoded in an array

            INPUTS:
                an action represented as integer in [0,7]

            OUTPUTS:
                a numpy array with [xpos, ypos, reward, done, time]
                with
                    xpos    - float
                    ypos    - float
                    reward  - 1 if target reached, 0 otherwise
                    done    - 1 if target or time limit reached 0 otherwise
                    time    - float
                    reset   - 1 if environment was reset in current time step
        '''
        action = action.astype(int)[0]

        self.time += self.timestep
        self.actionmemory.append(action)
        if self.done:  # check whether simulation has ended
            pn = self._get_random_start()
            self.reset(pn)  # random starting position
            returnarr = np.array([self.mousepos[0], self.mousepos[1], 0, 0, self.time, 1])
            return returnarr

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

        doneval = 1 if self.done else 0

        # pack return in an array
        returnarr = np.array([self.mousepos[0], self.mousepos[1], reward, doneval, self.time, 0])
        return returnarr

    def reset(self, mousepos=np.array([0,1], dtype='float')):
        '''
        reset the environment, returns initial position
        '''
        self.time = 0
        self.done = False
        self.mousepos = mousepos
        return self.mousepos

    def _get_random_start(self):
        '''
        random restart position close to the edge of the arena
        '''
        radius=0.95 * self.diameter / 2
        angle = np.random.random()*2*np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x,y])


class TestEnv:

    def __init__(self):
        self.stepsize = 0.002  # with standard dt and track length 2 this
                                # leads to episode length of 10 seconds
        self.pathlen = 2
        self.agentpos = 0
        self.reward = 1
        self.goalcounter = 0
        self.reset = 500

    def step(self):
        reset = 0
        self.agentpos += self.stepsize
        if self.goalReached():
            reward = self.reward
            if self.goalcounter == self.reset:
                self.agentpos = 0
                self.goalcounter = 0
            else:
                self.goalcounter += 1
                self.agentpos = self.pathlen

                if self.goalcounter == self.reset:
                    reset = 1
        else:
            reward = 0
            reset = 0
        return np.array([self.agentpos, reward, reset])

    def goalReached(self):
        return abs(self.agentpos - self.pathlen) < self.stepsize

class TestEnvActor:

    def __init__(self, dt, trial_length):
        self.stepsize = .002
        self.reward = 1
        self.goalcounter = 0
        self.reset = 500
        self.pos = self._get_random_start()
        self.goal = 0
        self.goald = .1
        self.diameter = 2
        self.dt = dt
        self.trial_length = trial_length
        self.t = .0

    def _get_random_start(self):
        return np.random.choice(np.array([-.5, .5]), 1)[0]

    def _goalReached(self):
        return abs(self.pos) < abs(self.goal - self.goald / 2)

    def step(self, action):
        self.t += self.dt

        action = action[0]
        prob_left = 1 / (1 + np.exp(-action))
        prob_right = 1 - prob_left
        direction = np.random.choice(np.array([-1, 1]), 1, p=np.array([prob_left, prob_right]))[0]
        dt_mov = direction * self.stepsize

        if abs(self.pos + dt_mov) < (self.diameter / 2):
            self.pos += dt_mov
        else:
            print("not moving because new_pos = " + str(self.pos + dt_mov))

        reset = 0

        if self._goalReached():
            reward = self.reward
            if self.goalcounter == self.reset:
                self.pos = self._get_random_start()
                self.t = 0
                self.goalcounter = 0
            else:
                self.goalcounter += 1
                self.pos = self.goal

                if self.goalcounter == self.reset:
                    reset = 1
        else:
            reward = 0

        if reward is not self.reward and self.t > self.trial_length:
            self.t = 0
            reset = 0
            self.pos = self._get_random_start()

        return [self.pos, reward, reset]
