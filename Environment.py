import numpy as np
import nengo


class Maze:
    '''
    Environment Class, interactions are defined by reset and step
    '''
    def __init__(self):
        # coordinates given in meters, (0,0) marks center of the maze
        self.speed = 0.3  # m/s speed of the mouse
        self.timestep = 0.001  # seconds, discrete time steps
        self.diameter = 2  # meters, diameter of the maze TODO: if timesteps are small agent gets to make many choices -> wiggling ensues
        self.platformsize = 0.1 # meters, diameter of platform
        self.platform_loc = np.array([0,0], dtype='float')  # location of platform x,y coordinates in meters
        self.mousepos, self.mouserot = self._get_random_start()
        self.maxrot = np.pi/2
        self.done = False  # whether mouse has reached platform
        # self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]], dtype='float') # mapping from action (index) to direction vector
        self.max_time = 120  # maximum trial duration in seconds # originally 120 seconds
        self.time = 0
        self.actionmemory = list()
        self.deltamemory = list()

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
        '''
        # action = action.astype(int)[0]
        action = action[0]

        self.time += self.timestep
        self.actionmemory.append(action)
        if self.done:  # check whether simulation has ended
            self.reset()  # random starting position
            returnarr = np.array([self.mousepos[0], self.mousepos[1], self.mouserot, 0, 0, self.time])
            return returnarr

        # direction = self.actions[action]
        len_step = self.speed * self.timestep
        rot = self.mouserot + action * self.maxrot
        delta_pos = len_step * np.array([np.cos(rot), np.sin(rot)])
        self.deltamemory.append(delta_pos)
        if not self._outOfBounds(self.mousepos + delta_pos): # if mouse would go out of bounds bounce back
            self.mousepos += delta_pos
            self.mouserot = rot

        if self._outOfBounds(self.mousepos):
            print("out of bounds!")

        self.done = self._goalReached()
        reward = 1 if self.done else 0

        if not self.done and self.time > self.max_time: # if timelimit is exceeded and goal is not reached, stop without reward
            self.done = True

        doneval = 1 if self.done else 0

        # pack return in an array
        returnarr = np.array([self.mousepos[0], self.mousepos[1], self.mouserot, reward, doneval, self.time])
        return returnarr

    def reset(self, start_cond=None, dtype='float'):
        '''
        reset the environment, returns initial position
        '''
        if start_cond is None:
            self.mousepos, self.mouserot = self._get_random_start()
        else:
            self.mousepos, self.mouserot = start_cond

        self.time = 0
        self.done = False
        return self.mousepos, self.mouserot

    def _get_random_start(self):
        '''
        random restart position close to the edge of the arena
        '''
        radius=0.95 * self.diameter / 2
        angle = np.random.random()*2*np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # random facing direction
        rot = (np.random.rand() - 0.5) * np.pi
        # always facing the platform
        r = np.sqrt(x**2 + y**2)
        a = np.cos(x / r)
        rot = a - np.pi
        return np.array([x,y]), rot
