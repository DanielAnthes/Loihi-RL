import numpy as np
import nengo

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
        self.starting_pos = np.array([[0,1],[1,0],[0,-1],[-1,0]], dtype='float') # possible starting locations
        
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
        # print(f"RECEIVED ACTION: {action}")
        self.time += self.timestep
        if self.done: # check whether simulation has ended
            # print("mouse has reached the platform, resetting")
            self.reset(self.starting_pos[np.random.choice(4),:]) # random starting position north, south, east, or west
            returnarr = np.array([self.mousepos[0], self.mousepos[1], 0, 0, self.time])
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
        returnarr = np.array([self.mousepos[0], self.mousepos[1], reward, doneval, self.time])
        return returnarr

    def reset(self, mousepos=np.array([0,1], dtype='float')):
        '''
        reset the environment, returns initial position
        '''
        self.time = 0
        self.done = False
        self.mousepos = mousepos
        return self.mousepos
