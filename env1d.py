import math
import numpy as np


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

class Marathon:
    def __init__(self, invert=False):
        self.invert = invert
        self.stepsize = 0.002  # with standard dt and track length 2 this
                                # leads to episode length of 10 seconds
        self.pathlen = 2
        self.agentpos = self.startpos()
        self.goal = 0
        self.reward = 1
        self.goalcounter = 0
        self.reset = 500
        self.stepcounter = 0
        self.maxsteps = 1500

        self.actionmem = []
        self.pmem = []

    def step(self, action):
        reset = 0
        x = action.astype(float)[0]
        # x = 1 -> p = 1, x = -1 -> p = 0
        p = 1 / (1 + np.exp(-4 * x))
        self.actionmem.append(x)
        self.pmem.append(p)
        probs = [p, 1-p]
        delta_pos = self.stepsize * np.random.choice([1,-1], p=probs)
        if np.abs(self.agentpos + delta_pos) <= 1:
            self.agentpos += delta_pos
        if self.goalReached():
            reward = self.reward
            if self.goalcounter == self.reset:
                self.agentpos = self.startpos()
                self.goalcounter = 0
            else:
                self.goalcounter += 1
                self.agentpos = self.goal

                if self.goalcounter == self.reset:
                    reset = 1
        else:
            reward = 0
            reset = 0
        
        self.agentpos = round_nearest(self.agentpos, self.stepsize)

        return np.array([self.agentpos, reward, reset])

    def goalReached(self):
        return np.abs(self.goal - self.agentpos) < self.stepsize

    def startpos(self):
        return np.random.choice([-0.5, 0.5])