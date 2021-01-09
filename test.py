import numpy as np
import matplotlib.pyplot as plt
import nengo

from Environment import Maze


def opt_angle(in_):
    x,y = in_[0:2]
    # avoid 0 division
    x += 1e-8
    y += 1e-8
    a = np.arctan(y/x)
    # in quadrant III&II add pi
    if (x < 0 and y < 0) or (x < 0 and y > 0):
        a += np.pi
    # in quadrant IV add 2pi
    elif x > 0 and y < 0:
        a += 2 * np.pi
    
    return a * 180 / np.pi - 180

env = Maze(max_time=30, speed=0.3)
with nengo.Network() as model:
    envstate = nengo.Node(lambda t, action: env.step(action), size_in=1, size_out=5)
    input = nengo.Ensemble(
            n_neurons=250,
            dimensions=2,
            radius=env.diameter / 1.8
    )
    actor = nengo.Ensemble(
                n_neurons=500, 
                dimensions=1, 
                radius= 1.2*np.pi
    )
    nengo.Connection(envstate[:2], input, synapse=0)
    conn = nengo.Connection(input, actor,
                            function = opt_angle, synapse=0 )
    nengo.Connection(actor, envstate, synapse=0)
    
    opt = nengo.Node(lambda t, pos: opt_angle(pos), size_in=2, size_out=1)
    nengo.Connection(input, opt)

