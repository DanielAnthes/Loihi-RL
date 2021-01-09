import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.dists import DistOrArrayParam, Uniform

from Environment import Maze


def opt_angle(in_):
    x,y = in_[0:2]
    a = np.arctan2(y, x)
    
    a -= np.pi
    # this makes it jump between -pi and pi when crossing across x at x>.5,y=0
    if a < -np.pi:
        a += 2*np.pi
    
    return a
    
def mouse_angle(in_):
    x,y = in_[0:2]
    return np.arctan2(y, x)

env = Maze(max_time=120, speed=0.3)
with nengo.Network() as model:
    envstate = nengo.Node(lambda t, action: env.step(action), size_in=1, size_out=5)
    input = nengo.Ensemble(
            n_neurons=450,
            dimensions=1,
            radius=1.5*np.pi
            # neuron_type=nengo.LIF(tau_rc=0.001, tau_ref=0.001),
            # max_rates=Uniform(100,450)
    )
    actor = nengo.Ensemble(
                n_neurons=750, 
                dimensions=1, 
                radius= 1.5*np.pi,
                # neuron_type=nengo.LIF(tau_rc=0.01, tau_ref=0.001)
    )
    nengo.Connection(envstate[:2], input, synapse=0, function=mouse_angle)
    nengo.Connection(input, actor, synapse=0,
                            function = lambda x: 0,
                            learning_rule_type=nengo.Oja(),
                            solver=nengo.solvers.LstsqL2(weights=True))
    
    # opt = nengo.Node(lambda t, pos: opt_angle(pos), size_in=2, size_out=1)
    # nengo.Connection(input, opt, synapse=0)
    
    
    conn = nengo.Connection(actor, envstate, synapse=0)
    
    
    # err = nengo.Ensemble(n_neurons=150, dimensions=1)
    # nengo.Connection(opt, err, transform=-1)
    # nengo.Connection(actor, err)
    # nengo.Connection(err, conn.learning_rule)
    