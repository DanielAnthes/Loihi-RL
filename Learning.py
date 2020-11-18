import warnings

from nengo.config import SupportDefaultsMixin
from nengo.exceptions import ValidationError
from nengo.params import (Default, IntParam, FrozenObject, NumberParam,
                          Parameter, Unconfigurable)
from nengo.synapses import Lowpass, SynapseParam

from nengo.learning_rules import LearningRuleTypeSizeInParam, LearningRuleType

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import BuildError
from nengo.node import Node

import numpy as np

class TDL(LearningRuleType):
    '''
    Temporal difference learning rule
    Class for nengo.learning_rules
    '''

    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1e-6
    )
    pre_synapse = SynapseParam(
        'pre_synapse', default=Lowpass(tau=0.005), readonly=True
    )
    post_synapse = SynapseParam(
        'post_synapse', default=None, readonly=True
    )
    beta = NumberParam('beta', low=0, readonly=True, default=1.0)

    def __init__(self, learning_rate=Default, pre_synapse=Default, post_synapse=Default, beta=Default):
        super(TDL, self).__init__(learning_rate, size_in=0)

        self.beta = beta
        self.pre_synapse = pre_synapse
        self.post_synapse = (self.pre_synapse if post_synapse is Default else post_synapse)

    @property
    def _argdefaults(self):
        return (('learning_rate', TDL.learning_rate.default),
                ('pre_synapse', TDL.pre_synapse.default),
                ('post_synapse', TDL.post_synapse.default),
                ('beta', TDL.beta.default))

class SimTDL(Operator):
    '''
    Temporal difference learning rule
    Class for nengo.builder.learning_rules
    '''

    def __init__(self, pre_filtered, post_filtered, weights, delta, learning_rate, beta, tag=None):
        super(SimTDL, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.beta = beta

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.reads[2]

    def _descstr(self):
        return 'pre=%s, post=%s -> %s' % (self.pre_filtered, self.post_filtered, self.delta)

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt
        beta = self.beta

        def step_simtdl():
            post_squared = alpha * post_filtered * post_filtered
            delta[...] = -beta * weights * post_squared[:,None]

            delta[...] += np.outer(alpha * post_filtered, pre_filtered)

        return step_simtdl

@Builder.register(TDL)
def build_tdl(model, tdl, rule):
    '''
    Build the .BCM object into our model
    '''

    conn = rule.connection

    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, tdl.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, tdl.post_synapse, post_activities)

    model.add_op(SimTDL(pre_filtered,
                        post_filtered,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        learning_rate=tdl.learning_rate,
                        beta=tdl.beta))

    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered
