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
try:
    import pyopencl as cl
    import nengo_ocl
    from nengo_ocl.plan import Plan
    from nengo_ocl.utils import as_ascii, indent, nonelist, round_up
    GPU = True
except ModuleNotFoundError:
    GPU = False
from mako.template import Template

import numpy as np

class TDL(LearningRuleType):
    '''
    Temporal difference learning constructor.
    Initialises the learning rule for nengo.learning_rules such that Learning.TDL()
    becomes a recognised callable and sets its a default configuration.
    '''

    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')

    learning_rate = NumberParam('learning_rate', low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam('pre_synapse', default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam('post_synapse', default=None, readonly=True)

    def __init__(self, learning_rate=Default, pre_synapse=Default, post_synapse=Default):
        '''
        INPUTS:
            learning_rate   -   Learning rate for connection, 1e-6 by default
            pre_synapse     -   PreSynaptic neuron population
            post_synapse    -   PostSynaptic neuron population
        NOTE:
            pre_synapse and post_synapse must be of type Ensemble if LstsqL2
            is employed as specified solver for weights
        '''

        super(TDL, self).__init__(learning_rate, size_in=1)

        self.pre_synapse = pre_synapse
        self.post_synapse = (self.pre_synapse if post_synapse is Default else post_synapse)  # if no post_synapse is supplied, recurrent connection is assumed

    @property
    def _argdefaults(self):
        return (('learning_rate', TDL.learning_rate.default),
                ('pre_synapse', TDL.pre_synapse.default),
                ('post_synapse', TDL.post_synapse.default))

class SimTDL(Operator):
    '''
    Simulator object for temporal difference learning on CPU.
    This is the class used after simulation steps to compute weight changes that
    are implemented as:

        d_ij = α𝛿(PreSynapticActivity * PostSynapticActivity)

        where:

            α = LearningRate * dt
            𝛿 = PredictionError

    Delta has to be a single float value that is fed into the connection (e.g.,
    nengo.Connection(my_error_ensemble, my_tdl.conn.learning_rule).
    '''

    def __init__(self, pre_filtered, post_filtered, weights, error, delta, learning_rate, encoders=None, tag=None):
        '''
        INPUTS:
            pre_filtered    -   Filtered activity from presynaptic neuron(s)
            post_filtered   -   Filtered activity from postsynaptic neuron(s)
            weights         -   Weight matrix
            error           -   Prediction error 𝛿 (single-valued float)
            delta           -   Weight updates
            learning_rate   -   Learning rate
            encoders        -   Encoders
            tag             -   Connection tag
        '''

        super(SimTDL, self).__init__(tag=tag)

        self.learning_rate = learning_rate
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights, error] + ([] if encoders is None else [encoders])
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

    @property
    def error(self):
        return self.reads[3]

    @property
    def encoders(self):
        return None if len(self.reads) < 5 else self.reads[4]

    def _descstr(self):
        '''
        Returns nengo description string.
        '''
        return 'pre=%s, post=%s -> %s' % (self.pre_filtered, self.post_filtered, self.delta)

    def make_step(self, signals, dt, rng):
        '''
        Returns the actual function that computes the weight changes
        to nengo.
        '''

        weights = signals[self.weights]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        error = signals[self.error]
        delta = signals[self.delta]

        def step_simtdl():
            delta[...] = self.learning_rate * dt * error * np.outer(post_filtered, pre_filtered)  # adjust formula here
        return step_simtdl

@Builder.register(TDL)
def build_tdl(model, tdl, rule):
    '''
    Registers a .BCM object in model for nengo builder such that
    simulator knows what class to use for weight computations.

    INPUTS:
        model   -   Model wherein operation is added
        tdl     -   TDL() connection object for which rule is implemented
        rule    -   Nengo identification string for outputs to be read in SimTDL
    '''

    conn = rule.connection

    error = Signal(shape=rule.size_in, name="TDL:error")
    model.add_op(Reset(error))

    model.sig[rule]["in"] = error
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    encoders = model.sig[get_post_ens(conn)]["encoders"][:,conn.post_slice]
    pre_filtered = build_or_passthrough(model, tdl.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, tdl.post_synapse, post_activities)

    model.add_op(SimTDL(pre_filtered,
                        post_filtered,
                        model.sig[conn]['weights'],
                        error,
                        model.sig[rule]['delta'],
                        learning_rate=tdl.learning_rate,
                        encoders=encoders))

    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered
    model.sig[rule]['error'] = error

def plan_tdl(queue, pre, post, weights, errors, delta, alpha, encoders=None, tag=None):
    '''
    Full planning function for temporal difference learning on GPU.
    C implementation of SimTDL. Please refer to SimTDL for formulation of
    the learning rule applied.

    INPUTS:
        pre         -   Filtered activity from presynaptic neuron(s)
        post        -   Filtered activity from postsynaptic neuron(s)
        weights     -   Weight matrix
        errors      -   Prediction error 𝛿 (single-valued float)
        delta       -   Weight updates
        alpha       -   Effective learning rate (α = LearningRate * dt)
        encoders    -   Encoders
        tag         -   Connection tag
    OUTPUTS:
        plan        -   Plan for GPU
    '''

    assert (
        len(pre) == len(post) == len(weights) == len(delta) == alpha.size
    )
    N = len(pre)

    for arr in (pre, post):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (delta, weights):  # matrices
        assert (arr.stride1s == 1).all()

    assert (post.shape0s == weights.shape0s).all()
    assert (pre.shape0s == weights.shape1s).all()
    assert (weights.shape0s == delta.shape0s).all()
    assert (weights.shape1s == delta.shape1s).all()

    assert (
        pre.ctype
        == post.ctype
        == weights.ctype
        == delta.ctype
        == alpha.ctype
    )

    '''
    text = function to be executed on GPU. This is the equivalent
    of SimTDL::make_step::step_simtdl() that computes weight the
    weight changes. In the case of the GPU, weight changes are
    computed in strides between pre- and postsynaptic activity.
    '''

    text = """
    __kernel void tdl(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *pre_stride0s,
        __global const int *pre_starts,
        __global const ${type} *pre_data,
        __global const int *post_stride0s,
        __global const int *post_starts,
        __global const ${type} *post_data,
        __global const int *weights_stride0s,
        __global const int *weights_starts,
        __global const ${type} *weights_data,
        __global const ${type} *errors,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const ${type} *alphas
    )
    {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);
        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;
        __global ${type} *delta = delta_data + delta_starts[k];
        const ${type} pre = pre_data[pre_starts[k] + j*pre_stride0s[k]];
        const ${type} post = post_data[post_starts[k] + i*post_stride0s[k]];
        const ${type} weight = weights_data[
            weights_starts[k] + i*weights_stride0s[k] + j];
        const ${type} error = errors[sizeof(errors) / sizeof(errors[0])];
        const ${type} alpha = alphas[k];

        if (i < shape0) {
            delta[i*delta_stride0s[k] + j] =
                alpha * error * (pre * post);
        }
    }
    """

    textconf = dict(type=pre.ctype)
    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))

    full_args = (
        delta.cl_shape0s,
        delta.cl_shape1s,
        pre.cl_stride0s,
        pre.cl_starts,
        pre.cl_buf,
        post.cl_stride0s,
        post.cl_starts,
        post.cl_buf,
        weights.cl_stride0s,
        weights.cl_starts,
        weights.cl_buf,
        errors.cl_buf,
        delta.cl_stride0s,
        delta.cl_starts,
        delta.cl_buf,
        alpha,
    )
    if GPU:
        _fn = cl.Program(queue.context, text).build().tdl
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_tdl", tag=tag)
    plan.full_args = full_args  # prevent garbage-collection
    plan.flops_per_call = 6 * delta.sizes.sum()
    plan.bw_per_call = (
        pre.nbytes
        + post.nbytes
        + weights.nbytes
        + errors.nbytes
        + delta.nbytes
        + alpha.nbytes
    )
    return plan

def plan_SimTDL(self, ops):
    '''
    SimTDL planner for GPU. Effectively, this retrieves all
    the relevant data out of the simulator at the current
    timestep and passes it on to our TDL planner.

    INPUTS:
        ops     -   Nengo operations (supplied by nengo)
    OUTPUTS:
        plan    -   Plan for GPU
    '''

    pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
    post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
    weights = self.all_data[[self.sidx[op.weights] for op in ops]]
    error = self.all_data[[self.sidx[op.error] for op in ops]]
    delta = self.all_data[[self.sidx[op.delta] for op in ops]]
    alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
    return [plan_tdl(self.queue, pre, post, weights, error, delta, alpha)]

if GPU:
    setattr(nengo_ocl.Simulator, 'plan_SimTDL', plan_SimTDL)  # Register GPU callable for our learning rule with nengo_ocl simulator if libraries are available
