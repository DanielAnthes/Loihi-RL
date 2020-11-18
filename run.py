import nengo

from Environment import Maze
from Agent import Mouse




# set up simulation, connect networks
env = Maze()

with nengo.Network() as model:
    agent = Mouse(env, 23, 23, act_lr=0.01, crit_lr=0.01)

    # TODO add error node
    # environment node, step function expects integer so need to cast from float
    envstate = nengo.Node(lambda time, action: env.step(int(action)), size_in=1, size_out=5)
    
    # compute place cell activations
    nengo.Connection(envstate[:2], agent.PlaceCells.net.placecells)
    
    # place cells give input to actor and critic
    nengo.Connection(agent.PlaceCells.net.placecells, agent.Critic.net.input)
    nengo.Connection(agent.PlaceCells.net.placecells, agent.Actor.net.input)

    # take actor net as input to decision node
    nengo.Connection(agent.Actor.net.output, agent.DecisionMaker.net.choicenode)

    # execute action in environment
    nengo.Connection(agent.DecisionMaker.net.choicenode, envstate)

with nengo.Simulator(model) as sim:
    sim.run(10)