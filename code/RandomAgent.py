import nengo
from Environment import Maze
from random import choice

def DebugRandomChoice(actions):
    print(f"possible actions: {actions}")
    action = choice(actions)
    print(f"action: {action}")
    return action



actions = [0,1,2,3,4,5,6,7]

print(f"POSSIBLE ACTIONS: {actions}")

env = Maze()





with nengo.Network() as model:
    envstate = nengo.Node(lambda time, action: env.step(int(action[0])), size_in=1, size_out=5)
    actionnode = nengo.Node(lambda t: DebugRandomChoice(actions))
    nengo.Connection(actionnode, envstate)

with nengo.Simulator(model) as sim:
    sim.run(1)