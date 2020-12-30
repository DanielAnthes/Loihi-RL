import nengo
import nengo_loihi
from Networks import ActorNet, DeterministicCritic, DecisionNode, ErrorNode

n_place_cells = 100  # arbitrary
act_lr = 1e-5  # arbitrary
crit_lr = 1e-5  # arbitrary
gamma = 0.9995  # also arbitrary
action_indices = list(range(4))


with nengo.Network() as model:
    # Create shared input node
    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        input = nengo.Ensemble(
        n_neurons=n_place_cells,
        dimensions=2,
        radius=2
        )
        net.config[input].on_chip = True

    # initialize neural net for actor
    Actor = ActorNet(
        n_pc=n_place_cells,
        input_node=input,
        n_neuron_out=200,
        lr=act_lr
    )

    # initialize neural net for critic
    Critic = DeterministicCritic(
        n_pc=n_place_cells,
        input_node=input,
        n_neuron_out=100,
        lr=crit_lr
    )

    DecisionMaker = DecisionNode(action_indices)
    Error = ErrorNode(gamma)

with nengo_loihi.Simulator(net) as sim:
    sim.run(30)

