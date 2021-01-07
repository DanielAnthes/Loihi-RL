import os
import json
import dill as pickle
import atexit
import numpy as np
import nengo
from nengo.solvers import NoSolver
from shutil import copyfile

import util
from Agent import Mouse
from Environment import Maze
from Networks import Switch
from extended_sim import ProbeMaxLength, ProbeToFile


class PseudoStruct:
    pass

class Manager:
    def __init__(self, path:str, cfg, load_from_file=False):
        sim = cfg["simulation"]
        net = cfg["network"]
        env = cfg["environment"]
        data = cfg.get("data", None)

        self._path = path
        self._directory = None

        # really really dirty way to copy all the dictionaries to properties
        # without explicitly checking if all necessary entries exist
        # and without checking if all names are legal
        # #TrustTheConfig
        self.simulation = PseudoStruct()
        self.simulation.__dict__.update(sim)
        self.network = PseudoStruct()
        self.network.__dict__.update(net)
        self.network.actor = PseudoStruct()
        self.network.critic = PseudoStruct()
        self.network.actor.__dict__.update(net["actor"])
        self.network.critic.__dict__.update(net["critic"])
        self.environment = PseudoStruct()
        self.environment.__dict__.update(env)
        if data is not None:
            self.data = PseudoStruct()
            self.data.__dict__.update(data)
        
        # This is where I would store all my trusty object,
        # IF I HAD ANY
        self._created_agent:Mouse = None
        self._created_environment:Maze = None
        self._created_network:nengo.Network = None
        self._probes = dict()
        self._simulator = None

        self.load_from_file = load_from_file

        # make sure simulator gets closed and resources released when closed
        @atexit.register
        def on_end():
            if self._simulator is not None:
                self._simulator.close()

    def load_saved_weights(self, load=True):
        self.load_from_file = load

    @property
    def Agent(self) -> Mouse:
        """Returns this mouse. If it doesn't exist yet, it is created"""
        # Agent must be created in network context, so if no network exist,
        # one is created by accessing the property, which also creates the agent
        if self._created_network is None:
            self.Network
        return self._created_agent

    @property
    def Environment(self) -> Maze:
        """Returns this environment. If it doesn't exist yet, it is created"""
        if self._created_environment is None:
            self._created_environment = self._create_environment()
        return self._created_environment
    
    @property
    def Network(self) -> nengo.Network:
        """Returns the nengo model. If it doesn't exist yet, it is created"""
        if self._created_network is None:
            self._created_network = self._create_network()
        return self._created_network

    @property
    def Probes(self) -> dict:
        """Returns all known probes. Is empty when network wasn't created"""
        return self._probes
    
    @property
    def Simulator(self) -> nengo.Simulator:
        """Returns simulator. Calls run() to create if not yet happened."""
        if self._simulator is None:
            #raise AttributeError("Simulator is only created after running once. Run the simulation first using run()")
            print("Run was not called yet. Running simulation according to config.")
            self.run()
        return self._simulator

    @property
    def Directory(self) -> str:
        """Return path of data directory. If directory does not exist, creates it."""
        if self._directory is None:
            _dir = os.path.join(os.getcwd(), "data", str(self))
            _dir = os.path.join(_dir, f'{self.network.actor.n_neurons}ACTOR-w-{self.network.actor.learning_rate}LR_{self.network.critic.n_neurons}CRITIC-w-{self.network.critic.learning_rate}LR')
            os.makedirs(_dir, exist_ok=True)
            self._directory = _dir
        return self._directory

    def create_network_with_seed(self, seed:int) -> nengo.Network:
        """Create the network with all seeds fixed to the given value"""
        self._created_network = self._create_network(seed)
        return self.Network

    def _create_environment(self) -> Maze:
        """Set up a Maze environment according to config"""
        return Maze(
            self.environment.speed,
            self.simulation.timestep,
            self.environment.diameter,
            self.environment.platform_diameter,
            self.environment.max_time
        )

    def _create_agent(self) -> Mouse:
        """Set up a Mouse agent according to config"""
        a = Mouse(
            self.Environment,
            self.network.n_placecells_x,
            self.network.n_placecells_y,
            self.network.discount_factor,
            self.network.actor.n_neurons,
            self.network.actor.learning_rate,
            self.network.critic.n_neurons,
            self.network.critic.learning_rate
        )
        self._created_agent = a
        return a

    def _create_network(self, seed:int=None):
        """Set up a Actor-Critic Network according to config"""
        if seed:
            np.random.seed(seed)
        with (nengo.Network(seed=seed) if seed else nengo.Network()) as model:
            if seed:
                model.config[nengo.Ensemble].seed = seed
                model.config[nengo.Node].seed = seed
                model.config[nengo.Connection].seed = seed
                model.config[nengo.Probe].seed = seed

            env = self.Environment
            agent = self._create_agent()

            envstate = nengo.Node(lambda time, action: env.step(action), size_in=1, size_out=5, label='Env-State')
            # can toggle learning on or off
            model.switch = Switch(state=1)
            # place cells give input to actor and critic
            nengo.Connection(envstate[:2], agent.net.input, label='Env2Agent')
            # take actor net as input to decision node
            nengo.Connection(agent.Actor.net.output, agent.DecisionMaker.net.choicenode, label='Actor2Decision')
            # execute action in environment
            nengo.Connection(agent.DecisionMaker.net.choicenode, envstate, synapse=0, label='Action2Env')

            # connect error node
            nengo.Connection(envstate[2], agent.Error.net.errornode[0], label='Reward2Error')
            nengo.Connection(agent.Critic.net.output, agent.Error.net.errornode[1], label='Critic2Error')
            nengo.Connection(model.switch.net.switch, agent.Error.net.errornode[2], label='Switch2Error')
            nengo.Connection(agent.Error.net.errornode[1], agent.Error.net.errornode[3], label='Err2Err') # recurrent connection to save last state; TODO: synapse=0 if transmission too bad
            nengo.Connection(agent.Error.net.errornode[0], agent.Actor.net.conn.learning_rule, label='Err2Learning')

            if self.load_from_file:
                try:
                    actor_weights, critic_weights = self._load_weights()
                except IOError as e:
                    print("File does not exist or is not readable, not loading weights.")
                except ValueError:
                    print("    The file contains an object array, which should not have happened. Not loading weights.")
                else:
                    agent.Actor.net.conn.solver = NoSolver(actor_weights, True)
                    # agent.Critic.net.conn.solver = NoSolver(critic_weights, False)

            # add Probes
            errorprobe = nengo.Probe(agent.Error.net.errornode[0])
            envprobe = nengo.Probe(envstate)
            switchprobe = nengo.Probe(model.switch.net.switch, sample_every=1.0)
            actorwprobe = ProbeMaxLength(agent.Actor.net.conn, 1, "weights", sample_every=.5)
            criticwprobe = ProbeMaxLength(agent.Critic.net.conn, 1, "weights", sample_every=.5)
            criticprobe = nengo.Probe(agent.Critic.net.output)
        
        # self all of the probes for future reference
        self._probes = {
            "errorprobe": errorprobe,
            "envprobe": envprobe,
            "switchprobe": switchprobe,
            "criticprobe": criticprobe,
            "actorweightprobe": actorwprobe,
            "criticweightprobe": criticwprobe
        }

        return model

    def run(self, backend:str):
        """Run the simulation on the given backend"""
        if self._simulator is None or self._simulator.closed:
            try:
                sim = util.create_simulator(
                        backend, 
                        model=self.Network,
                        timestep=self.simulation.timestep
                    )
            except Exception as e:
                if type(e) == KeyboardInterrupt:
                    raise e
                print(e)
                print("WARNING: Falling back to CPU backend")
                sim = util.create_simulator(
                        'CPU', 
                        model=self.Network,
                        timestep=self.simulation.timestep
                    )
            
            self._simulator = sim

        self._simulator.run(self.simulation.steps)

    def save(self):
        """Save data and values to the directory"""
        _dir = self.Directory
        # save the actual config file
        copyfile(self._path, os.path.join(_dir, os.path.split(self._path)[-1]))
        # save the time points
        np.save(os.path.join(_dir, "timepoints"), self.Simulator.trange())
        
        if self.data is not None:
            act_w = self.data.actor_weights
            crit_w = self.data.critic_weights
        else:
            act_w = "actorweight"
            crit_w = "criticweight"

        # save all the probe data with special treatment for the weights
        for pname, p in self.Probes.items():
            if pname.endswith("weightprobe"):
                # only save latest weights
                d = self.Simulator.data[p][-1]
                if pname.startswith("actor"):
                    pname = act_w
                elif pname.startswith("critic"):
                    pname = crit_w
            else:
                d = self.Simulator.data[p]
            
            np.save(os.path.join(_dir, pname), d)

        # save the actionmemory
        np.save(os.path.join(_dir, "env_actionmemory"), self.Environment.actionmemory)

        with open(os.path.join(_dir, "simulator"), "wb") as f:
            pickle.dump(self.Simulator, f)

    def _load_weights(self):
        """Load weights from the directory. Does not check if exists"""
        if self.data is not None:
            act_w = self.data.actor_weights
            crit_w = self.data.critic_weights
        else:
            act_w = "actorweight"
            crit_w = "criticweight"

        return ( np.load(os.path.join(self.Directory, act_w)), 
                 np.load(os.path.join(self.Directory, crit_w)) )

    def __str__(self):
        return f'{self.simulation.steps}STEPS_{self.simulation.timestep}DT_{self.environment.max_time}EPISODE'


def load(path:str) -> Manager:
    """Supplied with a config file, it will be loaded into a Manager"""
    with open(path, "r") as f:
        cfg = json.load(f)
    obj = Manager(path, cfg)

    return obj
