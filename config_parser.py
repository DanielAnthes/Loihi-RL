import os
import json
import pickle
import numpy as np
import nengo
from shutil import copyfile

import util
from Agent import Mouse
from Environment import Maze
from Networks import Switch


class PseudoStruct:
    pass

class Manager:
    def __init__(self, path:str, sim:dict, net:dict, env:dict):
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
        
        # This is where I would store all my trusty object,
        # IF I HAD ANY
        self._created_agent:Mouse = None
        self._created_environment:Maze = None
        self._created_network:nengo.Network = None
        self._probes = dict()
        self._simulator = None

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
        _dir = os.path.join(os.getcwd(), "data", str(self))
        if os.path.isdir(_dir):
            _dir = os.path.join(_dir, f'{self.network.actor.n_neurons}ACTOR-w-{self.network.actor.learning_rate}LR_{self.network.critic.n_neurons}CRITIC-w-{self.network.critic.learning_rate}LR')
        os.makedirs(_dir, exist_ok=True)
        self._directory = _dir
        return self._directory

    def create_network_with_seed(self, seed:int) -> nengo.Network:
        self._created_network = self._create_network(seed)
        return self.Network

    def _create_environment(self) -> Maze:
        return Maze(
            self.environment.speed,
            self.simulation.timestep,
            self.environment.diameter,
            self.environment.platform_diameter,
            self.environment.max_time
        )

    def _create_agent(self) -> Mouse:
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

            envstate = nengo.Node(lambda time, action: env.step(action), size_in=1, size_out=5)
            # can toggle learning on or off
            model.switch = Switch(state=1)
            # place cells give input to actor and critic
            nengo.Connection(envstate[:2], agent.net.input)
            # take actor net as input to decision node
            nengo.Connection(agent.Actor.net.output, agent.DecisionMaker.net.choicenode)
            # execute action in environment
            nengo.Connection(agent.DecisionMaker.net.choicenode, envstate, synapse=0)

            # connect error node
            nengo.Connection(envstate[2], agent.Error.net.errornode[0])
            nengo.Connection(agent.Critic.net.output, agent.Error.net.errornode[1])
            nengo.Connection(model.switch.net.switch, agent.Error.net.errornode[2])
            nengo.Connection(agent.Error.net.errornode[1], agent.Error.net.errornode[3]) # recurrent connection to save last state; TODO: synapse=0 if transmission too bad
            nengo.Connection(agent.Error.net.errornode[0], agent.Actor.net.conn.learning_rule)

            # add Probes
            errorprobe = nengo.Probe(agent.Error.net.errornode[0])
            envprobe = nengo.Probe(envstate)
            switchprobe = nengo.Probe(model.switch.net.switch)
            # actorwprobe = nengo.Probe(agent.Actor.net.conn)
            # criticwprobe = nengo.Probe(agent.Critic.net.conn)
            criticprobe = nengo.Probe(agent.Critic.net.output)
        
        self._probes = {
            "errorprobe": errorprobe,
            "envprobe": envprobe,
            "switchprobe": switchprobe,
            "criticprobe": criticprobe
        }

        return model

    def run(self, backend:str):
        try:
            sim = util.simulate_with_backend(
                    backend, 
                    model=self.Network,
                    duration=self.simulation.steps,
                    timestep=self.simulation.timestep
                )
        except Exception as e:
            if type(e) == KeyboardInterrupt:
                raise e
            print(e)
            print("WARNING: Falling back to CPU backend")
            BACKEND='CPU'
            sim = util.simulate_with_backend(
                    backend, 
                    model=self.Network,
                    duration=self.simulation.steps,
                    timestep=self.simulation.timestep
                )

        self._simulator = sim

    def save(self):
        _dir = self.Directory
        copyfile(self._path, os.path.join(_dir, os.path.split(self._path)[-1]))
        np.save(os.path.join(_dir, "timepoints"), self.Simulator.trange())
        
        for pname, p in self.Probes.items():
            d = self.Simulator.data[p]
            np.save(os.path.join(_dir, pname), d)
        np.save(os.path.join(_dir, "env_actionmemory"), self.Environment.actionmemory)

        # with open(os.path.join(_dir, "sim_data"), "wb") as f:
        #     pickle.dump(self.Simulator.data, f)

    def __str__(self):
        return f'{self.simulation.steps}STEPS_{self.simulation.timestep}DT_{self.environment.max_time}EPISODE'


def load(path:str) -> Manager:
    with open(path, "r") as f:
        cfg = json.load(f)
    sim = cfg["simulation"]
    net = cfg["network"]
    env = cfg["environment"]
    obj = Manager(path, sim, net ,env)

    return obj
