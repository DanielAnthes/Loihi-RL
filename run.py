import os
import matplotlib.pyplot as plt

import util
from config_parser import load


BACKEND = 'CPU' # choice of CPU, GPU and LOIHI
PLOT_TUNING = False  # Be aware that producing this plot is quite slow

manager = load("config.json")
manager.run(BACKEND)
manager.save()

sim = manager.Simulator
env = manager.Environment

if PLOT_TUNING:
    util.plot_tuning_curves(manager.Network, manager.Agent.net.input)

criticprobe = manager.Probes["criticprobe"]
envprobe = manager.Probes["envprobe"]
errorprobe = manager.Probes["errorprobe"]
switchprobe = manager.Probes["switchprobe"]


cdat = sim.data[criticprobe]
print(cdat.shape)
fig = util.plot_sim(sim, envprobe, errorprobe, switchprobe)
fig.savefig(os.path.join(manager.Directory, "sim.png"))
#util.plot_value_func(model, agent, env, BACKEND)
fig = util.plot_trajectories(sim, env, envprobe, cdat)
fig.savefig(os.path.join(manager.Directory, "trajectory.png"))
#util.plot_actions_by_activation(env, agent)
#util.plot_actions_by_probability(env, agent)
#util.plot_actions_by_decision(env)
#util.plot_weight_evolution_3d(sim, actorwprobe, title="Weight evolution of place cells to actor")
#util.plot_weight_evolution_2d(sim, criticwprobe, title="Weight evolution of place cells to critic")
#util.plot_place_cell(model, agent, env, BACKEND, [0.0, 0.0])
plt.show()
