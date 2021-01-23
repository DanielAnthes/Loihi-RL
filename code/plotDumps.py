import pandas as pd
import numpy as np
import pathlib
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Read in csv dumps

dump = pathlib.Path('../dumps/')
t = np.genfromtxt(dump / "trange.csv", delimiter=",")
sim_error = np.genfromtxt(dump / "sim_error.csv", delimiter=",")
state = np.genfromtxt(dump / "state.csv", delimiter=",")
reward = np.genfromtxt(dump / "reward.csv", delimiter=",")
criticout = np.genfromtxt(dump / "criticout.csv", delimiter=",")
learnswitch = np.genfromtxt(dump / "learnswitch.csv", delimiter=",")

# Output
output = pathlib.Path('../plots/')
output.mkdir(exist_ok=True)

fig = plt.figure(figsize=(12,10))
plt.subplot(311)
plt.plot(t, state, label='position')
plt.plot(t, reward, label='reward')
plt.plot(t, criticout, label='value')
plt.legend()
plt.subplot(312)
plt.plot(t, sim_error, label='error')
plt.plot(t, learnswitch, label='learning')
plt.legend()

'''
plt.subplot(413)
plt.title('error node inputs')
plt.plot(t, reward, label='reward')
# plt.plot(t, criticout, label='value')
plt.plot(t, pd.Series(error.valuemem).rolling(5).mean(), label="value", alpha=.5)
plt.plot(t, sim_error, label='error')
# plt.plot(t, sim.data[errorprobe][:,1], label='state')
plt.plot(t, pd.Series(error.statemem).rolling(5).mean(), label="state", alpha=.5)
# plt.plot(t, sim.data[envprobe][:,2], label='reset')
plt.legend()
'''

plt.subplot(313)
plt.plot(t, criticout)
plt.title("Critic Value Prediction")

plt.tight_layout()
fig.savefig(output / "FigureCriticSim.png")
