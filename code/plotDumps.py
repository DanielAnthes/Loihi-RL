import pandas as pd
import numpy as np
import pathlib
from numpy import genfromtxt
import matplotlib.pyplot as plt

BACKEND='GPU'

# Read in csv dumps
dump = pathlib.Path('../dumps/')
t = np.genfromtxt(dump / "{}_trange.csv".format(BACKEND), delimiter=",")
sim_error = np.genfromtxt(dump / "{}_sim_error.csv".format(BACKEND), delimiter=",")
state = np.genfromtxt(dump / "{}_state.csv".format(BACKEND), delimiter=",")
reward = np.genfromtxt(dump / "{}_reward.csv".format(BACKEND), delimiter=",")
criticout = np.genfromtxt(dump / "{}_criticout.csv".format(BACKEND), delimiter=",")
learnswitch = np.genfromtxt(dump / "{}_learnswitch.csv".format(BACKEND), delimiter=",")

# The following is not probe information, but bookkeeping from a list
# We may disable this on Loihi; hence this "failsafe"
statemem = None
try:
    statemem = np.genfromtxt(dump / "{}_statemem.csv".format(BACKEND), delimiter=",")
except:
    print("State memory not available")

# Output
output = pathlib.Path('../plots/')
output.mkdir(exist_ok=True)

fig = plt.figure(figsize=(12,10))
plt.subplot(311)
plt.title('Encoded values for position, reward, value')
plt.plot(t, state, label='agent position')
plt.plot(t, reward, label='reward')
plt.plot(t, criticout, label='critic value')
plt.legend()

plt.subplot(312)
plt.title("Simulation error")
plt.plot(t, sim_error, label='error node probe')
plt.plot(t, learnswitch, label='learning')
if statemem is not None:
    plt.plot(t, pd.Series(statemem).rolling(5).mean(), color='red', label="error state", alpha=.5)
plt.legend()

plt.subplot(313)
plt.plot(t, criticout, label='critic value')
plt.title("Critic Value Prediction")
plt.legend()

plt.tight_layout()
fig.savefig(output / "{}CriticSim.png".format(BACKEND))
