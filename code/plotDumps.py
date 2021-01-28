import pandas as pd
import numpy as np
import pathlib
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Read in csv dumps
# Use different folder for critic-only and for actor-critic
#folder = '../dumps/'  # critic only
dump = pathlib.Path(os.path.join(os.path.dirname(__file__), '../dumps-actor-critic/'))

# From which back end was the data generated?
BACKEND='CPU'

t = np.genfromtxt(dump / "{}_trange.csv".format(BACKEND), delimiter=",")
sim_error = np.genfromtxt(dump / "{}_sim_error.csv".format(BACKEND), delimiter=",")
state = np.genfromtxt(dump / "{}_state.csv".format(BACKEND), delimiter=",")
reward = np.genfromtxt(dump / "{}_reward.csv".format(BACKEND), delimiter=",")
criticout = np.genfromtxt(dump / "{}_criticout.csv".format(BACKEND), delimiter=",")
learnswitch = np.genfromtxt(dump / "{}_learnswitch.csv".format(BACKEND), delimiter=",")

activity = None

try:
    # Stuff that's not in criticTest but for combined actor and critic
    delta = np.genfromtxt(dump / "{}_delta.csv".format(BACKEND), delimiter=",")
    delta_positive = np.genfromtxt(dump / "{}_delta_positive.csv".format(BACKEND), delimiter=",")
    delta_negative = np.genfromtxt(dump / "{}_delta_negative.csv".format(BACKEND), delimiter=",")
    activity = np.genfromtxt(dump / "{}_activity.csv".format(BACKEND), delimiter=",")
except:
    print("Selecting critic-only data")

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

# Subplot 1
fig = plt.figure(figsize=(12,10))
plt.subplot(311)
plt.title('Encoded values for position, reward, value', alpha=0.6)
plt.plot(t, state, label='Agent Position', alpha=0.6)
plt.plot(t, reward, label='Reward', alpha=0.6)
plt.plot(t, criticout, label='Critic Value', alpha=0.6)
try:
    plt.plot(t, delta, label="Delta", alpha=0.6)
except:
    print("Delta not recorded")
plt.legend()

# Subplot 2
plt.subplot(312)
plt.title("Simulation error")
axes = plt.gca()
plt.plot(t, sim_error, label='error node probe', alpha=0.6)
plt.plot(t, learnswitch, label='learning', alpha=0.6)
try:
    # TODO what's the timescale on this? 
    plt.scatter(t, delta_positive, s=1, marker='x', label="Positive Delta", alpha=0.6)
    plt.scatter(t, delta_negative, s=1, marker='x', label="Negative Delta", alpha=0.6)
    #plt.scatter(t, p_delta_naught, s=1, marker='x', label="Naught Delta", alpha=0.6)
    #axes.set_ylim([-5e-2, 5e-2])
except:
    print("Delta's not recorded")
if statemem is not None:
    plt.plot(t, pd.Series(statemem).rolling(5).mean(), color='red', label="error state", alpha=.6)
plt.legend()

# Subplot 3
plt.subplot(313)
plt.plot(t, criticout, label='Critic', alpha=0.6)
try:
    plt.plot(t, activity, label='Critic', alpha=0.6)
    plt.title("Critic and Actor Value Prediction")
except:
    print("Actor activity not stored")
    plt.title("Critic Value Prediction")
plt.legend()

plt.tight_layout()
if activity is not None:
    fig.savefig(output / "{}_ActorCriticSim.png".format(BACKEND))
else:
    fig.savefig(output / "{}_CriticSim.png".format(BACKEND))
