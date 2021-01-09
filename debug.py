import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb as hsv2rgb
import nengo

from Environment import Maze
import util
import Learning


def opt_angle(in_):
    x,y = in_[0:2]
    # avoid 0 division
    x += 1e-8
    y += 1e-8
    a = np.arctan2(y, x)
    
    return a - np.pi

env = Maze(max_time=60, speed=0.3)
with nengo.Network() as model:
    envstate = nengo.Node(lambda t, action: env.step(action), size_in=1, size_out=5)
    input = nengo.Ensemble(
            n_neurons=250,
            dimensions=2,
            radius=env.diameter / 1.75
    )
    actor = nengo.Ensemble(
                n_neurons=500, 
                dimensions=1, 
                radius= 1.2*np.pi,
                # neuron_type=nengo.Direct()
    )
    nengo.Connection(envstate[:2], input, synapse=0)
    conn = nengo.Connection(envstate[:2], actor,
                            function = opt_angle, synapse=0,
                            solver=nengo.solvers.LstsqL2(weights=True),
                            learning_rule_type=Learning.TDL(learning_rate=1e-5))
    
    opt = nengo.Node(lambda t, pos: opt_angle(pos), size_in=2, size_out=1)
    nengo.Connection(envstate[:2], opt)

    nengo.Connection(actor, envstate, synapse=0)

    optP = nengo.Probe(opt, synapse=0)
    inP = nengo.Probe(input, synapse=0)
    actorP = nengo.Probe(actor, synapse=0)
    envP = nengo.Probe(envstate, synapse=0)


with nengo.Simulator(model, 0.001) as sim:
    sim.run(30)
# sim = util.simulate_with_backend("CPU", model, duration=30, timestep=env.timestep)

t = sim.trange()
dat_opt = sim.data[optP]
dat_inp = sim.data[inP]
dat_act = sim.data[actorP]
dat_env = sim.data[envP]


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='polar')

episode_indices = np.where(dat_env[:,3] == 1.0)
episode_indices = np.append(episode_indices[0], t[-1] / env.timestep)
episode_indices = episode_indices.astype(int)

ss=1
last_episode = 2
for i, episode in enumerate(episode_indices):
    if last_episode >= episode:
        continue
    
    vx = dat_inp[last_episode:episode:ss,0]
    vy = dat_inp[last_episode:episode:ss,1]
    thetas = np.arctan2(vy, vx)
    rs = np.sqrt(vx**2 + vy**2)
    # if np.any(rs < 0.1):
        # print(f"ALERT IN INPUT EPISODE < {episode}. Found at {np.where(rs < 0.1)}")
    ax.plot(thetas, rs, "h",label=f"INP-ENS: {last_episode:d}-{episode:d}", color='blue')
    
    vx = dat_env[last_episode:episode:ss,0]
    vy = dat_env[last_episode:episode:ss,1]
    thetas = np.arctan2(vy, vx)
    rs = np.sqrt(vx**2 + vy**2)
    # if np.any(rs < 0.1):
    #     print(f"ALERT IN EPISODE < {episode}. Found at {np.where(rs < 0.1)}")
    # c = hsv2rgb((130/360,.7,1-.2*i))
    c = "#059244ff"
    ax.plot(thetas, rs, "o",label=f"ENV: {last_episode:d}-{episode:d}", color=c, markersize=2)
    
    theta = np.arctan2(vy[0], vx[0])
    r = np.sqrt(vx[0]**2 + vy[0]**2)
    ax.text(theta, r, str(int(round(last_episode * env.timestep))), alpha=0.8, fontsize=8)
    ax.plot(theta, r, 'o', alpha=0.9, color=c)
    theta = np.arctan2(vy[-1], vx[-1])
    r = np.sqrt(vx[-1]**2 + vy[-1]**2)
    ax.plot(theta, r, 'x', alpha=0.9, color=c)#, markersize=20)
    last_episode = episode+2

ax.legend(loc='upper left')
ax.plot(np.linspace(-np.pi, np.pi, 100), [0.1]*100, color='black', alpha=0.7, lw=.75)
# plt.Circle((0,0),0.5, transform=ax.transData._b, color='red')

# s = 50
# d = dat_opt[0:-1:s].squeeze()
# pos = dat_inp[0:-1:s,:].squeeze()
# for i in range(len(d)):
#     r = np.sqrt(np.sum(pos[i,:]**2))
#     a = np.arctan2(pos[i,1], pos[i,0])
#     if not (0 < a < np.pi/2):
#         continue
#     ax.plot(a,r, "x")
#     theta = opt_angle(pos[i,:])
#     ax.plot([0, theta],[0, 0.15])

ax.set_rmin(0)
ax.set_rmax(1)
ax.grid(True)
plt.show()