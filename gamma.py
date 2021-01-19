import numpy as np
import matplotlib.pyplot as plt

def k(g, t):
    return .5 * (np.abs(g**(t-1) - g**t) + np.abs(g**(t+1) - g**t))

def F(g, t, T):
    return (1 / (T - 2)) * np.sum(np.array([k(g, n) for n in t[1:-1]]))

diameter = 2
#speed = 0.3
#dt = 0.001
#step_size = speed * dt
step_size = .002 # actor test env

L = diameter / 2
T = L / step_size
t = np.arange(T)

gamma_lower = .9
gamma_upper = 1.0
gamma_precision = 10000

gammas = np.linspace(gamma_lower, gamma_upper, num=gamma_precision, endpoint=False)
fitness = np.array([F(gamma, t, T) for gamma in gammas])

print('Optimal gamma is ' + str(gammas[np.argmax(fitness)]) + ' with F=' + str(np.max(fitness)))

plt.figure()
plt.title("Gamma optimisation")
plt.plot(gammas, fitness)
plt.xlabel("Gamma")
plt.ylabel("Goodness-of-fit")
plt.show()
