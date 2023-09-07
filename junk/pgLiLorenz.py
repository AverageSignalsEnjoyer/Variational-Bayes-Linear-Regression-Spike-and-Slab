import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

from myodes import lorenz_Li, sigmoid
from itertools import combinations
import pdb

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Generate measurement data
dt = .002

t_train = np.arange(0, 4, dt)
x0_train = [20, 4, 0, 0]
a = 2.6
b = 0.44
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(lorenz_Li, t_train_span, x0_train,
                    t_eval=t_train, args=(a, b), **integrator_keywords).y.T

axLabels = ['x', 'y', 'z', 'u']
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, (p0, p1) in enumerate(combinations(range(x_train.shape[1]), 2)):
    axs[i // 3, i % 3].plot(x_train[:, p0], x_train[:, p1])
    axs[i // 3, i % 3].set(xlabel=axLabels[p0], ylabel=axLabels[p1])

fig.show()

fig, axs = plt.subplots(2, 2)

for i in range(x_train.shape[1]):
    axs.ravel()[i].plot(t_train, x_train[:, i])

fig.show()

model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()


pdb.set_trace()

