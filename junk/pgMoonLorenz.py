import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

from myodes import lorenz_Moon
from itertools import combinations
import pdb

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Generate measurement data
dt = .002

t_train = np.arange(0, 10, dt)
x0_train = [10, -10, 10, -10, 10, -10]

rC = 2.8
Le = 10
rT = 80.9
s = 10
sigma = 10
b = 8/3

t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(lorenz_Moon, t_train_span, x0_train,
                    t_eval=t_train, args=(rT, sigma, b, rC, Le, s), **integrator_keywords).y.T

plt.plot(x_train[:, 1], x_train[:, 2])
plt.show()

model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()


pdb.set_trace()

