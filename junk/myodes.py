import numpy as np
import pdb
def lorenz_Li(t, x, a, b):
    return [
        x[1]-x[0],
        -x[0]*x[2] + x[3],
        x[0]*x[1] - a*sigmoid(x[0], -10, -10),
        -b*x[0]
    ]


def lorenz_Moon(t, x, rT, sigma, b, rC, Le, s):
    return [
        sigma*(x[1]-x[0]) - sigma*Le*x[4] + s*x[3],
        -x[0]*x[2] + rT*x[0] - x[1],
        x[0]*x[1] - b*x[2],
        -x[0] - sigma*x[3],
        -x[0]*x[5] + rC*x[0] - Le*x[4],
        x[0]*x[4] - Le*b*x[5]
    ]


def sigmoid(x, w, c):
    return 1/(1+np.exp(-c*(x-w)))