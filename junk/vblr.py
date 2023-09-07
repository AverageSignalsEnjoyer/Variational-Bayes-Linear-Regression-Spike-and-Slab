from scipy.stats import bernoulli, norm, gamma
import numpy as np

import pdb

beta = 0.1
a0 = 1
b0 = 1

N = 100000
x = norm.rvs(loc=1, scale=1, size=N)
w = 0.4
y = w*x + norm.rvs(loc=0, scale=1/np.sqrt(beta), size=N)

aq = 1
bq = 1
sN = 1
aN = a0
bN = b0

for i in range(10):

    Ew2 = 1/aq + (bq/aq) ** 2
    Ealpha = aN / bN

    # q(alpha)
    aN = a0 + 1/2
    bN = b0 + Ew2

    # q(w)
    aq = np.dot(x,x) * beta + Ealpha
    bq = np.dot(x,y) * beta

    print(aN, bN, aq, bq, Ew2, bq/aq)

pdb.set_trace()