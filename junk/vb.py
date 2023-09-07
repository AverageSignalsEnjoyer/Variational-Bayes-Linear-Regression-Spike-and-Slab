from scipy.stats import bernoulli, norm, gamma
from scipy.special import gamma, digamma
import numpy as np
import matplotlib.pyplot as plt
import pdb

np.random.seed(0)

tau = 0.1
alpha = 1
cAlpha = 1
dAlpha = 1
cTau = 1
dTau = 1
k = 0.9
beta = 10e10

N = 100
x = norm.rvs(loc=1, scale=1, size=N)
w = 1
y = w*x + norm.rvs(loc=0, scale=1/np.sqrt(tau), size=N)

# print(y)
# pdb.set_trace()

# Initialize q(w, z)

az1 = 1
az0 = 1
b = 1

Ls = []

for i in range(10):

    # ---- q(alpha) ----

    # E[z]
    Ez1 = k * np.sqrt(2 * np.pi / az1)
    Ez0 = (1-k) * np.sqrt(2 * np.pi / az0)

    # Ez1 = k
    # Ez0 = 1-k

    # E[zw^2]
    Ezw2 = k * (1/az1 + (b/az1) ** 2)

    # Update
    cAlphaN = cAlpha + 1/2 * Ez1
    dAlphaN = dAlpha + 1/2 * Ezw2

    # print(cAlphaN, dAlphaN)

    # ---- q(tau) ----
    # Update
    cTauN = cTau + N/2
    dTauN = dTau + 1/2 * np.dot(y,y) - np.dot(x,y) * (b/az1 + b/az0) + 1/2 * np.dot(x,x) * (1/az1 + (b/az1) ** 2 + 1/az0 + (b/az0) ** 2)

    # print(cTauN, dTauN)

    # ---- q(w, z) ----
    # E[tau]
    Etau = cTauN / dTauN

    # E[alpha]
    Ealpha = cAlphaN / dAlphaN

    az1 = np.dot(x, x) * Etau + Ealpha
    az0 = np.dot(x, x) * Etau + beta
    b = np.dot(x, y) * Etau

    # print(az1, Ez1, Ealpha, Etau, Ezw2, cAlphaN, dAlphaN, cTauN, dTauN)
    # print(b / az1 / Ez1)
    print(b / az1)

    # pdb.set_trace()

    # ---- Calculate ELBO ----
    L = 0

    # E[log p(y | w, tau)]

    L += N/2 * (digamma(cTauN) - np.log(dTauN)) # 1/2 E[log tau]
    L += - N/2 * np.log(np.pi)
    L += - (1/2) * np.dot(y,y) * (cTauN / dTauN) # 1/2 y^2 E[tau]
    L += 2 * np.dot(y, x) * (cTauN / dTauN) * (b/az1 + b/az0) # 2yx E[tau] E[w]
    L += - (1/2) * (cTauN / dTauN) * (1/az1 + (b/az1) ** 2 + 1/az0 + (b/az0) ** 2) # 1/2 E[tau] E[w^2]

    # E[log p(w, z, | alpha)
    Ez = (k * np.sqrt(2 * np.pi / az1))

    L += (1/2) * (digamma(cAlphaN) - np.log(dAlphaN)) * Ez # 1/2 E[log alpha] E[z]
    L += (-1/2 * np.log(np.pi) + np.log(k)) * Ez
    L += (1/2) * (cAlphaN / dAlpha) * (1/az1 + (b/az1) ** 2) # (1/2) E[alpha] E[w^2 z]

    # E[log p(alpha)]

    L += (cAlpha-1) * (digamma(cAlphaN) - np.log(dAlphaN)) - dAlpha * (cAlphaN / dAlphaN)

    # E[log p(tau)]

    L += (cTau-1) * (digamma(cTauN) - np.log(dTauN)) - dTau * (cTauN / dTauN)

    # E[log q(w, z)]

    L -= -(1/2) * az1 * (1/az1 + (b/az1) ** 2) + b * (b/az1)
    L -= -(1/2) * az0 * (1/az0 + (b/az0) ** 2) + b * (b/az0)

    # E[log q(alpha)]

    L -= (cAlphaN-1) * (digamma(cAlphaN) - np.log(dAlphaN)) - dAlphaN * (cAlphaN / dAlphaN)

    # E[log q(tau)]

    L -= (dTauN-1) * (digamma(dTauN) - np.log(dTauN)) - dTauN * (cTauN / dTauN)

    print(L)

    Ls.append(L)


plt.plot(Ls)
plt.show()

pdb.set_trace()