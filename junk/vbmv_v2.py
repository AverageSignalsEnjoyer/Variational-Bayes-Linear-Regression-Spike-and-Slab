from scipy.stats import bernoulli, norm, gamma, multivariate_normal
from scipy.special import digamma
import numpy as np

import pdb

# np.random.seed(0)

N = 1000
G = 2
K = 5

tau = np.zeros((G, )) + 0.1


W = np.random.randint(low=-10, high=10, size=(G, K))
# W = np.ones(shape=(G, K))
X = norm.rvs(loc=1, scale=1, size=(K, N))
# X = np.ones(shape=(G, K))
Y = np.zeros((G, N))
# X = norm.rvs(loc=1, scale=1, size=(1, N))

# Y = W*X + norm.rvs(loc=0, scale=1/np.sqrt(tau), size=(1, N))

tau_observe = np.zeros((G, )) + 10

c_alpha = np.zeros((K,)) + 1
d_alpha = np.zeros((K,)) + 1

c_tau = np.zeros((G,)) + 1
d_tau = np.zeros((G,)) + 1

pi = np.zeros((K,)) + 0.9

for j in range(N):
    Y[:, j] = multivariate_normal.rvs(mean=np.dot(W, X[:, j]), cov=np.diag(1/tau_observe))

indices = np.random.choice(W.size, np.floor(W.size*0.2).astype(np.int32), replace=False)
indices_2d = np.unravel_index(indices, W.shape)
W[indices_2d] = 0

# ---- Initialize sufficient parameters for distributions ----

Wa_slab = np.zeros(shape=W.shape) + 1
Wa_spike = np.zeros(shape=W.shape) + 1

Wb_slab = np.zeros(shape=W.shape) + 1
Wb_spike = np.zeros(shape=W.shape) + 1

c_alpha_q = np.zeros(shape=c_alpha.shape) + 1
d_alpha_q = np.zeros(shape=d_alpha.shape) + 1

c_tau_q = np.zeros(shape=c_tau.shape) + 1
d_tau_q = np.zeros(shape=d_tau.shape) + 1

for e in range(20):

    # Calculate E_z
    E_z = np.zeros(shape=W.shape)
    for l in range(G):
        for i in range(K):
            E_z[l, i] = pi[i] * np.exp((1/2) * digamma(c_alpha_q[i]) - np.log(d_alpha_q[i])) * np.sqrt(2*np.pi) * np.exp(Wb_slab[l, i]**2 / (2*Wa_slab[l, i])) / np.sqrt(Wa_slab[l, i])
            # E_z[l, i] = pi[i]

    pdb.set_trace()

    # Calculate E_zww
    E_zww = np.zeros(shape=W.shape)
    for l in range(G):
        for i in range(K):
            E_zww[l, i] = 1/Wa_slab[l, i] + (E_z[l, i]*Wb_slab[l, i]/Wa_slab[l, i]) ** 2

    # Calculate E_w
    E_w = np.zeros(shape=W.shape)
    for l in range(G):
        for i in range(K):
            E_w[l, i] = (E_z[l, i]*Wb_slab[l, i]/Wa_slab[l, i]) + ((1-pi[i])*Wb_spike[l, i]/Wa_spike[l, i])

    # Calculate E_ww
    E_ww = np.zeros(shape=W.shape)
    for l in range(G):
        for i in range(K):
            E_ww[l, i] = 1/Wa_slab[l, i] + (E_z[l, i]*Wb_slab[l, i]/Wa_slab[l, i]) ** 2 \
                         + 1/Wa_spike[l, i] + ((1-pi[i])*Wb_spike[l, i]/Wa_spike[l, i]) ** 2

    # # q(a)
    for i in range(K):
        c_alpha_q[i] = c_alpha[i] + (1/2) * np.sum(E_z[:, i])
        d_alpha_q[i] = d_alpha[i] + (1/2) * np.sum(E_zww[:, i])

        # print(c_alpha_q, d_alpha_q)

    E_alpha = c_alpha_q / d_alpha_q

    # q(tau)
    for l in range(G):
        E_ywx = 0
        for j in range(N):

            E_ywx += Y[l, j] ** 2

            for i in range(K):
                E_ywx += (X[i, j] ** 2) * E_ww[l, i]

            for i in range(K):
                for i_ in range(K):
                    if i != i_:
                        E_ywx += 2 * X[i, j] * X[i_, j] * E_w[l, i] * E_w[l, i_]

            for i in range(K):
                E_ywx += -2 * Y[l, j] * X[i, j] * E_w[l, i]

        # if E_ywx < 0:
        #     pdb.set_trace()

        c_tau_q[l] = c_tau[l] + N/2
        d_tau_q[l] = d_tau[l] + (1/2) * E_ywx

    E_tau = c_tau_q / d_tau_q
    # E_tau = tau_observe

        # print(c_tau_q, d_tau_q)

    # q(w, z)
    for l in range(G):
        for i in range(K):

            # Quadratic term
            Wli_a_slab_q = 0

            for j in range(N):
                Wli_a_slab_q += E_tau[l] * (X[i, j] ** 2)

            Wli_a_slab_q -= E_alpha[i]

            # Linear term
            Wli_b_slab_q = 0

            for j in range(N):
                for i_ in range(K):
                    if i_ != i:
                        Wli_b_slab_q += - E_tau[l] * X[i, j] * X[i_, j] * E_w[l, i_]

                Wli_b_slab_q += E_tau[l] * Y[l, j] * X[i, j]

            Wa_slab[l, i] = Wli_a_slab_q
            Wb_slab[l, i] = Wli_b_slab_q

            Wa_spike[l, i] = 10e10
            Wb_spike[l, i] = 1

    print(E_alpha)
    print(E_tau)
    print(Wb_slab / Wa_slab)

print(W)


pdb.set_trace()
