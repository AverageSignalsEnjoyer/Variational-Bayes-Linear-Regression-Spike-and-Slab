from scipy.stats import bernoulli, norm, gamma, multivariate_normal
import numpy as np

import pdb

# np.random.seed(0)

N = 1000
G = 2
K = 2

tau = np.zeros((G, )) + 0.1


W = np.random.randint(low=-10, high=10, size=(G, K))
# W = np.ones(shape=(G, K))
X = norm.rvs(loc=1, scale=1, size=(G, N))
# X = np.ones(shape=(G, K))
Y = np.zeros((G, N))
# X = norm.rvs(loc=1, scale=1, size=(1, N))

# Y = W*X + norm.rvs(loc=0, scale=1/np.sqrt(tau), size=(1, N))

tau_observe = np.zeros((G, )) + 0.1

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

Wa_slab_q = np.zeros(shape=W.shape) + 1
Wa_spike_q = np.zeros(shape=W.shape) + 1

Wb_slab_q = np.zeros(shape=W.shape) + 1
Wb_spike_q = np.zeros(shape=W.shape) + 1

c_alpha_q = np.copy(c_alpha)
d_alpha_q = np.copy(d_alpha)

c_tau_q = np.copy(c_tau)
d_tau_q = np.copy(d_tau)

for e in range(20):

    # q(a)
    for i in range(K):
        E_z = 0
        E_zww = 0
        for l in range(G):
            E_z += pi[l] * np.sqrt(2 * np.pi / Wa_slab[l, i])
            E_zww += (1/Wa_slab[l, i] + (pi[l]**2)*(Wb_slab[l, i]/Wa_slab[l, i]) ** 2)

        c_alpha_q[i] = c_alpha[i] + (1/2) * E_z
        d_alpha_q[i] = d_alpha[i] + (1/2) * E_zww

        # print(c_alpha_q, d_alpha_q)

    # q(tau)
    for l in range(G):
        E_ywx = 0
        for j in range(N):

            E_ywx += Y[l, j] ** 2

            for i in range(K):
                E_ywx += (X[i, j] ** 2) * (1/Wa_slab[l, i] + (pi[l]**2)*((Wb_slab[l, i]/Wa_slab[l, i]) ** 2) +
                                           1/Wa_spike[l, i] + (1-pi[l]**2)*((Wb_spike[l, i]/Wa_spike[l, i]) ** 2))

            for i in range(K):
                for i_ in range(K):
                    if i != i_:
                        E_ywx += 2 * X[i, j] * X[i_, j] * (pi[l]**2) * (Wb_slab[l, i]/Wa_slab[l, i] + Wb_spike[l, i]/Wa_spike[l, i]) * (Wb_slab[l, i_]/Wa_slab[l, i_] + Wb_spike[l, i_]/Wa_spike[l, i_])

            for i in range(K):
                E_ywx += -2 * Y[l, j] * X[i, j] * pi[l] * (Wb_slab[l, i]/Wa_slab[l, i] + Wb_spike[l, i]/Wa_spike[l, i])

        if E_ywx < 0:
            pdb.set_trace()

        c_tau_q[l] = c_tau[l] + N/2
        d_tau_q[l] = d_tau[l] + (1/2) * E_ywx

        # print(c_tau_q, d_tau_q)

    # q(w, z)
    for l in range(G):
        for i in range(K):

            # Quadratic term
            Wli_a_slab_q = 0

            for j in range(N):
                Wli_a_slab_q += (c_tau_q[l] / d_tau_q[l]) * (X[i, j] ** 2) * (pi[l]**2)

            Wli_a_slab_q += c_alpha_q[i] / d_alpha_q[i]

            # Linear term
            Wli_b_slab_q = 0

            for j in range(N):
                for i_ in range(K):
                    if i_ != i:
                        Wli_b_slab_q += - X[i, j] * X[i_, j] * (Wb_slab[l, i_] / Wa_slab[l, i_] * pi[l]**2 + Wb_spike[l, i_]/Wa_spike[l, i_]) * (c_tau_q[l] / d_tau_q[l])

                Wli_b_slab_q += Y[l, j] * X[i, j] * (c_tau_q[l] / d_tau_q[l]) * pi[l]

            Wa_slab_q[l, i] = Wli_a_slab_q
            Wb_slab_q[l, i] = Wli_b_slab_q

            Wa_spike_q[l, i] = 10e10
            Wb_spike_q[l, i] = 1

    # Update
    Wa_slab = np.copy(Wa_slab_q)
    Wb_slab = np.copy(Wb_slab_q)
    Wa_spike = np.copy(Wa_spike_q)
    Wb_spike = np.copy(Wb_spike_q)

    # c_alpha = np.copy(c_alpha_q)
    # d_alpha = np.copy(d_alpha_q)
    # c_tau = np.copy(c_tau_q)
    # d_tau = np.copy(d_tau_q)

    # print(c_alpha_q / d_alpha_q)
    # print(c_tau_q / d_tau_q)
    print(Wb_slab / Wa_slab)
    # print(Wa_slab)

print(W)
pdb.set_trace()
