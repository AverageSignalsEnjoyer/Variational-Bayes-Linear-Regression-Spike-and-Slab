from scipy.stats import bernoulli, norm, gamma, multivariate_normal, bernoulli
from scipy.special import digamma
import numpy as np

import pdb
T = 2000
S = 3
K = 10

# Hyperparameters
tau = np.zeros((S, )) + 10
alpha = np.zeros((K, )) + 0.5

c_alpha = np.zeros((K,)) + 0.1
d_alpha = np.zeros((K,)) + 1

c_tau = np.zeros((S,)) + 1
d_tau = np.zeros((S,)) + 1

pi = np.zeros((K,)) + 0.1
beta = 1e20

X = norm.rvs(loc=1, scale=1, size=(K, T))
Y = np.zeros((S, T))

W = np.zeros((S, K))
for s in range(S):
    for k in range(K):
        W[s, k] = norm.rvs(loc=0, scale=1/np.sqrt(alpha[k]))

Z = bernoulli.rvs(p=0.5, size=(S, K))
# W *= Z

for s in range(S):
    for t in range(T):
        Y[s, t] = norm.rvs(loc=np.dot(W[s, :], X[:, t]), scale=1/np.sqrt(tau[s]))

indices = np.random.choice(W.size, np.floor(W.size*0.2).astype(np.int32), replace=False)
indices_2d = np.unravel_index(indices, W.shape)
W[indices_2d] = 0

# ---- Initialize sufficient parameters for distributions ----

# gamma_z1 = np.zeros(shape=W.shape) + 1
# gamma_z0 = np.zeros(shape=W.shape) + 1

gamma_z1 = np.abs(norm.rvs(loc=0, scale=1, size=W.shape))
gamma_z0 = np.abs(norm.rvs(loc=0, scale=1, size=W.shape))

rho_z1 = np.zeros(shape=W.shape) + 1
rho_z0 = np.zeros(shape=W.shape) + 1

c_alpha_q = np.zeros(shape=c_alpha.shape) + 1
d_alpha_q = np.zeros(shape=d_alpha.shape) + 1

c_tau_q = np.zeros(shape=c_tau.shape) + 1
d_tau_q = np.zeros(shape=d_tau.shape) + 1

# ---- Inference ----

for e in range(100):

    # q(alpha)
    for k in range(K):

        c_alpha_qk = c_alpha[k]
        d_alpha_qk = d_alpha[k]

        for s in range(S):

            c_alpha_qk += (1/2)
            d_alpha_qk += (1/2) + (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2)

            if c_alpha_qk <= 0 or d_alpha_qk <= 0:
                pdb.set_trace()

        c_alpha_q[k] = c_alpha_qk
        d_alpha_q[k] = d_alpha_qk

    # q(tau)
    for s in range(S):

        c_tau_qs = c_tau[s]
        d_tau_qs = d_tau_q[s]

        m = 0
        n = 0
        p = 0

        c_tau_qs += T/2

        for t in range(T):

            d_tau_qs += (1/2) * Y[s, t]**2

            m += (1/2) * Y[s, t]**2

            for k in range(K):

                E_w = (rho_z1[s, k] / gamma_z1[s, k])
                d_tau_qs += - Y[s, t] * X[k, t] * E_w

                n += - Y[s, t] * X[k, t] * E_w

                for k_ in range(K):
                    if k_ != k:

                        E_w_ = (rho_z1[s, k_] / gamma_z1[s, k_])
                        d_tau_qs += E_w * E_w_ * X[k, t] * X[k_, t]

                E_ww = (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2)

                d_tau_qs += (1/2) * E_ww * X[k, t]**2

                p += (1/2) * E_w ** 2 * X[k, t]**2

        if c_tau_qs <= 0 or d_tau_qs <= 0:
            pdb.set_trace()

        E_w = rho_z1 / gamma_z1

        b = E_w[s, :]

        temp = (1/2) * Y[s, :].dot(Y[s, :].T) - b.T.dot(X).dot(Y[s, :]) + (1/2)*b.T.dot(X).dot(X.T).dot(b)

        print(m)
        print(n)
        print(p)

        print((1/2) * Y[s, :].dot(Y[s, :].T))
        print(- b.T.dot(X).dot(Y[s, :]))
        print((1/2)*b.T.dot(X).dot(X.T).dot(b))

        pdb.set_trace()

        c_tau_q[s] = c_tau_qs
        d_tau_q[s] = d_tau_qs + temp

        # print(d_tau_qs)


    # q(w, z)
    for s in range(S):

        E_taus = c_tau_q[s] / d_tau_q[s]
        # E_taus = tau[s]

        for k in range(K):

            gamma_sk_z1 = 0
            rho_sk = 0

            E_alphak = c_alpha_q[k] / d_alpha_q[k]

            # Quadratic term
            for t in range(T):
                gamma_sk_z1 += E_taus * (X[k, t] ** 2)

            gamma_sk_z1 += E_alphak

            # Linear term
            for t in range(T):

                rho_sk += E_taus * Y[s, t] * X[k, t]

                for k_ in range(K):
                    if k_ != k:

                        E_w_ = (rho_z1[s, k_] / gamma_z1[s, k_])
                        rho_sk += - E_taus * E_w_ * X[k, t] * X[k_, t]

            gamma_z1[s, k] = gamma_sk_z1

            rho_z1[s, k] = rho_sk

    np.set_printoptions(precision=1)
    np.set_printoptions(suppress=True)

    print(rho_z1 / gamma_z1)
    print(W)

pdb.set_trace()