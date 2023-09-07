from scipy.stats import bernoulli, norm, gamma, multivariate_normal, bernoulli
from scipy.special import digamma
import numpy as np

import pdb

T = 1000
S = 4
K = 4

np.random.seed(0)

# Hyperparameters
tau = np.zeros((S, )) + 10
alpha = np.zeros((K, )) + 0.5

c_alpha = np.zeros((K,)) + 1
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
W *= Z

for t in range(T):
    Y[:, t] = multivariate_normal.rvs(mean=np.dot(W, X[:, t]), cov=np.diag(1/tau))

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

Z = np.zeros(W.shape)
# ---- Inference ----


def q_z(c_alpha, d_alpha, s, k, z, pi, rho_z, gamma_z, beta):
    if z == 1:

        temp = pi[k] * np.exp((1/2) * (digamma(c_alpha[k]) - np.log(d_alpha[k]))) \
            * np.sqrt(2*np.pi) * np.exp(rho_z[s, k] ** 2 / (2 * gamma_z[s, k])) / np.sqrt(gamma_z[s, k])

        # if temp > 1e4 or np.isnan(temp):
        #     pdb.set_trace()

        return temp


def q_z_(c_alpha, d_alpha, s, k, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta):

    u = np.log(pi[k]) + (1/2) * (digamma(c_alpha[k]) - np.log(d_alpha[k])) \
        + (rho_z1[s, k] ** 2 / (2 * gamma_z1[s, k])) - (1/2) * np.log(gamma_z1[s, k])

    u -= np.log(1-pi[k]) + (1/2) * np.log(beta) \
         + (rho_z0[s, k] ** 2 / (2 * gamma_z0[s, k])) - (1/2) * np.log(gamma_z0[s, k])

    return 1 / (1 + np.exp(-u))


def E_zww(c_alpha, d_alpha, s, k, pi, rho_z, gamma_z, beta):
    E_z1 = q_z(c_alpha, d_alpha, s, k, 1, pi, rho_z, gamma_z, beta)
    return E_z1 * (1/gamma_z[s, k]**2 + rho_z[s, k]/gamma_z[s, k])


for e in range(10):

    # q(alpha)
    for k in range(K):

        c_alpha_qk = c_alpha[k]
        d_alpha_qk = d_alpha[k]

        for s in range(S):

            # q_z1 = q_z(c_alpha_q, d_alpha_q, s, k, 1, pi, rho_z1, gamma_z1, beta)
            #
            # c_alpha_qk += (1/2) * q_z1
            # d_alpha_qk += (1/2) + q_z1 * (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2)
            #
            # c_alpha_qk += (1/2) * q_z1
            # d_alpha_qk += (1/2) + q_z1 * (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2)

            # q_z1 = q_z(c_alpha_q, d_alpha_q, s, k, 1, pi, rho_z1, gamma_z1, beta)
            # q_z0 = q_z(c_alpha_q, d_alpha_q, s, k, 0, pi, rho_z0, gamma_z0, beta)
            #
            # u = np.log(q_z1) - np.log(q_z0)
            # w = 1 / (1 + np.exp(-u))

            w = q_z_(c_alpha_q, d_alpha_q, s, k, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)

            c_alpha_qk += (1/2) * w
            d_alpha_qk += (1/2) * w * (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2)

            if c_alpha_qk <= 0 or d_alpha_qk <= 0:
                pdb.set_trace()

        c_alpha_q[k] = c_alpha_qk
        d_alpha_q[k] = d_alpha_qk

    # print(c_alpha_q)
    # print(d_alpha_q)
    # pdb.set_trace()

    # q(tau)
    E_w = np.zeros(shape=W.shape)
    for s in range(S):
        for k in range(K):

            w = q_z_(c_alpha_q, d_alpha_q, s, k, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
            E_w[s, k] = (rho_z1[s, k] / gamma_z1[s, k]) * w + (rho_z0[s, k] / gamma_z0[s, k]) * (1-w)

    for s in range(S):

        c_tau_qs = c_tau[s]
        d_tau_qs = d_tau[s]

        c_tau_qs += T/2

        # for t in range(T):
        #
        #     d_tau_qs += (1/2) * Y[s, t]**2
        #
        #     for k in range(K):
        #
        #         # q_z1 = q_z(c_alpha_q, d_alpha_q, s, k, 1, pi, rho_z1, gamma_z1, beta)
        #         # q_z0 = q_z(c_alpha_q, d_alpha_q, s, k, 0, pi, rho_z0, gamma_z0, beta)
        #         #
        #         # u = np.log(q_z1) - np.log(q_z0)
        #         # w = 1 / (1 + np.exp(-u))
        #
        #         w = q_z_(c_alpha_q, d_alpha_q, s, k, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
        #
        #         # E_w = (rho_z1[s, k] / gamma_z1[s, k]) * q_z1 + (rho_z0[s, k] / gamma_z0[s, k]) * q_z0
        #         E_w = (rho_z1[s, k] / gamma_z1[s, k]) * w + (rho_z0[s, k] / gamma_z0[s, k]) * (1-w)
        #
        #         d_tau_qs += - Y[s, t] * E_w * X[k, t]
        #
        #         for k_ in range(K):
        #             if k_ != k:
        #
        #                 # q_z1_ = q_z(c_alpha_q, d_alpha_q, s, k_, 1, pi, rho_z1, gamma_z1, beta)
        #                 # q_z0_ = q_z(c_alpha_q, d_alpha_q, s, k_, 0, pi, rho_z0, gamma_z0, beta)
        #                 #
        #                 # u_ = np.log(q_z1_) - np.log(q_z0_)
        #                 # w_ = 1 / (1 + np.exp(-u_))
        #
        #                 w_ = q_z_(c_alpha_q, d_alpha_q, s, k_, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
        #
        #                 # E_w_ = (rho_z1[s, k_] / gamma_z1[s, k_]) * q_z1_ + (rho_z0[s, k_] / gamma_z0[s, k_]) * q_z0_
        #                 E_w_ = (rho_z1[s, k_] / gamma_z1[s, k_]) * w_ + (rho_z0[s, k_] / gamma_z0[s, k_]) * (1-w_)
        #
        #                 d_tau_qs += E_w * E_w_ * X[k, t] * X[k_, t]
        #
        #         # E_ww = q_z1 * (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2) \
        #         #     + q_z0 * (1/gamma_z0[s, k] + (rho_z0[s, k]/gamma_z0[s, k]) ** 2)
        #
        #         E_ww = w * (1/gamma_z1[s, k] + (rho_z1[s, k]/gamma_z1[s, k]) ** 2) \
        #             + (1-w) * (1/gamma_z0[s, k] + (rho_z0[s, k]/gamma_z0[s, k]) ** 2)
        #
        #         d_tau_qs += (1/2) * E_ww * X[k, t]**2
        #
        #     # if c_tau_qs <= 0 or d_tau_qs <= 0:
        #     #     pdb.set_trace()

        b = E_w[s, :]
        temp = (1/2) * Y[s, :].dot(Y[s, :].T) - b.T.dot(X).dot(Y[s, :]) + (1/2)*b.T.dot(X).dot(X.T).dot(b)

        c_tau_q[s] = c_tau_qs
        d_tau_q[s] = d_tau_qs + temp

        if c_tau_qs <= 0 or d_tau_qs <= 0:
            pdb.set_trace()


    # print(c_tau_q)
    # print(d_tau_q)
    # pdb.set_trace()

    # q(w, z)
    for s in range(S):

        E_taus = c_tau_q[s] / d_tau_q[s]

        for k in range(K):

            gamma_sk_z1 = 0
            gamma_sk_z0 = 0

            rho_sk = 0

            E_alphak = c_alpha_q[k] / d_alpha_q[k]
            # E_alphak = alpha[k]

            # Quadratic term
            for t in range(T):
                gamma_sk_z1 += E_taus * (X[k, t] ** 2)
                gamma_sk_z0 += E_taus * (X[k, t] ** 2)

            gamma_sk_z1 += E_alphak
            gamma_sk_z0 += beta

            # Linear term
            for t in range(T):

                rho_sk += E_taus * Y[s, t] * X[k, t]

                for k_ in range(K):
                    if k_ != k:

                        # q_z1_ = q_z(c_alpha_q, d_alpha_q, s, k_, 1, pi, rho_z1, gamma_z1, beta)
                        # q_z0_ = q_z(c_alpha_q, d_alpha_q, s, k_, 0, pi, rho_z0, gamma_z0, beta)
                        #
                        # # E_w_ = (rho_z1[s, k_] / gamma_z1[s, k_]) * q_z1_ + (rho_z0[s, k_] / gamma_z0[s, k_]) * q_z0_
                        #
                        # u_ = np.log(q_z1_) - np.log(q_z0_)
                        # w_ = 1 / (1 + np.exp(-u_))

                        w_ = q_z_(c_alpha_q, d_alpha_q, s, k_, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)

                        E_w_ = (rho_z1[s, k_] / gamma_z1[s, k_]) * w_ + (rho_z0[s, k_] / gamma_z0[s, k_]) * (1-w_)

                        rho_sk += - E_taus * E_w_ * X[k, t] * X[k_, t]

            gamma_z1[s, k] = gamma_sk_z1
            gamma_z0[s, k] = gamma_sk_z0

            rho_z1[s, k] = rho_sk
            rho_z0[s, k] = rho_sk

            # print(rho_sk)
            # pdb.set_trace()

    # print(rho_z1 / gamma_z1)

    for s in range(S):
        for k in range(K):

            # q_z1 = q_z(c_alpha_q, d_alpha_q, s, k, 1, pi, rho_z1, gamma_z1, beta)
            # q_z0 = q_z(c_alpha_q, d_alpha_q, s, k, 0, pi, rho_z0, gamma_z0, beta)
            #
            # u = np.log(q_z1) - np.log(q_z0)
            # w = 1 / (1 + np.exp(-u))

            w = q_z_(c_alpha_q, d_alpha_q, s, k, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)

            Z[s, k] = w

    np.set_printoptions(precision=1)
    np.set_printoptions(suppress=True)
    print(Z)
    print(rho_z1 / gamma_z1 * Z + rho_z0 / gamma_z0 * (1-Z))
    print(W)
    # print(np.exp(rho_z1 ** 2 / (2 * gamma_z1)))
    print()
    print(c_alpha_q / d_alpha_q)
    print(c_tau_q / d_tau_q)

    # pdb.set_trace()