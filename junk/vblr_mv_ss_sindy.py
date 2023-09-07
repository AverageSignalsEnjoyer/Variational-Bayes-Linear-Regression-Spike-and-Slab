from scipy.stats import bernoulli, norm, gamma, multivariate_normal, bernoulli
from scipy.special import digamma
import numpy as np

from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso

from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
import pysindy as ps
import matplotlib.pyplot as plt

import pdb

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Generate measurement data
dt = .02

t_train = np.arange(0, 20, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(lorenz, t_train_span, x0_train,
                    t_eval=t_train, **integrator_keywords).y.T

T, S = x_train.shape

model = ps.SINDy()
model.fit(x_train, t=dt)


def myfunc(t, x, W, model):
    return W @ model.feature_library.transform(x)


temp = solve_ivp(myfunc, t_train_span, x0_train,
                 t_eval=t_train, args=(model.coefficients(), model), **integrator_keywords).y.T

fig, axs = plt.subplots(1, 2, figsize=(15, 10))

axs[0].plot(x_train)
axs[1].plot(temp)

fig.show()

pdb.set_trace()

K = len(model.feature_library.get_feature_names())

# Hyperparameters
tau = np.zeros((S, )) + 10
alpha = np.zeros((K, )) + 0.5

c_alpha = np.zeros((K,)) + 1
d_alpha = np.zeros((K,)) + 1

c_tau = np.zeros((S,)) + 1
d_tau = np.zeros((S,)) + 1

pi = np.zeros((K,)) + 0.05
beta = 1e20

X = model.feature_library.transform(x_train).T
Y = model.differentiate(x_train).T + norm.rvs(loc=0, scale=0.1, size=(x_train.T.shape))

# ---- Initialize sufficient parameters for distributions ----

# gamma_z1 = np.zeros(shape=W.shape) + 1
# gamma_z0 = np.zeros(shape=W.shape) + 1

gamma_z1 = np.abs(norm.rvs(loc=1, scale=1, size=(S, K)))
gamma_z0 = np.abs(norm.rvs(loc=1, scale=1, size=(S, K)))

rho_z1 = np.zeros(shape=(S, K)) + 5
rho_z0 = np.zeros(shape=(S, K)) + 5

c_alpha_q = np.zeros(shape=(K, )) + 20
d_alpha_q = np.zeros(shape=(K, )) + 1

c_tau_q = np.zeros(shape=(S, )) + 1
d_tau_q = np.zeros(shape=(S, )) + 0.5

Z = np.zeros((S, K))

def calc_q_z(c_alpha, d_alpha, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta):

    u = np.log(pi) + (1/2)*(digamma(c_alpha) - np.log(d_alpha)) \
        + (rho_z1**2 / (2*gamma_z1)) - (1/2)*np.log(gamma_z1)

    u -= np.log(1-pi) + (1/2)*np.log(beta) \
        + (rho_z0**2 / (2*gamma_z0)) - (1/2)*np.log(gamma_z0)

    return 1 / (1 + np.exp(-u))


for e in range(10):

    # q(alpha)
    q_z = calc_q_z(c_alpha, d_alpha, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
    E_z = q_z
    E_zww = q_z * (1/gamma_z1 + (rho_z1/gamma_z1)**2)

    c_alpha_q = c_alpha + (1/2)*np.sum(E_z, axis=0)
    d_alpha_q = d_alpha + (1/2)*np.sum(E_zww, axis=0)

    # print(c_alpha_q)
    # print(d_alpha_q)
    # pdb.set_trace()

    # q(tau)
    for s in range(S):
        q_z = calc_q_z(c_alpha, d_alpha, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
        E_w = q_z * (rho_z1/gamma_z1) + (1-q_z) * (rho_z0/gamma_z0)

        c_tau_q[s] = c_tau[s] + T/2
        d_tau_q[s] = d_tau[s] + (1/2) * Y[s, :].dot(Y[s, :].T) - E_w[s, :].T.dot(X).dot(Y[s, :]) + (1/2)*E_w[s, :].T.dot(X).dot(X.T).dot(E_w[s, :])

    # q_z = calc_q_z(c_alpha, d_alpha, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
    # E_w = q_z * (rho_z1/gamma_z1) + (1-q_z) * (rho_z0/gamma_z0)
    #
    # c_tau_q = c_tau + T/2
    # d_tau_q = d_tau + np.diagonal((1/2) * Y.dot(Y.T) - E_w.dot(X).dot(Y.T) + (1/2)*E_w.dot(X).dot(X.T).dot(E_w.T))

    # print(c_tau_q)
    # print(d_tau_q)
    # pdb.set_trace()

    # q(w, z)
    E_tau = c_tau_q / d_tau_q
    E_alpha = c_alpha_q / c_alpha_q

    for s in range(S):
        for k in range(K):

            q_z = calc_q_z(c_alpha, d_alpha, pi, rho_z1, rho_z0, gamma_z1, gamma_z0, beta)
            E_w = q_z * (rho_z1 / gamma_z1) + (1 - q_z) * (rho_z0 / gamma_z0)

            rho_z1_q = 0
            rho_z0_q = 0

            # Quadratic term
            gamma_z1_q = E_tau[s] * (X[k, :] @ X[k, :].T) + E_alpha[k]
            gamma_z0_q = E_tau[s] * (X[k, :] @ X[k, :].T) + beta

            # Linear term
            rho_z1_q += E_tau[s] * (Y[s, :] @ X[k, :].T)
            rho_z0_q += E_tau[s] * (Y[s, :] @ X[k, :].T)

            for k_ in range(K):
                if k_ != k:
                    rho_z1_q -= E_tau[s] * E_w[s, k_] * (X[k, :] @ X[k_, :].T)
                    rho_z0_q -= E_tau[s] * E_w[s, k_] * (X[k, :] @ X[k_, :].T)

            gamma_z1[s, k] = gamma_z1_q
            gamma_z0[s, k] = gamma_z0_q
            rho_z1[s, k] = rho_z1_q
            rho_z0[s, k] = rho_z0_q

    # # Quadratic term
    # for s in range(S):
    #     for k in range(K):
    #
    #         gamma_z1[s, k] = E_tau[s] * (X[k, :] @ X[k, :].T) + E_alpha[k]
    #         gamma_z0[s, k] = E_tau[s] * (X[k, :] @ X[k, :].T) + beta
    #
    # # Linear term
    # for s in range(S):
    #     for k in range(K):
    #
    #         rho_z1[s, k] = E_tau[s] * (Y[s, :] @ X[k, :].T)
    #         rho_z0[s, k] = E_tau[s] * (Y[s, :] @ X[k, :].T)
    #
    #         for k_ in range(K):
    #             if k_ != k:
    #                 rho_z1[s, k] -= E_tau[s] * E_w[s, k_] * (X[k, :] @ X[k_, :].T)
    #                 rho_z0[s, k] -= E_tau[s] * E_w[s, k_] * (X[k, :] @ X[k_, :].T)



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

                        u = np.log(pi) + (1 / 2) * (digamma(c_alpha) - np.log(d_alpha)) \
                            + (rho_z1 ** 2 / (2 * gamma_z1)) - (1 / 2) * np.log(gamma_z1)

                        u -= np.log(1 - pi) + (1 / 2) * np.log(beta) \
                             + (rho_z0 ** 2 / (2 * gamma_z0)) - (1 / 2) * np.log(gamma_z0)

                        q_z = 1 / (1 + np.exp(-u))
                        E_w = q_z * (rho_z1 / gamma_z1) + (1 - q_z) * (rho_z0 / gamma_z0)

                        rho_sk += - E_taus * E_w[s, k_] * X[k, t] * X[k_, t]

            gamma_z1[s, k] = gamma_sk_z1
            gamma_z0[s, k] = gamma_sk_z0

            rho_z1[s, k] = rho_sk
            rho_z0[s, k] = rho_sk

    np.set_printoptions(precision=1)
    np.set_printoptions(suppress=True)
    print(q_z)
    print(rho_z1 / gamma_z1 * q_z + rho_z0 / gamma_z0 * (1-q_z))
    print(model.coefficients())

    print()
    print(c_alpha_q / d_alpha_q)
    print(c_tau_q / d_tau_q)

W = (rho_z1 / gamma_z1) * q_z + rho_z0 / gamma_z0 * (1-q_z)

fig, axs = plt.subplots(1, 3, figsize=(15, 10))

axs[0].plot((W @ X).T)
axs[1].plot(Y.T)

temp = solve_ivp(myfunc, t_train_span, x0_train,
                 t_eval=t_train, args=(W, model), **integrator_keywords).y.T

axs[2].plot(temp)

colors = ['r', 'g', 'b']

for p in range(3):
    for i,j in enumerate(axs[p].lines):
        j.set_color(colors[i])

fig.show()





pdb.set_trace()