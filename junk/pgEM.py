import numpy as np
import pdb
from sklearn.linear_model import LinearRegression
from scipy.stats import bernoulli, norm, gamma, multivariate_normal

np.random.seed(0)

T = 2000
S = 3
K = 10

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

model = LinearRegression()
model.fit(X.T, Y.T)

np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)

print(model.coef_)
print(W)
pdb.set_trace()