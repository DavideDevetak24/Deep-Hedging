import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

S0 = 100
v0 = 0.05

T = 1
N = 360
dt = T/N

r = 0.03
mu = 0.10 #to mu=r if risk neutral
kappa = 2
theta = 0.04
sigma_v = 0.1 #volatility of volatility
rho = 0.7 #corr of BMs

n_simulations = 100


data_S = np.zeros((n_simulations, N))
data_v = np.zeros((n_simulations, N))
t = np.linspace(0, T, N)

W_mean = np.array([0,0])
W_covariance = np.array([[1, rho],
                       [rho,1]])


for i in range(n_simulations):
    np.random.seed(42+i)
    
    S = np.zeros(N)
    v = np.zeros(N)
    S[0] = S0
    v[0] = v0

    W = np.random.multivariate_normal(W_mean, W_covariance, (1,N))
    W_s = W[:, :, 0].flatten()
    W_v = W[:, :, 1].flatten()

    for j in range(1,N):
        #Euler-Maruyama Discretization
        v[j] = v[j-1] + kappa * (theta - v[j-1]) * dt + sigma_v * np.sqrt(max(v[j-1], 0)) * np.sqrt(dt) * W_v[j-1]
        S[j] = S[j-1] * np.exp((mu - 1/2 * v[j-1]) * dt + np.sqrt(v[j-1]) * np.sqrt(dt) * W_s[j-1])

    data_S[i,:] = S
    data_v[i,:] = v

data_S = pd.DataFrame(data_S).T
data_v = pd.DataFrame(data_v).T


plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(data_S)
plt.title('Asset Price Process - Heston')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(data_v)
plt.title('Variance Process - Heston')
plt.xlabel('t')
plt.ylabel('v(t)')
plt.grid()

plt.tight_layout()
plt.show()



def black_scholes(S, K, r, T, IV, q, option_type='Call'):
    d1 = (np.log(S / K) + (r - q + (IV**2 / 2)) * T) / ( IV* np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - (IV**2 / 2)) * T) / (IV * np.sqrt(T))

    if option_type == 'Call':
        call = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        return call
    if option_type == 'Put':        
        put = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
        return put
    else:
        print('Wrong option type: choose between "Call" and "Put"')



