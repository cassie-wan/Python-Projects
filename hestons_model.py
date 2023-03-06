import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

S0 = 2300
r = 0.01
q = 0.02
T = 1.0
N = 252 # number of steps
M = 100 # number of simulations

kappa = 0.7
theta = 0.16**2
v0 = 0.25**2  # current instantaneous return variance
rho = -0.8
xi = 0.2  # volatility of volatility

def hestons_model(S0, T, N, M, kappa, theta, v0, rho, xi):
    mu = np.array([0, 0])
    cov = np.array([[1, rho], 
                    [rho, 1]])
    dt = T/N

    St = np.full((N+1, M), fill_value=S0)
    vt = np.full((N+1, M), fill_value=v0)

    Zt = np.random.multivariate_normal(mu, cov, size = (M, N))

    for i in range(1, N+1):
        St[i] = St[i-1] * np.exp((r - q - 0.5 * vt[i-1])*dt + np.sqrt(vt[i-1] * dt) * Zt[:, i-1,0])
        vt[i] = np.maximum(vt[i-1] + kappa * (theta - vt[i-1]) * dt + xi * np.sqrt(vt[i-1] * dt) * Zt[:,i-1,1],0)

    return St, vt

stock_price, vol = hestons_model(S0, T, N, M, kappa, theta, v0, rho, xi)

fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,5))
time = np.linspace(0,T,N+1)
ax1.plot(time,stock_price)
ax1.set_title('Heston Model Asset Prices')
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Prices')

ax2.plot(time,vol)
ax2.set_title('Heston Model Variance Process')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')

plt.show()

payoff = np.where(stock_price[-1, :] - 100 < 0, 0, stock_price[-1, :] - 100)
c_mc = np.exp(-r * T) * np.mean(payoff)
print('Option price via Heston\'s model:', c_mc)