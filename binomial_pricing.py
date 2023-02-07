from math import exp, log, sqrt
import numpy as np
import matplotlib.pyplot as plt

N = 1001
S0 = 40
K = 40
r = 0.04
sigma = 0.3
T = 1/2
call = False
European = False

dt = T/N
u = exp(sigma * sqrt(dt))
d = 1/u
q = (exp(r * dt) - d)/(u - d)

S = np.zeros((N+1, N+1), float) # stock price
P = np.zeros((N+1, N+1), float) # early exercise payoff
for j in range(N+1):
    for i in range(j+1):
        S[i, j] = S0 * u**(j-i) * d**i
        if call == True:
            P[i, j] = max(S[i, j] - K, 0)
        else:
            P[i, j] = max(K - S[i, j], 0)

Phi = np.zeros((N+1, N+1), float)
for i in range(N+1):
    Phi[i, i] = q
    if i != N:
        Phi[i, i+1] = 1-q    
Phi = Phi * exp(-r * dt)

V = np.zeros((N+1, N+1), float)
compare = np.full((N+1, N+1), True)
V[:, N] = P[:, N]
if European == True:
    for j in reversed(range(N)):
        V[:, j] = Phi @ V[:, j + 1]
        V[j + 1, j] = 0
else:
    for j in reversed(range(N)):
        ori_Vi = np.copy(Phi @ V[:, j + 1])
        V[:, j] = np.fmax(P[:, j], Phi @ V[:, j + 1])
        compare[:, j] = (ori_Vi == V[:, j])
        V[j + 1, j] = 0
V[0, 0]

x = []
y = []

for j in range(N+1):
    get = False
    for i in range(j):
        if (compare[i, j] == False) and (get == False):
            x.append(j)
            y.append(S[i, j])
            get = True

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()      




