import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(S0, mu, sigma, T, N, seed=None):
    np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    W = np.random.standard_normal(size=N+1)
    W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return t, S

# Parameters
S0 = 100   # initial stock price
mu_values = [0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]  # drift (expected return)
sigma = 0.2  # volatility (standard deviation of returns)
T = 1.0    # time horizon (1 year)
N = 252    # number of time steps (trading days in a year)

# Plot
plt.figure(figsize=(10, 6))

for mu in mu_values:
    t, S = geometric_brownian_motion(S0, mu, sigma, T, N)
    plt.plot(t, S, lw=2, label=f'mu = {mu}')

plt.title('Geometric Brownian Motion with Different mu Values')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()



