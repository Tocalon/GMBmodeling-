import yfinance as yf
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

def plot_stock_gbm(symbol):
    data = yf.download(symbol, start='2021-01-01', end='2022-01-01')
    S0 = data['Close'].iloc[0]  # initial stock price
    returns = data['Close'].pct_change().dropna()
    mu = returns.mean() * 252  # drift (annualized expected return)
    sigma = returns.std() * np.sqrt(252)  # volatility (annualized standard deviation of returns)
    T = 1.0    # time horizon (1 year)
    N = 252    # number of time steps (trading days in a year)

    # Simulate GBM
    t, S = geometric_brownian_motion(S0, mu, sigma, T, N)

    # Plot
    plt.plot(t, S, lw=2, label=symbol)

# List of stock symbols
symbols = ['AAPL', 'MSFT', 'GOOG']

# Plot GBM for each symbol
plt.figure(figsize=(10, 6))
for symbol in symbols:
    plot_stock_gbm(symbol)

plt.title('Geometric Brownian Motion for Stocks')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid(True)
plt.legend()
plt.show()

