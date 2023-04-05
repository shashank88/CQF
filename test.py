# CQF practice code for exam
# Import pandas & yfinance
import pandas as pd

# Import numpy
import numpy as np
from numpy import *
from numpy.linalg import multi_dot

# Import cufflinks
import cufflinks as cf
cf.set_config_file(offline=True, dimensions=((1000,600))) # theme= 'henanigans'

# Import plotly express for EF plot
import plotly.express as px
# px.defaults.template = "plotly_dark"
px.defaults.width, px.defaults.height = 1000, 600


def gen_allocation(allocation_size):
    # Generate 4 random numbers b/w 0 and 1
    rands = np.random.random(4)
    # Standardize by dividing by their sum
    std_rands = rands / sum(rands)
    return std_rands
    
def gen_allocations(n_simulations, allocation_size):
    return (gen_allocation(allocation_size) for _ in range(n_simulations))
def calculate_porfolio(n_simulations=700):
    mu = np.array([0.02, 0.07, 0.15, 0.20])
    sigma = np.array([0.05, 0.12, 0.17, 0.25])
    corr = np.array([[1, 0.3, 0.3, 0.3], [0.3, 1, 0.6, 0.6], [0.3, 0.6, 1, 0.6], [0.3, 0.6, 0.6, 1]])
    results = []
    for weight in gen_allocations(n_simulations=n_simulations, allocation_size=len(mu)):
        p_mu = np.dot(weight.T, mu)
        p_sigma = np.sqrt(np.dot(np.dot(weight.T, corr), weight))
        results.append((p_mu, p_sigma))
    
    return results
  
r = calculate_porfolio(700)
px.scatter(x=[res[1] for res in r], y=[res[0] for res in r])

----------
Q3

strike = 100
spot = 100
rate = 0.05
time = 1  # year  

np.linspace(0.05, 0.80)

sigma = 0.20
for time_steps in range(4, 51):
    v = binomial_option(spot, strike, rate, sigma, time, time_steps, 2)
    print(f'value with time step={time} is {option_val}')
v[0][0]

np.arange(0.05, 0.80, 0.05)
sigmas = []
vals = []
for sigma in np.arange(0.05, 0.8, 0.05):
    option_val = binomial_option(spot, strike, rate, sigma, time, 4, 2)[0][0]
    print(f'value with sigma={sigma} is {option_val}')
    sigmas.append(sigma)
    vals.append(option_val)
    
    px.scatter(sigmas, vals)    
    
    ------
    
    # Create a user defined function
def binomial_option(spot, strike, rate, sigma, time, steps, output=0):
    
    """
    binomial_option(spot, strike, rate, sigma, time, steps, output=0)
    
    Function to calculate binomial option pricing for european call option
    
    Params
    ------
    spot       -int or float    - spot price
    strike     -int or float    - strike price
    rate       -float           - interest rate
    time       -int or float    - expiration time
    steps      -int             - number of time steps
    output     -int             - [0: price, 1: payoff, 2: option value, 3: option delta]
    
    Returns
    --------
    out: ndarray
    An array object of price, payoff, option value and delta as specified by the output flag
    
    """
    
    # define parameters
    ts = time / steps
    u  = 1 + sigma*sqrt(ts) 
    v  = 1 - sigma*sqrt(ts)
    p  = 0.5 + rate *sqrt(ts) / (2*sigma)
    df = 1/(1+rate*ts)
    
    # initialize the arrays
    px = zeros((steps+1, steps+1))
    cp = zeros((steps+1, steps+1))
    V = zeros((steps+1, steps+1))
    d = zeros((steps+1, steps+1))
    
    # binomial loop : forward loop
    for j in range(steps+1):
        for i in range(j+1):
            px[i,j] = spot * power(v,i) * power(u,j-i)
            cp[i,j] = maximum(px[i,j] - strike, 0)
         
    # reverse loop
    for j in range(steps+1, 0, -1):
        for i in range(j):
            if (j==steps+1):
                V[i,j-1] = cp[i,j-1]
                d[i,j-1] = 0 
            else:
                V[i,j-1] = df*(p*V[i,j]+(1-p)*V[i+1,j])
                d[i,j-1] = (V[i,j]-V[i+1,j])/(px[i,j]-px[i+1,j])
    
    results = around(px,2), around(cp,2), around(V,2), around(d,4)

    return results[output]
  
  ----------
  Q4
  percentiles = [99.95, 99.75, 99.5, 99.25,99, 98.5, 98, 97.5]
  import scipy.stats as st
[st.norm.ppf(1 - p/100) for p in percentiles]
from tabulate import tabulate

es_arr = []

for p in percentiles:
    es_x = 0 - 1 * (st.norm.ppf(1 - p/100) / (1 - p/100))
#     print(f'Expected shortfall: {es_x}, percentile: {p}')
    es_arr.append(es_x)

ctable = [[str(percentiles[idx]), v] for idx, v in enumerate(es_arr)]
cheader = ['Confidence Level', 'Conditional Value At Risk']
print(tabulate(ctable, headers=cheader))
---
Q5

df = pd.read_csv('Data_SP500.csv', index_col='Date', parse_dates=True)
df.head()

df['log_price'] = np.log(df['SP500'])

df['log_returns'] = df['log_price'].diff()


std = df['log_returns'].std()
std
std_10 = std * sqrt(10)

# number of stdev from the mean
import scipy.stats as st
var = st.norm.ppf(1 - 0.99, df['log_returns'].mean(), std_10)

df

def test_func(mini_df):
    
df['rolling_std'] = df[['log_returns']].rolling(window=21).std()
df['rolling_var'] = df

# breach = lambda x: np.log(x[-1] / x[0])
glob_breach = []
def breach(mini_df):
    fwd_ret = np.log(mini_df[-1] / mini_df[0])
    if fwd_ret < 0 and fwd_ret < var:
        glob_breach.append(1)
#         print('breach')
    else:
        glob_breach.append(0)
#         print('no breach')
    return fwd_ret

df[['SP500']].rolling(window=10).apply(breach)
# print(glob_breach)
bs = [g for g in glob_breach if g == 1]
print(f'total breach: {len(bs)} / {len(glob_breach)}, percentage: {len(bs) / len(glob_breach) * 100}')

n_consecutive = 0
for idx, b in enumerate(glob_breach):
    if idx == 0:
        continue
    if glob_breach[idx - 1] == b == 1:
        n_consecutive += 1

print(f'consecutive breach: {n_consecutive} / {len(glob_breach)}, percentage: {n_consecutive / len(glob_breach) * 100}')

------
    
lambda_calc = 0.72
initial_variance = std_10 ** 2

df.head()

def breach(mini_df):
    fwd_ret = np.log(mini_df['SP500'][-1] / mini_df['SP500'][0])
    var = 
    if fwd_ret < 0 and fwd_ret < mini_df['']:
        glob_breach.append(1)
#         print('breach')
    else:
        glob_breach.append(0)
#         print('no breach')
    return fwd_ret

# # breach = lambda x: np.log(x[-1] / x[0])
# glob_breach = []
# last_var = initial_variance
# def calc(mini_df):
#     next_var = lambda_calc * last_var + (1 - lambda_calc)
    

# def breach(mini_df):
#     fwd_ret = np.log(mini_df[-1] / mini_df[0])
#     if fwd_ret < 0 and fwd_ret < var:
#         glob_breach.append(1)
# #         print('breach')
#     else:
#         glob_breach.append(0)
# #         print('no breach')
#     return fwd_ret

# # df['rolling_variance'] = df[['SP500']].rolling(window=10).apply(calc)
# # print(glob_breach)
# df['rolling_var'] = d
# df[['SP500']].rolling(window=10).apply(breach)
# bs = [g for g in glob_breach if g == 1]
# print(f'total breach: {len(bs)} / {len(glob_breach)}, percentage: {len(bs) / len(glob_breach) * 100}')

# n_consecutive = 0
# for idx, b in enumerate(glob_breach):
#     if idx == 0:
#         continue
#     if glob_breach[idx - 1] == b == 1:
#         n_consecutive += 1

# print(f'consecutive breach: {n_consecutive} / {len(glob_breach)}, percentage: {n_consecutive / len(glob_breach) * 100}')


# df['rolling_var'] = df['rolling_var'].shift(-1) * lambda_calc + (1 - lambda_calc) * df['log_returns']

df['log_returns'] * (1 - lambda_calc ) + lambda_calc * df['rolling_var'].shift(1)

lambda_calc

log_returns = list(df['log_returns'])

rolling_vars = [initial_variance]
for idx, ret in enumerate(log_returns[1:]):
    new_var = rolling_vars[-1] * lambda_calc + (1 - lambda_calc) * ret
    rolling_vars.append(new_var)

df['rolling_vars'] = rolling_vars

# df = df.drop(columns='rolling_var')

df

# rolling_vars = [initial_variance]
# for idx in range(1, len(df)):
#     prev = df.iloc[idx - 1]
#     row = df.iloc[idx]
#     print(f"prev rolling var: {rolling_vars[-1]}, {prev['log_returns']}, {rolling_vars[-1] * lambda_calc + (1 - lambda_calc) * prev['log_returns']}")
#     rolling_vars.append(rolling_vars[-1] * lambda_calc + (1 - lambda_calc) * prev['log_returns'])
    
# #     df.iloc[idx, 'rolling_var'] = df.loc[idx - 1, 'rolling_var'] * lambda_calc + (1 - lambda_calc) * df.loc[idx - 1, 'log_returns']
