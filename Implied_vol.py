"""
Calculates implied volatility using BSM.
To solve the equation, use Newton's method: 
   approximation to x: x_n+1 = x_n - f(x_n)/f'(x_n)
   in this case, f: C(S0, K, r, T, sigma_imp) - C* = 0
   therefore, sigma_imp_n+1 = sigma_imp_n - (C(S0, K, r, T, sigma_imp_n) - C*)/vega_imp_n
"""

def bsm_call_value(S0, K, r, T, sigma):
   from math import log, sqrt, exp
   from scipy import stats

   S0 = float(S0)
   d1 = (log(S0/K)+(r + 1/2 * sigma**2) * T) / (sigma * sqrt(T))
   d2 = (log(S0/K)+(r - 1/2 * sigma**2) * T) / (sigma * sqrt(T))
   value = S0 * stats.norm.cdf(d1) - exp(-r*T) * K * stats.norm.cdf(d2)
   return value

def bsm_vega(S0, K, r, T, sigma):
   from math import log, sqrt
   from scipy import stats

   S0 = float(S0)
   d1 = (log(S0/K)+(r + 1/2 * sigma**2) * T) / (sigma * sqrt(T))
   vega = S0 * sqrt(T) * stats.norm.cdf(d1)
   return vega

def imp_vol(S0, K, r, T, C_mkt, sigma_est, it):
   for i in range(it):
      sigma_est = sigma_est - (bsm_call_value(S0, K, r, T, sigma_est) - C_mkt) / bsm_vega(S0, K, r, T, sigma_est)
   return sigma_est

V0 = 17.6639
r = 0.01
import pandas as pd
h5 = pd.HDFStore('/Users/cassie/Documents/Projects/Python-for-Finance/data/vstoxx_data_31032014.h5', 'r')
futures_data = h5['futures_data']
options_data = h5['options_data']
h5.close()

options_data['IMP_VOL'] = 0.0

tol = 0.5
for option in options_data.index:
   forward = futures_data[futures_data['MATURITY'] == options_data.loc[option]['MATURITY']]['PRICE'].values[0]
   print(forward)
   print(options_data.loc[option]['STRIKE'])
   if forward * (1 - tol) < options_data.loc[option]['STRIKE'] < forward * (1 + tol):
      imp = imp_vol(V0, 
      options_data.loc[option]['STRIKE'],
      r,
      options_data.loc[option]['TTM'],
      options_data.loc[option]['PRICE'],
      sigma_est=2.,
      it=100
      )
      options_data.loc[option]['IMP_VOL'] = imp

plot_data = options_data[options_data['IMP_VOL'] > 0]
maturities = sorted(set(options_data['MATURITY']))

'''import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for maturity in maturities:
    data = plot_data[options_data.MATURITY == maturity]
      # select data for this maturity
    plt.plot(data['STRIKE'], data['IMP_VOL'],
             label=maturity.date(), lw=1.5)
    plt.plot(data['STRIKE'], data['IMP_VOL'], 'r.')
plt.grid(True)
plt.xlabel('strike')
plt.ylabel('implied volatility of volatility')
plt.legend()
plt.show()
'''


