#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 6 2022

@author: Patricia de Bruin
"""

import os
import time

import arviz as az
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import nbinom

az.style.use(['arviz-darkgrid', 'arviz-colors'])

# set directory
os.chdir('/Users/patricia/Documents/Mathematical Sciences/Research Thesis/Artikel/Python')
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
time_str = time.strftime('%Y%m%d-%H%M%S')

n = 200 # number of households in the population of interest
r = 60 # months at risk

r_data = np.empty(n)
r_data.fill(r) # vector of months at risk

# parameters
rho0 = 1 # amount of heterogeneity between households
mu0 = 0.05 # overall probability of a fire incident per month 

# simulate data set
x_data = np.random.negative_binomial(n = rho0, p = 1 / (1 + r * mu0 / rho0), size = n) # vector of the number of incidents

#save simulated data set
df = pd.DataFrame({'incidents': x_data, 'time': r_data})
df.to_excel(os.path.join(output_dir, f'simulated incidents and time {time_str}.xlsx'))
print(df)

niter = 1000
chains = 4

with pm.Model() as mixed_poisson:
    # define priors
    rho = pm.Gamma('rho', alpha = 1, beta = 1)
    mu = pm.Beta('mu', alpha = 0.01, beta = 1)
    
    # model heterogeneity
    lamb = pm.Gamma('lamb', alpha = rho, beta = rho / mu, shape = n)
    
    # observations
    x_obs = pm.Poisson('x_obs', mu = r_data * lamb, observed = x_data)
    
    # model specifications
    step = pm.NUTS()

with mixed_poisson:
    # draw posterior samples
    trace = pm.sample(niter, step = step, return_inferencedata = False, chains = chains, cores = 1)
    
    # specifications plots
    var_names = ['rho', 'mu'] 
    lines = (('rho', {}, np.mean(trace['rho'])), ('mu', {}, np.mean(trace['mu'])))
    
    # traceplot of the parameters
    pm.plot_trace(trace, figsize = (12,8), var_names = ['rho', 'mu'], lines = lines, compact = False, legend = True)
    plt.savefig(os.path.join(output_dir, f'traceplot rho and mu MCMC {time_str}.pdf'), format = 'pdf', dpi = 600)
    
# summary of parameters
with mixed_poisson:
    summary = pm.summary(trace, round_to = 10, kind = 'all')
    print(summary)

# show the first two plots
plt.show()

# estimated MCMC parameters
rho_MCMC = np.mean(trace['rho'])
mu_MCMC = np.mean(trace['mu'])

# estimated MOM parameters
rho_MOM = (np.mean(x_data) / np.mean(r_data)) ** 2 / (np.mean(np.power(x_data, 2)) / np.mean(np.power(r_data, 2)) - (np.mean(x_data) / np.mean(r_data)) ** 2 - np.mean(x_data) / np.mean(np.power(r_data, 2)))
mu_MOM = np.mean(x_data) / np.mean(r_data)

# save outcome 
df2 = pd.DataFrame({'rho': rho0, 'mu': mu0, 'rho_MOM': rho_MOM, 'mu_MOM': mu_MOM, 'rho_MCMC': rho_MCMC, 'mu_MCMC': mu_MCMC}, index = [0])
df2.to_excel(os.path.join(output_dir, f'outcome parameter estimation {time_str}.xlsx'))
print(df2) 

# plot probability mass function negative binomial with true parameters
x = np.arange(16)
P0 = nbinom.pmf(x, rho0, 1 / (1 + r * mu0 / rho0))

# plot probability mass function negative binomial with estimated MCMC parameters
P_MCMC = nbinom.pmf(x, rho_MCMC, 1 / (1 + r * mu_MCMC / rho_MCMC))

# plot probability mass function negative binomial with estimated MOM parameters
P_MOM = nbinom.pmf(x, rho_MOM, 1 / (1 + r * mu_MOM / rho_MOM))

# probability mass function
width = 0.25
fig, ax = plt.subplots()
ax1 = ax.bar(x - width, P0, width, label = 'true pdf')
ax2 = ax.bar(x, P_MCMC, width, label = 'pdf with MCMC estimates')
ax3 = ax.bar(x + width, P_MOM, width, label = 'pdf with moment estimates')
plt.xlabel("$n$")
plt.ylabel("$P(N(t) = n)$")
plt.xticks(x)
plt.xlim(-0.75, 15.75)
plt.legend(loc = 'upper right', prop = {'size': 8})
plt.savefig(os.path.join(output_dir, f'probability mass function incidents {time_str}.pdf'), format = 'pdf', dpi = 600)
