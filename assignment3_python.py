#####  ASSIGNMENT 3 #########

##Apply Monte Carlo simulations combined with bootstrap methods to evaluate the quality of inference on B1
​ ## using serially correlated data.

##1 Simulate data according to simulate_regression_with_ar1_errors.

##2 Calculate bootstrap standard errors.

##3 Construct a 95% confidence interval for 
​ ## using both the bootstrap and the theoretical standard errors.

##4 Perform Monte Carlo simulations for T=100 and T=500, and assess the empirical coverage of the confidence intervals.



import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
np.random.seed(0)

def simulate_ar1(n, phi, sigma):
    errors = np.zeros(n)
    eta = np.random.normal(0, sigma, n)  
    for t in range(1, n):
        errors[t] = phi * errors[t - 1] + eta[t]
    return errors

def simulate_regression_with_ar1_errors(n, beta0, beta1, phi_x, phi_u, sigma):
    x = simulate_ar1(n, phi_x, sigma)
    u = simulate_ar1(n, phi_u, sigma)
    y = beta0 + beta1 * x + u
    return x, y, u

def moving_block_bootstrap(x, y, block_length, num_bootstrap):
    T = len(y)  # Total number of observations
    num_blocks = T // block_length + (1 if T % block_length else 0)

    # Fit the original model
    X = sm.add_constant(x)
    original_model = sm.OLS(y, X)
    original_results = original_model.fit()

    bootstrap_estimates = np.zeros((num_bootstrap, 2))  # Storing estimates for beta_0 and beta_1

    # Perform the bootstrap
    for i in range(num_bootstrap):
        # Create bootstrap sample
        bootstrap_indices = np.random.choice(np.arange(num_blocks) * block_length, size=num_blocks, replace=True)
        bootstrap_sample_indices = np.hstack([np.arange(index, min(index + block_length, T)) for index in bootstrap_indices])
        bootstrap_sample_indices = bootstrap_sample_indices[:T]  # Ensure the bootstrap sample is the same size as the original data

        x_bootstrap = x[bootstrap_sample_indices]
        y_bootstrap = y[bootstrap_sample_indices]

        # Refit the model on bootstrap sample
        X_bootstrap = sm.add_constant(x_bootstrap)
        bootstrap_model = sm.OLS(y_bootstrap, X_bootstrap)
        bootstrap_results = bootstrap_model.fit()

        # Store the estimates
        bootstrap_estimates[i, :] = bootstrap_results.params

    return bootstrap_estimates

# Parameters
T = 500              
beta0 = 1.          
beta1 = 2           
phi_x = 0.7             
phi_u = 0.7             
sigma = 1            

# Generate data
x, y, errors = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)

# Set parameters for the moving block bootstrap
block_length = 12
num_bootstrap = 1000

# Run moving block bootstrap
bootstrap_results = moving_block_bootstrap(x, y, block_length, num_bootstrap)

# Calculate and print standard errors
bootstrap_standard_errors = bootstrap_results.std(axis=0)
print("Bootstrap Standard Errors:")
print("SE(beta_0):", bootstrap_standard_errors[0])
print("SE(beta_1):", bootstrap_standard_errors[1])

# Fit the linear model using statsmodels
X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()

# Standard errors from statsmodels
statsmodels_se = results.bse
print("Standard Errors from statsmodels OLS:")
print("SE(beta_0):", statsmodels_se[0])
print("SE(beta_1):", statsmodels_se[1])

# Confidence intervals
beta1_hat = results.params[1]
se_theoretical = results.bse[1]

ci_theoretical = (
    float(beta1_hat - 1.96 * se_theoretical),
    float(beta1_hat + 1.96 * se_theoretical)
)

se_bootstrap_beta1 = bootstrap_standard_errors[1]
ci_bootstrap = (
    float(beta1_hat - 1.96 * se_bootstrap_beta1),
    float(beta1_hat + 1.96 * se_bootstrap_beta1)
)

print("Intervallo di confidenza 95% per beta1 (teorico):", ci_theoretical)
print("Intervallo di confidenza 95% per beta1 (bootstrap):", ci_bootstrap)










