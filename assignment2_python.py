import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import minimize
from scipy import optimize
from scipy.optimize import Bounds
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg

## df=pd.read_csv('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64')
df=pd.read_csv('C:\\Users\\Daniel192\\Downloads\\current.csv')

df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format= '%m/%d/%Y')

df_cleaned['INDPRO_log'] = np.log(df_cleaned['INDPRO'])
df_cleaned['INDPRO_diff'] = df_cleaned['INDPRO_log'].diff()

df_cleaned = df_cleaned.dropna(subset=['INDPRO_diff'])
df_cleaned.reset_index(drop=True, inplace=True)
print(df_cleaned['INDPRO_diff'].head())

data = df_cleaned['INDPRO_diff']
p = 2

mean_value = df_cleaned['INDPRO_diff'].mean()
print("Media della colonna INDPRO_diff:", mean_value)

## UNCONDITIONAL
def ar_likelihood(params, data, p):
    ##Calculate the negative (unconditional) log likelihood for an AR(p) model.
    # Extract AR coefficients and noise variance
    c = params[0]
    phi = params[1:p+1]
    sigma2 = params[-1]
        
    # Calculate residuals
    T = len(data)
    residuals = data[p:] - c - np.dot(np.column_stack([data[p-j-1:T-j-1] for j in range(p)]), phi)
    
    # Calculate negative log likelihood
    log_likelihood = (-T/2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2))
    
    return -log_likelihood

def estimate_ar_parameters(data, p):

    # Initial parameter guess (random AR coefficients, variance of 1)
    params_initial = np.zeros(p+2)
    params_initial[-1] = 1.0

    ## Bounds
    bounds = [(None, None)]
    # Then p AR coefficients, each bounded between -1 and 1
    bounds += [(-1, 1) for _ in range(p)]
    # The variance parameter, bounded to be positive
    bounds += [(1e-6, None)]

    # Minimize the negative log likelihood
    result = minimize(ar_likelihood, params_initial, args=(data, p), bounds=bounds)
    
    if result.success:
        estimated_params = result.x
        return estimated_params
    else:
        raise Exception("Optimization failed:", result.message)

 
paramsUC = estimate_ar_parameters(data, p)
print("Estimated parameters:", paramsUC)
print("Estimated log likelihood:", -ar_likelihood(paramsUC, data, p))

 ## auto 
modell = sm.tsa.ARIMA(data, order=(2, 0, 0))
results = modell.fit()
print(results.summary())

## no ...

## UNCONDITIONAL 
def fit_ar_ols_xx(data, p):
    T = len(data)
    Y = data[p:]  
    X = np.column_stack([data[p-i-1:T-i-1] for i in range(p)])
    X = np.column_stack((np.ones(X.shape[0]), X))
    
    XTX = np.dot(X.T, X)  
    XTY = np.dot(X.T, Y)  
    beta_hat = np.linalg.solve(XTX, XTY) 
    
    return beta_hat

beta_hat = fit_ar_ols_xx(data, p)
print("Estimated AR coefficients:", beta_hat)

newX= pd.DataFrame({
    'newX_t-1': data.shift(1),
    'newX_t-2': data.shift(2),
})
newX = newX.dropna()
newdata= data[2:]
newX = sm.add_constant(newX)

print("Indici di y:", newdata.index)
print("Indici di X:", newX.index)
modelll = sm.OLS(newdata, newX)
results = modelll.fit()
print(results.summary())
##### tutto ok 

### CONDITIONAL
def ar2_exact_loglikelihood(paramsCD, y):
    """
    Calculate the exact log-likelihood for an AR(2) model.
    
    Parameters:
    -----------
    y : array-like
        data (T x 1)
    params : tuple or list
        Model parameters (c, phi1, phi2, sigma2)
        c: constant term
        phi1: coefficient of y_{t-1}
        phi2: coefficient of y_{t-2}
        sigma2: error variance
    
    Returns:
    --------
    float
        Exact log-likelihood value
    """
    # Extract parameters
    c, phi1, phi2, sigma2 = paramsCD
    
    # Check stationarity conditions
    if not (phi2 > -1 and phi1 + phi2 < 1 and phi2 - phi1 < 1):
        return -np.inf  # Return negative infinity if not stationary
    
    T = len(y)
    
    if T < 3:
        raise ValueError("Time series must have at least 3 observations for AR(2)")
    
    # Calculate the unconditional mean of the process
    mu = c / (1 - phi1 - phi2)
    
    # Calculate autocovariances for stationary process
    gamma0 = sigma2 / (1 - phi2**2 - phi1**2)  # Variance
    gamma1 = phi1 * gamma0 / (1 - phi2)        # First-order autocovariance
    
    # Create initial variance-covariance matrix
    Sigma0 = np.array([[gamma0, gamma1], 
                        [gamma1, gamma0]])
    
    # Calculate determinant of Sigma0
    det_Sigma0 = gamma0**2 - gamma1**2
    
    # Calculate inverse of Sigma0
    if det_Sigma0 <= 0:  # Check for positive definiteness
        return -np.inf
    
    inv_Sigma0 = np.array([[gamma0, -gamma1], 
                            [-gamma1, gamma0]]) / det_Sigma0
    
    # Initial distribution contribution (Y1, Y2)
    y_init = np.array([y[0], y[1]])
    mu_init = np.array([mu, mu])
    
    diff_init = y_init - mu_init
    quad_form_init = diff_init.T @ inv_Sigma0 @ diff_init
    
    loglik_init = -np.log(2 * np.pi * np.sqrt(det_Sigma0)) - 0.5 * quad_form_init
    
    # Conditional log-likelihood contribution (Y3, ..., YT | Y1, Y2)
    residuals = np.zeros(T-2)
    for t in range(2, T):
        y_pred = c + phi1 * y[t-1] + phi2 * y[t-2]
        residuals[t-2] = y[t] - y_pred
    
    loglik_cond = -0.5 * (T-2) * np.log(2 * np.pi * sigma2) - \
                   0.5 * np.sum(residuals**2) / sigma2
    
    # Total exact log-likelihood
    exact_loglik = loglik_init + loglik_cond
    
    ## Return the negative loglik
    return -exact_loglik

def fit_ar2_mle(y, initial_params=None):
    """
    Fit an AR(2) model using maximum likelihood estimation
    
    Parameters:
    -----------
    y : array-like
        Time series data
    initial_params : tuple, optional
        Initial guess for (c, phi1, phi2, sigma2)
    """
    # Set default initial parameters if not provided
    if initial_params is None:
      # Simple initial estimates
      c_init = 0.0
      phi1_init = 0
      phi2_init = 0
      sigma2_init = np.var(y)
        
      initial_params = (c_init, phi1_init, phi2_init, sigma2_init)
      # Constraints to ensure positive variance
    
    lbnds = (-np.inf, -0.99, -0.99, 1e-6)  # Lower bounds for params
    ubnds = (np.inf, 0.99, 0.99, np.inf)     # Upper bounds for params

    bnds = optimize.Bounds(lb=lbnds, ub=ubnds)
    # Optimize
    result = optimize.minimize(
        ar2_exact_loglikelihood, 
        initial_params,
        args = (y,),
        bounds = bnds,
        method='L-BFGS-B', 
        options={'disp': False} # set to true to get more info
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. {result.message}")
    
    # Return parameters and maximum log-likelihood
    return result.x, result.fun

exact_lik= ar2_exact_loglikelihood((0.0, 0.0, 0.0, 1.0), data)
print(exact_lik)
y = data

paramsCD_= fit_ar2_mle(y, initial_params=None)
paramsCD = paramsCD_[0]
print("Estimated parameters:", paramsCD)
print("Estimated parameters[1]:", paramsCD[1])

model = AutoReg(data, lags=p)
model_fitted = model.fit()
print(model_fitted.summary())

## no...

print(len(data))
print("paramsCD:", paramsCD)
print("Lunghezza di paramsCD:", len(paramsCD))
print("paramsUC:", paramsUC)
print("Lunghezza di paramsUC:", len(paramsUC))



### FORECASTING




##forecast log differences of indpro for next 8 months h=1,2....8
def forecastAR2(data, paramsUC, paramsCD, n, i):
    yhat_UC = []
    yhat_CD = []
    
    # Ultimi due valori osservati per UC
    y_prev1_UC = data[i]
    y_prev2_UC = data[i-1]
    
    # Ultimi due valori osservati per CD
    y_prev1_CD = data[i]
    y_prev2_CD = data[i-1]
    
    for h in range(n):
        # Previsione con il modello UC
        y_next_UC = paramsUC[0] + paramsUC[1] * y_prev1_UC + paramsUC[2] * y_prev2_UC
        yhat_UC.append(y_next_UC)
        
        # Aggiorna i valori per il prossimo passo (per UC)
        y_prev2_UC = y_prev1_UC
        y_prev1_UC = y_next_UC
        
        # Previsione con il modello CD
        y_next_CD = paramsCD[0] + paramsCD[1] * y_prev1_CD + paramsCD[2] * y_prev2_CD
        yhat_CD.append(y_next_CD)
        
        # Aggiorna i valori per il prossimo passo (per CD)
        y_prev2_CD = y_prev1_CD
        y_prev1_CD = y_next_CD

    return yhat_UC, yhat_CD

n = 8
i = 791  # Ultimo punto disponibile

yhat_UC, yhat_CD = forecastAR2(data, paramsUC, paramsCD, n, i)
yhat_UC = [float(value) for value in yhat_UC]
yhat_CD = [float(value) for value in yhat_CD]
print(yhat_UC)
print(yhat_CD)