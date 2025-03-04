import pandas as pd
from numpy.linalg import solve
import numpy as np

# Load the dataset
df = pd.read_csv('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64')


# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format= '%m/%d/%Y')
## df_cleaned contains the data cleaned

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

## transformation_codes contains the transformation codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$



# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))


df_cleaned=df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head()

############################################################################################################
## Plot transformed series
############################################################################################################
import matplotlib.pyplot as plt         
import matplotlib.dates as mdates       

series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']         
series_names = ['Industrial Production',                 
                'Inflation (CPI)',                        
                '3-month Treasury Bill rate']            


# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))       

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:                                
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y') 
        ax.plot(dates, df_cleaned[series_name], label=plot_title)        
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))           
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))         
        ax.set_title(plot_title)                                         
        ax.set_xlabel('Year')                                            
        ax.set_ylabel('Transformed Value')                               
        ax.legend(loc='upper left')                                      
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right') 
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout() 
plt.show()         

############################################################################################################
## Create y and X for estimation of parameters
############################################################################################################

Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

## Number of lags and leads
num_lags  = 4  ## this is p
num_leads = 1  ## this is h

X = pd.DataFrame()
## Add the lagged values of Y
col = 'INDPRO'
for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Yraw.shift(lag)
## Add the lagged values of X
for col in Xraw.columns:
    for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
## Add a column on ones (for the intercept)
X.insert(0, 'Ones', np.ones(len(X)))

## X is now a DataFrame with lagged values of Y and X
X.head()

## Y is now the leaded target variable
y = Yraw.shift(-num_leads)


############################################################################################################
## Estimation and forecast
############################################################################################################

## Save last row of X (converted to numpy)
X_T = X.iloc[-1:].values

## Subset getting only rows of X and y from p+1 to h-1
## and convert to numpy array
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

## Import the solve function from numpy.linalg
from numpy.linalg import solve

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X.T @ X, X.T @ y)

## Produce the One step ahead forecast
## % change month-to-month of INDPRO
forecast = X_T@beta_ols*100

print(forecast)

###############################################
## NOW FORECAST ASSIGNMENT
def calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999',target = 'INDPRO', xvars = ['CPIAUCSL', 'TB3MS']):

    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)
        
    Yraw = rt_df[target]
    Xraw = rt_df[xvars]
    X = pd.DataFrame()
    
    for lag in range(0,p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)

    for col in Xraw.columns:
        for lag in range(0,p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    
    X.insert(0, 'Ones', np.ones(len(X)))
    X_T = X.iloc[-1:].values

    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h) 
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        Yhat.append(X_T@beta_ols*100)

    return np.array(Y_actual) - np.array(Yhat)

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = t0)
    e.append(ehat.flatten())
    T.append(t0)


edf = pd.DataFrame(e)

RMSFE1 = np.sqrt(edf.apply(np.square).mean())

print(RMSFE1)  ##first model, now different variables

Yraw2 = df_cleaned['INDPRO']
Xraw2 = df_cleaned[['S&P 500', 'INVEST']]

X2 = pd.DataFrame()
col = 'INDPRO'
for lag in range(0,num_lags+1):
        X2[f'{col}_lag{lag}'] = Yraw2.shift(lag)
for col in Xraw2.columns:
    for lag in range(0,num_lags+1):
        X2[f'{col}_lag{lag}'] = Xraw2[col].shift(lag)

X2.insert(0, 'Ones', np.ones(len(X2)))
X2_T = X2.iloc[-1:].values
X2 = X2.iloc[num_lags:-num_leads].values
forecast2 = X2_T@beta_ols*100

def calculate_forecast2(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999', target2 = 'INDPRO', xvars2 = ['S&P 500', 'INVEST']):
    rt_df2 = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    
    Y2_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y2_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target2]*100)
      ##if os in df_cleaned['sasdate'].values:
      ##      Y2_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target2].values[0] * 100)
       ## else:
       ##     Y2_actual.append(np.nan)  


    Yraw2 = rt_df2[target2]
    Xraw2 = rt_df2[xvars2]
    X2 = pd.DataFrame()
    
    for lag in range(0, p):
        X2[f'{target2}_lag{lag}'] = Yraw2.shift(lag)

    for col in Xraw2.columns:
        for lag in range(0, p):
            X2[f'{col}_lag{lag}'] = Xraw2[col].shift(lag)
    
    X2.insert(0, 'Ones', np.ones(len(X2)))
    X2_T = X2.iloc[-1:].values

    Yhat2 = []
    for h in H:
        y2_h = Yraw2.shift(-h) 
        y2 = y2_h.iloc[p:-h].values
        X2_ = X2.iloc[p:-h].values
        ##if X2_.shape[0] != y2.shape[0]:
        ##raise ValueError(f"Dimension mismatch: X2_ has {X2_.shape[0]} rows, but y2 has {y2.shape[0]} rows")
        beta_ols2 = solve(X2_.T @ X2_, X2_.T @ y2)
        Yhat2.append(X2_T @ beta_ols2 * 100)

    return np.array(Y2_actual) - np.array(Yhat2)

t0 = pd.Timestamp('12/1/1999')
e2 = []
T2 = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat2 = calculate_forecast2(df_cleaned, p = 4, H = [1,4,8], end_date = t0)
    e2.append(ehat2.flatten())
    T2.append(t0)

edf2 = pd.DataFrame(e2)

RMSFE2 = np.sqrt(edf2.apply(np.square).mean())
print(RMSFE2)

### with such 2 covariates, forecast is worse comparing RMSFE !
