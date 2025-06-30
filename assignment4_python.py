### Given code (Edit start at line 250) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams['figure.figsize'] = (10, 6)

n_samples = 200        
n_features = 50        
n_informative = 10      
noise_level = 1.0   

X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
print(f"True informative features indices: {sorted(informative_features)}")

for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3

    Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level
    data_dict = {
    'X': X,
    'Y': Y,
    'true_coefficients': true_coefficients,
    'informative_features': informative_features
}

coef_df = pd.DataFrame({
    'feature_index': range(n_features),
    'true_coefficient': true_coefficients
})

print("\nNon-zero coefficients:")
print(coef_df[coef_df['true_coefficient'] != 0])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

lasso_results = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, Y_train)
    
    Y_train_pred = lasso.predict(X_train_scaled)
    Y_test_pred = lasso.predict(X_test_scaled)
    
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    lasso_results[alpha] = {
        'model': lasso,
        'coefficients': lasso.coef_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_nonzero_coef': n_nonzero
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

ridge_results = {}

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, Y_train)
    
    Y_train_pred = ridge.predict(X_train_scaled)
    Y_test_pred = ridge.predict(X_test_scaled)
    
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    
    threshold = 0.001
    n_small = np.sum(np.abs(ridge.coef_) < threshold)
    
    ridge_results[alpha] = {
        'model': ridge,
        'coefficients': ridge.coef_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_small_coef': n_small
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Coefficients < {threshold}: {n_small}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

selected_alpha = 0.1


lasso_coef = lasso_results[selected_alpha]['coefficients']
ridge_coef = ridge_results[selected_alpha]['coefficients']


fig, axes = plt.subplots(2, 2, figsize=(15, 12))


ax1 = axes[0, 0]
ax1.scatter(true_coefficients, lasso_coef, alpha=0.6)
ax1.plot([-5, 5], [-5, 5], 'r--', label='Perfect recovery')
ax1.set_xlabel('True Coefficients')
ax1.set_ylabel('Lasso Coefficients')
ax1.set_title(f'Lasso Coefficient Recovery (α={selected_alpha})')
ax1.legend()
ax1.grid(True, alpha=0.3)


ax2 = axes[0, 1]
ax2.scatter(true_coefficients, ridge_coef, alpha=0.6)
ax2.plot([-5, 5], [-5, 5], 'r--', label='Perfect recovery')
ax2.set_xlabel('True Coefficients')
ax2.set_ylabel('Ridge Coefficients')
ax2.set_title(f'Ridge Coefficient Recovery (α={selected_alpha})')
ax2.legend()
ax2.grid(True, alpha=0.3)


ax3 = axes[1, 0]
for idx in informative_features:
    coef_path = [lasso_results[alpha]['coefficients'][idx] for alpha in alphas]
    ax3.plot(alphas, coef_path, 'b-', linewidth=2, alpha=0.8)

for idx in range(n_features):
    if idx not in informative_features:
        coef_path = [lasso_results[alpha]['coefficients'][idx] for alpha in alphas]
        ax3.plot(alphas, coef_path, 'gray', linewidth=0.5, alpha=0.3)
ax3.set_xscale('log')
ax3.set_xlabel('Alpha (log scale)')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Lasso Coefficient Path')
ax3.grid(True, alpha=0.3)


ax4 = axes[1, 1]
nonzero_counts = [lasso_results[alpha]['n_nonzero_coef'] for alpha in alphas]
ax4.plot(alphas, nonzero_counts, 'o-', linewidth=2, markersize=8)
ax4.axhline(y=n_informative, color='r', linestyle='--', 
            label=f'True number ({n_informative})')
ax4.set_xscale('log')
ax4.set_xlabel('Alpha (log scale)')
ax4.set_ylabel('Number of Non-zero Coefficients')
ax4.set_title('Sparsity vs Regularization Strength')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

from sklearn.linear_model import LassoCV

alphas_cv = np.linspace(0.0001, 0.3, 50)

lasso_cv = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, Y_train)

optimal_alpha = lasso_cv.alpha_
print(f"Optimal alpha from cross-validation: {optimal_alpha:.4f}")

Y_test_pred_cv = lasso_cv.predict(X_test_scaled)
test_mse_cv = mean_squared_error(Y_test, Y_test_pred_cv)
test_r2_cv = r2_score(Y_test, Y_test_pred_cv)

print(f"Test MSE with optimal alpha: {test_mse_cv:.4f}")
print(f"Test R² with optimal alpha: {test_r2_cv:.4f}")

plt.figure(figsize=(10, 6))
plt.errorbar(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), 
            yerr=lasso_cv.mse_path_.std(axis=1), 
            label='Mean CV MSE ± 1 std')
plt.axvline(x=optimal_alpha, color='r', linestyle='--', 
           label=f'Optimal α = {optimal_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



summary_data = []

for alpha in alphas:
    summary_data.append({
        'Method': 'Lasso',
        'Alpha': alpha,
        'Test MSE': lasso_results[alpha]['test_mse'],
        'Test R²': lasso_results[alpha]['test_r2'],
        'Non-zero Coefficients': lasso_results[alpha]['n_nonzero_coef']
    })

for alpha in alphas:
    summary_data.append({
        'Method': 'Ridge',
        'Alpha': alpha,
        'Test MSE': ridge_results[alpha]['test_mse'],
        'Test R²': ridge_results[alpha]['test_r2'],
        'Non-zero Coefficients': n_features  # Ridge doesn't set coefficients to zero
    })

summary_data.append({
    'Method': 'Lasso (CV)',
    'Alpha': optimal_alpha,
    'Test MSE': test_mse_cv,
    'Test R²': test_r2_cv,
    'Non-zero Coefficients': np.sum(lasso_cv.coef_ != 0)
})

summary_df = pd.DataFrame(summary_data)
print("\nModel Comparison Summary:")
print(summary_df)


## Ecercise 1: Effect of Sample Size
### Modify the code to investigate how the sample size affects Lasso’s ability to recover the true coefficients. Try n_samples = [100, 200, 1000] and plot the feature selection performance.


n_samples_list = [100, 200, 1000]

# Container for results
lasso_results = []

# Replying code but using bigger for loop
for n_samples in n_samples_list:
    
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.zeros(n_features)
    informative_features = np.random.choice(n_features, n_informative, replace=False)
    for idx in informative_features:
        true_coefficients[idx] = np.random.randn() * 3
    Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, Y_train)
        coef = lasso.coef_
        n_nonzero = np.sum(coef != 0)

        selected = set(np.where(coef != 0)[0])
        true_set = set(informative_features)
        tpr = len(selected & true_set) / len(true_set) if len(true_set) > 0 else 0
        fpr = len(selected - true_set) / (n_features - len(true_set)) if (n_features - len(true_set)) > 0 else 0

        lasso_results.append({
            'n_samples': n_samples,
            'alpha': alpha,
            'n_nonzero_coef': n_nonzero,
            'TPR': tpr,
            'FPR': fpr
        })

# Convert to DataFrame and print
df = pd.DataFrame(lasso_results)
print("\nLasso Feature Selection Summary:")
print(df)

# Plot: Number of non-zero coefficients vs sample size
plt.figure(figsize=(8,6))
for alpha in alphas:
    subset = df[df['alpha'] == alpha]
    plt.plot(subset['n_samples'], subset['n_nonzero_coef'], marker='o', label=f'α = {alpha}')
plt.axhline(y=n_informative, color='r', linestyle='--', label='True informative')
plt.xlabel('Sample size')
plt.ylabel('Number of non-zero coefficients')
plt.title('Lasso: Non-zero coefficients vs Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot: TPR vs sample size
plt.figure(figsize=(8,6))
for alpha in alphas:
    subset = df[df['alpha'] == alpha]
    plt.plot(subset['n_samples'], subset['TPR'], marker='o', label=f'α = {alpha}')
plt.xlabel('Sample size')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Lasso: TPR vs Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot: FPR vs sample size
plt.figure(figsize=(8,6))
for alpha in alphas:
    subset = df[df['alpha'] == alpha]
    plt.plot(subset['n_samples'], subset['FPR'], marker='o', label=f'α = {alpha}')
plt.xlabel('Sample size')
plt.ylabel('False Positive Rate (FPR)')
plt.title('Lasso: FPR vs Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
##### TPR and FPR aboout real coefficients



### Exercise 2: Different Sparsity Levels
## Change the number of informative features (n_informative) to see how sparsity affects performance. Try values like 5, 20, 50, and 100.

### Now on fixed samples and features and different sparsity levels
n_samples = 500
n_features = 100
n_informative_list = [5, 20, 50, 100]  

# Store results
results = []

## For loop again
for n_informative in n_informative_list:
    if n_informative > n_features:
        print(f"Skipping n_informative={n_informative}: exceeds n_features={n_features}")
        continue

    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.zeros(n_features)
    informative_features = np.random.choice(n_features, n_informative, replace=False)
    for idx in informative_features:
        true_coefficients[idx] = np.random.randn() * 3
    y = X @ true_coefficients + np.random.randn(n_samples) * noise_level
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        
        coef = lasso.coef_
        n_nonzero = np.sum(coef != 0)
        n_true_positive = np.sum((coef != 0) & (true_coefficients != 0))
        n_false_positive = np.sum((coef != 0) & (true_coefficients == 0))
        n_noise = n_features - n_informative
        
        tpr = n_true_positive / n_informative if n_informative > 0 else 0
        fpr = n_false_positive / n_noise if n_noise > 0 else 0
        
        results.append({
            'n_informative': n_informative,
            'alpha': alpha,
            'n_nonzero_coef': n_nonzero,
            'TPR': tpr,
            'FPR': fpr
        })

# Results DataFrame
df_results = pd.DataFrame(results)
print(df_results)
## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


## Exercise 3: Correlated Features
## Modify the data generating process to include correlated features. How does this affect Lasso’s performance?
# Hint for Exercise 3: Creating correlated features correlation_matrix = np.eye(n_features). Add some correlations between features and use numpy.random.multivariate_normal to generate X.

n_samples = 500
n_features = 100
n_informative = 20


# Create correlation matrix
correlation_matrix = np.eye(n_features)
correlated_blocks = [(0, 10), (20, 30), (40, 50), (60, 70), (80, 90)]
for start, end in correlated_blocks:
    for j in range(start, end):
        for k in range(start, end):
            if j != k:
                correlation_matrix[j, k] = np.random.uniform(0.4, 0.8)

# Generating data
mean = np.zeros(n_features)
X = np.random.multivariate_normal(mean, correlation_matrix, size=n_samples)

true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3
y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


results = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coef = lasso.coef_
    
    n_nonzero = np.sum(coef != 0)
    tp = np.sum((coef != 0) & (true_coefficients != 0))
    fp = np.sum((coef != 0) & (true_coefficients == 0))
    fn = np.sum((coef == 0) & (true_coefficients != 0))
    tn = np.sum((coef == 0) & (true_coefficients == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    results.append({
        'Alpha': alpha,
        'Non-zero Coef': n_nonzero,
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision
    })

# Results DataFrame
df_results = pd.DataFrame(results)
print("\nFeature Selection Performance Summary:")
print(df_results)

























