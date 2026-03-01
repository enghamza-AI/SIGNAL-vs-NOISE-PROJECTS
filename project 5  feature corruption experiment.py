import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

n_samples = 500
n_features = 10          # 10 input features (X columns)

# True signal: only first 3 features matter, others are useless
true_betas = np.array([3.0, -2.5, 1.8] + [0.0] * (n_features - 3))

# Generate clean X (some correlation between features for realism)
X_clean = np.random.normal(0, 1, (n_samples, n_features))
# Add a bit of correlation between first few features
X_clean[:, 1] = 0.6 * X_clean[:, 0] + np.random.normal(0, 0.5, n_samples)
X_clean[:, 2] = -0.4 * X_clean[:, 0] + 0.3 * X_clean[:, 1] + np.random.normal(0, 0.5, n_samples)

# Clean target: y = X @ true_betas + small noise
y_clean = X_clean @ true_betas + np.random.normal(0, 1.5, n_samples)

# Corruption levels: % of features we destroy
corruption_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

r2_scores = []
test_mses = []

for frac_corrupt in corruption_levels:
    num_corrupt = int(frac_corrupt * n_features)
    
    # Copy clean X
    X_corrupted = X_clean.copy()
    
    # Randomly choose which features to corrupt
    corrupt_indices = np.random.choice(n_features, num_corrupt, replace=False)
    
    # Replace chosen features with pure random noise (same scale)
    for idx in corrupt_indices:
        X_corrupted[:, idx] = np.random.normal(0, 1, n_samples)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_corrupted, y_clean, test_size=0.3, random_state=42
    )
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    test_mses.append(mse)
    r2_scores.append(r2)
    
    print(f"Corruption: {frac_corrupt*100:3.0f}%  |  Test MSE: {mse:8.2f}  |  R²: {r2:6.3f}")

# Plot the collapse
plt.figure(figsize=(10, 6))
plt.plot([f"{int(p*100)}%" for p in corruption_levels], test_mses, 'o-', color='red', label='Test MSE')
plt.plot([f"{int(p*100)}%" for p in corruption_levels], r2_scores, 'o-', color='blue', label='Test R²')
plt.axhline(y=1.5**2, color='green', linestyle='--', alpha=0.7, label='Irreducible ≈ 2.25 (noise variance)')
plt.xlabel('Percentage of features corrupted')
plt.ylabel('Performance')
plt.title('Feature Corruption Experiment\n(Only 3/10 features carry real signal)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(bottom=-0.2, top=max(max(test_mses)*1.1, 10))
plt.show()