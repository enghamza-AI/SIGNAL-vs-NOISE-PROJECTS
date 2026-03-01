import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

true_slope = 2.5
true_intercept = 1.0

# Fixed noise level — this creates the irreducible error floor
noise_std = 6.0                 
irreducible_mse_approx = noise_std ** 2   

sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

train_mses = []
test_mses = []

for n in sample_sizes:
    # Generate data
    X = np.random.uniform(-5, 5, n).reshape(-1, 1)
    y_clean = true_slope * X + true_intercept
    noise = np.random.normal(0, noise_std, n).reshape(-1, 1)
    y_noisy = y_clean + noise
    
    # Split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_noisy, test_size=0.3, random_state=42
    )
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions & errors
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    train_mses.append(train_mse)
    test_mses.append(test_mse)
    
    print(f"n = {n:5d} | Train MSE: {train_mse:8.2f} | Test MSE: {test_mse:8.2f}")

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, train_mses, 'o-', color='blue', label='Train MSE')
plt.plot(sample_sizes, test_mses, 'o-', color='red', label='Test MSE')
plt.axhline(y=irreducible_mse_approx, color='green', linestyle='--', 
            label=f'Irreducible ≈ {irreducible_mse_approx:.1f}')
plt.xscale('log')
plt.xlabel('Number of training samples (log scale)')
plt.ylabel('Mean Squared Error')
plt.title(f'Irreducible Error Demo (noise std = {noise_std})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()