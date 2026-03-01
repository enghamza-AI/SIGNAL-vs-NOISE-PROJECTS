import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

n_samples = 100
X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
true_slope = 2.0
true_intercept = 1.0
y_clean = true_slope * X + true_intercept

plt.scatter(X, y_clean, color='blue', label='Clean data')
plt.plot(X, true_slope * X + true_intercept, color='red', label='True Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Clean Signal: y = 2x + 1')
plt.legend()
plt.show()


def noise_experiment(noise_level):
    
    noise = np.random.normal(0, noise_level, size=n_samples).reshape(-1, 1)
    y_noisy = y_clean + noise
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.3, random_state=42)
    
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
   
    learned_slope = model.coef_[0][0]
    learned_intercept = model.intercept_[0]
    
    print(f"\nNoise level (std): {noise_level}")
    print(f"Learned slope: {learned_slope:.3f} (true=2.0)")
    print(f"Learned intercept: {learned_intercept:.3f} (true=1.0)")
    print(f"Train MSE: {train_mse:.3f}")
    print(f"Test MSE: {test_mse:.3f}")
    
    
    plt.scatter(X, y_noisy, color='blue', alpha=0.6, label='Noisy data')
    plt.plot(X, true_slope * X + true_intercept, color='red', label='True line')
    plt.plot(X, model.coef_[0][0] * X + model.intercept_[0], color='green', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Noisy data & fit (noise std = {noise_level})')
    plt.legend()
    plt.show()


noise_levels = [0.1, 1.0, 5.0, 15.0, 30.0, 50.0]
for level in noise_levels:
    noise_experiment(level)