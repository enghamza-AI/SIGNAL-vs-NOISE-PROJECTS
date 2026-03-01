#Sample output of the program
#== Fake Pattern Generator Results ===
#lrned slope:     0.043     (there is NO real slope!)
#leared intercept: 0.867  (no real intercept either)
   #Train MSE:         40.50
   #Test MSE:          42.99
    # The model thinks it found something... but it's all made up!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

n_samples = 100

X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)

y_random = np.random.uniform(-10, 12, n_samples)

plt.scatter(X, y_random, color='purple', alpha=0.7, label='pure random data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('project 2: No signal, Just noise')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_random, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

learned_slope = model.coef_[0]
learned_intercept = model.intercept_

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("\n=== Fake Pattern Generator Results ===")
print(f"Learned slope:     {learned_slope:.3f}     (there is NO real slope!)")
print(f"Learned intercept: {learned_intercept:.3f}  (no real intercept either)")
print(f"Train MSE:         {train_mse:.2f}")
print(f"Test MSE:          {test_mse:.2f}")
print("→ The model thinks it found something... but it's all made up!")

plt.scatter(X, y_random, color='purple', alpha=0.7, label='Pure random data')
plt.plot(X, learned_slope * X + learned_intercept, color='green', linewidth=3, label="Model's confident guess")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Pure Random Data + Fitted Line (Pure Hallucination)')
plt.legend()
plt.show()