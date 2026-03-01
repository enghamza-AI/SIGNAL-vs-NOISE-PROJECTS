import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)

n_samples = 150

# Create X the same for all three cases
X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)

# Case 1: Strong signal (clean linear pattern)
true_slope_strong = 2.5
true_intercept_strong = 1.2
y_strong = true_slope_strong * X + true_intercept_strong + np.random.normal(0, 0.8, n_samples).reshape(-1, 1)

# Case 2: Weak signal (same slope but huge noise)
y_weak = true_slope_strong * X + true_intercept_strong + np.random.normal(0, 8.0, n_samples).reshape(-1, 1)

# Case 3: No signal (pure random y)
y_none = np.random.uniform(-20, 20, n_samples).reshape(-1, 1)

# Function to fit a line and get two simple scores
def get_signal_scores(X, y):
    model = LinearRegression()
    model.fit(X, y.ravel())
    
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    
    return r2, abs(slope)

# Function that guesses signal strength
def guess_signal_strength(r2, abs_slope):
    if r2 > 0.75 and abs_slope > 1.0:
        return "strong"
    elif r2 > 0.15 and abs_slope > 0.5:
        return "weak"
    else:
        return "none"

# Run for all three cases
datasets = {
    "Strong signal": y_strong,
    "Weak signal": y_weak,
    "No signal": y_none
}

plt.figure(figsize=(15, 5))

for i, (name, y) in enumerate(datasets.items(), 1):
    r2, abs_slope = get_signal_scores(X, y)
    guess = guess_signal_strength(r2, abs_slope)
    
    print(f"\n{name}:")
    print(f"  R² score:     {r2:.3f}")
    print(f"  |Slope|:      {abs_slope:.3f}")
    print(f"  Guess:        {guess}")
    
    plt.subplot(1, 3, i)
    plt.scatter(X, y, color='blue', alpha=0.6, s=40)
    model = LinearRegression().fit(X, y)
    plt.plot(X, model.predict(X), color='red', linewidth=2.5)
    plt.title(f"{name}\nGuess: {guess}\nR²={r2:.2f}")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()