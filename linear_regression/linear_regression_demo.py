"""
Lesson 5: Linear Regression from scratch (using scikit-learn)
Predicting house prices from square footage.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- 1. DATA ---
# Feature (X): square footage. Label (y): price in $1000s.
# In real life this would be loaded from a CSV, but we're generating
# synthetic data so you can see the whole pipeline clearly.
np.random.seed(42)
square_footage = np.random.randint(800, 3500, 200)
# True relationship: price = 0.15*sqft + 20 (in $1000s), plus some noise
price = 0.15 * square_footage + 20 + np.random.normal(0, 15, 200)

X = square_footage.reshape(-1, 1)  # sklearn expects 2D array: (n_samples, n_features)
y = price

# --- 2. TRAIN/TEST SPLIT (Lesson 4!) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")

# --- 3. TRAIN THE MODEL ---
# Under the hood, sklearn is running the "predict -> measure error ->
# adjust w and b -> repeat" loop from Lesson 5 (using an optimized
# solver, not raw gradient descent, but the same underlying idea).
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. INSPECT THE LEARNED PARAMETERS ---
w = model.coef_[0]   # the learned weight (slope)
b = model.intercept_ # the learned bias (intercept)
print(f"\nLearned equation: price = {w:.4f} * sqft + {b:.2f}")
print("(True underlying relationship was: price = 0.15 * sqft + 20)")

# --- 5. EVALUATE ON TEST DATA ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE: {mse:.2f}")
print(f"Test R^2 score: {r2:.4f}  (1.0 = perfect, 0.0 = no better than guessing the mean)")

# --- 6. TRY A PREDICTION ---
new_house_sqft = np.array([[2000]])
predicted_price = model.predict(new_house_sqft)
print(f"\nPredicted price for a 2000 sqft house: ${predicted_price[0]:.2f}k")