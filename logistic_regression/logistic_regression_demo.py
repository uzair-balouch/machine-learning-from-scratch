"""
Lesson 6: Logistic Regression
Classifying tumors as malignant (1) or benign (0) based on cell measurements.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. DATA ---
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Features: {X.shape[1]} (e.g. {data.feature_names[:3]}...)")
print(f"Classes: {data.target_names} -> encoded as {np.unique(y)}")
print(f"Total examples: {len(y)}, Malignant: {sum(y==0)}, Benign: {sum(y==1)}")

# --- 2. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y ensures both classes are proportionally represented in train/test splits

# --- 3. SCALE FEATURES ---
# Logistic regression (and most ML algorithms) train better/faster when
# features are on similar scales. Raw features here range from ~0.05 to ~2500.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NOTE: fit only on train, never on test

# --- 4. TRAIN THE MODEL ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# --- 5. PREDICT ---
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

# --- 6. EVALUATE ---
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("                Predicted Malignant  Predicted Benign")
print(f"Actual Malignant      {cm[0][0]:>5}              {cm[0][1]:>5}")
print(f"Actual Benign         {cm[1][0]:>5}              {cm[1][1]:>5}")

# --- 7. LOOK AT A FEW INDIVIDUAL PREDICTIONS ---
print("\nSample predictions (first 5 test examples):")
for i in range(5):
    actual = data.target_names[y_test[i]]
    predicted = data.target_names[y_pred[i]]
    confidence = y_pred_proba[i] if y_pred[i] == 1 else 1 - y_pred_proba[i]
    print(f"  Actual: {actual:10s} | Predicted: {predicted:10s} | Confidence: {confidence:.2%}")