"""
LESSON 8 (continued): Predicting on a SINGLE new, real-world patient
======================================================================
This simulates: a doctor enters one new patient's measurements,
and we want the model to say malignant or benign.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Train the model (same as before) ---
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)

print("Feature names the model expects, in order:")
for i, name in enumerate(data.feature_names):
    print(f"  {i}: {name}")

# =========================================================
# SIMULATE A BRAND NEW PATIENT WALKING INTO THE OFFICE
# =========================================================
# WHY: in the real world, this array wouldn't come from the sklearn
# dataset - it would come from a lab machine, a form the doctor
# filled in, or a hospital database. But the MODEL doesn't care where
# the numbers came from - it just needs the same 30 numbers, in the
# same order, that it was trained on.

new_patient_measurements = np.array([[
    17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.40, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.60, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]])
# NOTE: sklearn's .predict() ALWAYS expects a 2D array, even for one
# patient - that's why there are double brackets [[ ]] above. Think of
# it as "a list containing one patient's list of measurements."

# =========================================================
# GET THE PREDICTION
# =========================================================
prediction = forest.predict(new_patient_measurements)
probability = forest.predict_proba(new_patient_measurements)

predicted_class = data.target_names[prediction[0]]
confidence = probability[0][prediction[0]]

print("\n" + "=" * 50)
print("PREDICTION FOR THIS NEW PATIENT")
print("=" * 50)
print(f"Predicted diagnosis: {predicted_class.upper()}")
print(f"Confidence: {confidence:.2%}")
print(f"Full probability breakdown: malignant={probability[0][0]:.2%}, benign={probability[0][1]:.2%}")