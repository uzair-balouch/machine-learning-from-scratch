"""
LESSON 6 + 7 COMBINED: Logistic Regression + Full Model Evaluation
====================================================================
Goal: Classify tumors as malignant (cancerous) or benign (not cancerous)
based on measurements taken from cell samples.

This single file walks through the ENTIRE real-world workflow:
  1. Load and understand the data
  2. Split into train/test (so we can honestly check our work)
  3. Scale the features (helps the model learn properly)
  4. Train the model (this is "Lesson 6" - logistic regression)
  5. Check basic performance: accuracy, precision, recall
  6. Check DEEPER performance: ROC curve, AUC (this is "Lesson 7")
  7. Check if our good results are real or just luck: cross-validation

Every step below explains WHY it exists, not just what it does.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =========================================================
# STEP 1: LOAD THE DATA
# =========================================================
# WHY: Before touching any model, we need to know what we're working with.
# This dataset has real measurements from breast tumor cells (like size,
# texture, smoothness) and a label saying whether each tumor was
# malignant (dangerous) or benign (not dangerous).

data = load_breast_cancer()
X = data.data          # the input features (30 numeric measurements per tumor)
y = data.target        # the label: 0 = malignant, 1 = benign

print("=" * 60)
print("STEP 1: UNDERSTAND THE DATA")
print("=" * 60)
print(f"Number of tumors in dataset: {len(y)}")
print(f"Number of features per tumor: {X.shape[1]}")
print(f"Malignant cases: {sum(y == 0)}, Benign cases: {sum(y == 1)}")


# =========================================================
# STEP 2: SPLIT INTO TRAINING AND TEST SETS
# =========================================================
# WHY: If we let the model "study" using ALL the data, we'd have no
# honest way to check if it actually learned the pattern, or just
# memorized the examples (this is the overfitting problem from Lesson 4).
# So we hide 20% of the data away and never let the model see it
# during training - it's our "final exam" at the end.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # hold back 20% of data as the test set
    random_state=42,    # makes the split reproducible (same split every run)
    stratify=y          # keeps the malignant/benign ratio the same in both sets
)

print(f"\nTraining examples (model learns from these): {len(y_train)}")
print(f"Test examples (model never sees these until the end): {len(y_test)}")


# =========================================================
# STEP 3: SCALE THE FEATURES
# =========================================================
# WHY: Our 30 features are on very different scales (e.g. one feature
# might range 0.05-0.2, another might range 100-2500). Logistic
# regression learns faster and more reliably when everything is on a
# similar scale. We calculate the scaling using ONLY training data,
# then apply that same scaling to test data - never the other way
# around, or we'd be "leaking" information from the test set.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn the scale AND apply it
X_test_scaled = scaler.transform(X_test)         # only apply, don't relearn


# =========================================================
# STEP 4: TRAIN THE LOGISTIC REGRESSION MODEL  (Lesson 6)
# =========================================================
# WHY: This is where the actual "learning" happens. The model looks at
# the training examples (features + correct answers) and adjusts its
# internal numbers (weights) until it gets good at predicting the
# right answer. Internally, it does NOT just say "malignant" or
# "benign" - it calculates a confidence score between 0 and 1 for
# each tumor (e.g. "92% sure this is benign").

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Now ask the trained model to make predictions on the test set
# (data it has NEVER seen before):
y_pred = model.predict(X_test_scaled)                       # hard yes/no answer
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]      # confidence score (0 to 1)


# =========================================================
# STEP 5: BASIC EVALUATION - Accuracy, Precision, Recall
# =========================================================
# WHY: Now that the model has made predictions, we need to measure how
# good those predictions actually are. We use THREE different
# measurements because each one tells a different part of the story
# (a lesson from Lesson 6: a single number like accuracy can hide
# important mistakes).

print("\n" + "=" * 60)
print("STEP 5: BASIC PERFORMANCE (using the default 0.5 cutoff)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}  (overall, how often was it right?)")
print(f"Precision: {precision_score(y_test, y_pred):.4f}  (of predicted 'benign', how many really were?)")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}  (of actual 'benign' cases, how many did we catch?)")

# The confusion matrix shows exactly WHAT KIND of mistakes were made,
# not just how many:
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                     Predicted Malignant   Predicted Benign")
print(f"Actual Malignant            {cm[0][0]:>5}                {cm[0][1]:>5}")
print(f"Actual Benign               {cm[1][0]:>5}                {cm[1][1]:>5}")
print("(top-right = missed cancer cases -> the most dangerous mistake)")


# =========================================================
# STEP 6: DEEPER EVALUATION - ROC Curve and AUC  (Lesson 7)
# =========================================================
# WHY: Accuracy/precision/recall above all depend on ONE fixed cutoff
# (0.5 by default). But what if a different cutoff would work better?
# The ROC curve answers: "what if we tried EVERY possible cutoff, from
# 0 to 1 - how would the model behave at each one?" AUC then squashes
# that entire curve into a single score (1.0 = perfect, 0.5 = random
# guessing) so we can judge the model's raw skill, independent of
# whatever cutoff we eventually choose to use.

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("STEP 6: ROC CURVE - testing every possible cutoff at once")
print("=" * 60)
print(f"{'Cutoff':>10} | {'False Alarms (FPR)':>20} | {'Cases Caught (TPR)':>20}")
print("-" * 60)
for i in [0, len(thresholds) // 4, len(thresholds) // 2, len(thresholds) - 1]:
    cutoff = thresholds[i] if thresholds[i] != np.inf else 1.0
    print(f"{cutoff:>10.3f} | {fpr[i]:>20.3f} | {tpr[i]:>20.3f}")

print(f"\nAUC score: {auc:.4f}  (0.5 = no better than a coin flip, 1.0 = perfect)")

# Draw the actual curve so we can SEE this instead of just reading numbers.
# The closer the blue line hugs the top-left corner, the better the model.
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'Our Model (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='orange', linestyle='--', linewidth=2, label='Random Guessing (AUC = 0.5)')
plt.xlabel('False Positive Rate (false alarms)')
plt.ylabel('True Positive Rate (real cases caught)')
plt.title('ROC Curve: Cancer Detection Model')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('logistic_regression/roc_curve.png', dpi=100, bbox_inches='tight')
print("Saved plot to: r oc_curve.png")


# =========================================================
# STEP 7: CROSS-VALIDATION - is this performance real, or luck?
# =========================================================
# WHY: We only tested our model on ONE particular 20% slice of data.
# What if that slice happened to be unusually easy (or hard) by pure
# chance? Cross-validation repeats the whole train/test process 5
# times, using a DIFFERENT 20% slice as the test set each time, so we
# get 5 separate scores instead of trusting just one.

# We rebuild scaling + model as a "pipeline" so that scaling is
# recalculated fresh inside EACH of the 5 rounds, using only that
# round's training data - keeping every round leak-free, just like Step 3.
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

print("\n" + "=" * 60)
print("STEP 7: CROSS-VALIDATION (5 different train/test splits)")
print("=" * 60)
for i, score in enumerate(cv_scores, 1):
    print(f"  Round {i}: AUC = {score:.4f}")

print(f"\nAverage AUC across all 5 rounds: {cv_scores.mean():.4f}")
print(f"Spread (std dev):                {cv_scores.std():.4f}")
print("(small spread = model is consistently good, not just lucky once)")