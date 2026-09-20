"""
LESSON 8: Decision Trees vs Random Forests
============================================
Goal: See directly, with numbers, why a single deep tree overfits,
and how a forest of trees fixes that problem.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# DATA (no scaling needed this time - trees don't require it!)
# =========================================================
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("PART 1: A SINGLE DECISION TREE - with NO depth limit")
print("=" * 60)
# WHY no max_depth: letting the tree grow as deep as it wants shows
# the overfitting problem in its most extreme, obvious form.
tree_deep = DecisionTreeClassifier(random_state=42)
tree_deep.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree_deep.predict(X_train))
test_acc = accuracy_score(y_test, tree_deep.predict(X_test))
print(f"Tree depth grown: {tree_deep.get_depth()}")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy:     {test_acc:.4f}")
print(f"Gap (overfitting signal): {train_acc - test_acc:.4f}")


print("\n" + "=" * 60)
print("PART 2: A SINGLE DECISION TREE - with a depth LIMIT")
print("=" * 60)
# WHY limit depth: forcing the tree to stay shallow prevents it from
# creating a super-specific branch for every training example.
tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X_train, y_train)

train_acc_shallow = accuracy_score(y_train, tree_shallow.predict(X_train))
test_acc_shallow = accuracy_score(y_test, tree_shallow.predict(X_test))
print(f"Tree depth (capped): {tree_shallow.get_depth()}")
print(f"Training accuracy: {train_acc_shallow:.4f}")
print(f"Test accuracy:     {test_acc_shallow:.4f}")
print(f"Gap (overfitting signal): {train_acc_shallow - test_acc_shallow:.4f}")


print("\n" + "=" * 60)
print("PART 3: RANDOM FOREST - many trees voting together")
print("=" * 60)
# WHY: instead of manually limiting depth to prevent overfitting,
# let trees grow deep individually, but combine hundreds of DIFFERENT
# deep trees so their individual overfitting mistakes cancel out.
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)

train_acc_forest = accuracy_score(y_train, forest.predict(X_train))
test_acc_forest = accuracy_score(y_test, forest.predict(X_test))
print(f"Number of trees: {forest.n_estimators}")
print(f"Training accuracy: {train_acc_forest:.4f}")
print(f"Test accuracy:     {test_acc_forest:.4f}")
print(f"Gap (overfitting signal): {train_acc_forest - test_acc_forest:.4f}")


print("\n" + "=" * 60)
print("BONUS: FEATURE IMPORTANCE - which measurements mattered most?")
print("=" * 60)
# WHY this is useful: unlike logistic regression's abstract weights,
# random forests give an intuitive "how much did this feature help
# make correct predictions overall" score. Great for explaining a
# model's behavior to non-technical stakeholders.
importances = forest.feature_importances_
top_5_idx = np.argsort(importances)[-5:][::-1]
print("Top 5 most important features:")
for idx in top_5_idx:
    print(f"  {data.feature_names[idx]:25s}: {importances[idx]:.4f}")