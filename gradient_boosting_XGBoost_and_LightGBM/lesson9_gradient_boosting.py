"""
LESSON 9: Gradient Boosting vs Random Forest
================================================
Goal: See the sequential, error-correcting nature of gradient boosting
in action, and compare it directly against last lesson's random forest.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("PART 1: RANDOM FOREST (from Lesson 8, for comparison)")
print("=" * 60)
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
forest_acc = accuracy_score(y_test, forest.predict(X_test))
forest_auc = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
print(f"Test accuracy: {forest_acc:.4f}")
print(f"Test AUC:      {forest_auc:.4f}")


print("\n" + "=" * 60)
print("PART 2: GRADIENT BOOSTING - trees built sequentially")
print("=" * 60)
# WHY these settings:
# n_estimators=200      -> build 200 trees, one after another
# learning_rate=0.1     -> how much each new tree's correction counts
#                          (small = safer/slower, large = faster/riskier)
# max_depth=3           -> keep EACH individual tree small/weak on purpose
#                          (boosting works best with many weak trees,
#                          not few strong ones - opposite of random forest!)
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
print(f"Test accuracy: {gb_acc:.4f}")
print(f"Test AUC:      {gb_auc:.4f}")


print("\n" + "=" * 60)
print("PART 3: WATCHING THE ERROR-CORRECTION HAPPEN, STEP BY STEP")
print("=" * 60)
# WHY: staged_predict lets us see the model's prediction using only
# the first N trees, showing accuracy improve as MORE trees get added
# one at a time - this is the "sequential fixing" idea made visible.
staged_accuracies = []
for i, y_pred_stage in enumerate(gb.staged_predict(X_test)):
    staged_accuracies.append(accuracy_score(y_test, y_pred_stage))

checkpoints = [0, 4, 9, 24, 49, 99, 199]
print(f"{'Trees used':>12} | {'Test Accuracy':>15}")
print("-" * 32)
for cp in checkpoints:
    print(f"{cp+1:>12} | {staged_accuracies[cp]:>15.4f}")