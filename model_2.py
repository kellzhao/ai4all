import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 1. Load the full imbalanced dataset
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Calculate class weight ratio
ratio = (y_train == 0).sum() / (y_train == 1).sum()

# 4. Train XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# 5. Get predicted probabilities for class 1 (Diabetes)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== ROC-AUC Score ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}\n")

# 6. THRESHOLD TUNING LOOP
print("==========================================")
print("     EVALUATING DECISION THRESHOLDS       ")
print("==========================================")

thresholds = [0.50, 0.40, 0.35, 0.30]

for t in thresholds:
    # Classify as 1 if probability >= threshold
    y_pred_t = (y_proba >= t).astype(int)
    
    print(f"\n--- Decision Threshold: {t} ---")
    print(classification_report(y_test, y_pred_t, digits=4))
