# train_model.py
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Paths ---
DATA = "data/heart.csv"
MODELS_DIR = "models"
ASSETS_DIR = "assets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA)

# Try to auto-detect target column
possible_targets = ["target", "Target", "HeartDisease", "Outcome", "outcome", "label", "Label"]
target = None
for t in possible_targets:
    if t in df.columns:
        target = t
        break
if target is None:
    # Fallback for common UCI “target” (0/1) — create if absent and data matches Cleveland
    raise ValueError("No target column found. Ensure your CSV has a binary target column (e.g., 'target').")

y = df[target].astype(int)
X = df.drop(columns=[target])

# Basic heuristic: numeric vs categorical by dtype
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optional: handle imbalance
if y_train.value_counts().min() / y_train.value_counts().max() < 0.7:
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
else:
    X_train_sm, y_train_sm = X_train, y_train

# Models to compare
candidates = {
    "LogReg": LogisticRegression(max_iter=200, n_jobs=None if hasattr(LogisticRegression(), 'n_jobs') else None),
    "RF": RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    ),
    "XGB": XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
        random_state=42
    )
}

results = []
best_name, best_pipe, best_scores, best_proba = None, None, None, None

for name, model in candidates.items():
    pipe = Pipeline([("prep", pre), ("clf", model)])
    pipe.fit(X_train_sm, y_train_sm)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    scores = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba)
    }
    results.append((name, scores))

    if best_scores is None or scores["roc_auc"] > best_scores["roc_auc"]:
        best_name, best_pipe, best_scores, best_proba = name, pipe, scores, proba

# Persist best model
# Also persist preprocessor separately for app-side transformations if needed
joblib.dump(best_pipe.named_steps["clf"], os.path.join(MODELS_DIR, "model.joblib"))
joblib.dump(best_pipe.named_steps["prep"], os.path.join(MODELS_DIR, "preprocessor.joblib"))

# Metrics JSON
metrics = {
    "selected": best_name,
    "by_model": {n: s for n, s in results}
}
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# --- Plots ---
# Model comparison
fig, ax = plt.subplots(figsize=(6, 4))
names = [n for n, _ in results]
roc_aucs = [s["roc_auc"] for _, s in results]
sns.barplot(x=roc_aucs, y=names, ax=ax)
ax.set_xlabel("AUROC")
ax.set_ylabel("Model")
ax.set_title("Model Performance (AUROC)")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "model_comparison.png"), dpi=150)
plt.close(fig)

# ROC for best
fig2, ax2 = plt.subplots(figsize=(6, 4))
RocCurveDisplay.from_predictions(y_test, best_proba, ax=ax2, name=best_name)
ax2.set_title(f"ROC Curve — {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "roc_curve.png"), dpi=150)
plt.close(fig2)

# SHAP global importance (tree models preferred, fallback to kernel)
try:
    clf = best_pipe.named_steps["clf"]
    X_trans = best_pipe.named_steps["prep"].fit_transform(X)
    shap_explainer = None
    if hasattr(clf, "get_booster") or "XGB" in best_name or "RF" in best_name:
        shap_explainer = shap.TreeExplainer(clf)
        shap_vals = shap_explainer.shap_values(X_trans)
        vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]
    else:
        shap_explainer = shap.KernelExplainer(clf.predict_proba, X_trans[:100])
        vals = shap_explainer.shap_values(X_trans[:100])[1]

    # SHAP summary plot (save as image)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(vals, X_trans, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "shap_summary.png"), dpi=150)
    plt.close()
except Exception as e:
    print("SHAP plot skipped:", e)

print("✅ Training complete. Artifacts saved in 'models/' and plots in 'assets/'.")
