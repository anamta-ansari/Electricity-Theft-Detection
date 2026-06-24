import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ========================================================
# LOAD DATASET
# ========================================================

df = pd.read_csv("Electricity_Theft_Data.csv")

# ========================================================
# DATA CLEANING
# ========================================================

for col in df.columns:
    if col not in ["CONS_NO", "CHK_STATE"]:
        df[col] = df[col].fillna(df[col].mean())

df = df.dropna(subset=["CHK_STATE"])
df["CHK_STATE"] = df["CHK_STATE"].astype(int)

# ========================================================
# FEATURE ENGINEERING
# ========================================================

date_cols = [
    col for col in df.columns
    if col not in ["CONS_NO", "CHK_STATE"]
]

df["mean_usage"] = df[date_cols].mean(axis=1)
df["max_usage"] = df[date_cols].max(axis=1)
df["min_usage"] = df[date_cols].min(axis=1)
df["std_usage"] = df[date_cols].std(axis=1)

df["total_usage"] = df[date_cols].sum(axis=1)

df["zero_usage_days"] = (
    df[date_cols] == 0
).sum(axis=1)

df["consumption_variance"] = (
    df[date_cols].var(axis=1)
)

df["drop_ratio"] = (
    df["min_usage"] /
    (df["max_usage"] + 1)
)

df["peak_usage_ratio"] = (
    df["max_usage"] /
    (df["mean_usage"] + 1)
)

# ========================================================
# KEEP ONLY ENGINEERED FEATURES
# ========================================================

engineered = [
    "mean_usage",
    "max_usage",
    "min_usage",
    "std_usage",
    "total_usage",
    "zero_usage_days",
    "consumption_variance",
    "drop_ratio",
    "peak_usage_ratio"
]

df_clean = df[
    engineered + ["CHK_STATE"]
]

# ========================================================
# FEATURES & TARGET
# ========================================================

X = df_clean.drop(columns=["CHK_STATE"])
y = df_clean["CHK_STATE"]

# ========================================================
# TRAIN TEST SPLIT
# ========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ========================================================
# MODEL
# ========================================================

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)

# ========================================================
# EVALUATION
# ========================================================

probs = model.predict_proba(X_test)[:, 1]

y_pred = (probs > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ========================================================
# SAVE MODEL
# ========================================================

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully")