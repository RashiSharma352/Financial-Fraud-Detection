import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ============================================================
# 1. LOAD EXCEL DATA
# ============================================================

df = pd.read_excel("Data/Fraud Detection.xlsx")

print("Original dataset:", df.shape)


# ============================================================
# 2. KEEP ONLY VALID LABELS
# ============================================================

df = df[df["isFraud"].isin(["Safe", "Fraud"])].copy()

print("\nAfter removing 'Not reviewed':", df.shape)


# ============================================================
# 3. CREATE TARGET
# ============================================================

df["Fraud_Label"] = df["isFraud"].map({
    "Safe": 0,
    "Fraud": 1
})


# ============================================================
# 4. REMOVE UNNECESSARY / DUPLICATE COLUMNS
# ============================================================

columns_to_drop = [
    "isFraud",
    "isFraud - Copy",
    "Column1",
    "nameOrig",
    "nameDest"
]

df = df.drop(
    columns=[c for c in columns_to_drop if c in df.columns]
)


# ============================================================
# 5. HANDLE DATE
# ============================================================

if "Date of transaction" in df.columns:

    df["Date of transaction"] = pd.to_datetime(
        df["Date of transaction"],
        errors="coerce"
    )

    df["Transaction_Year"] = df["Date of transaction"].dt.year
    df["Transaction_Month"] = df["Date of transaction"].dt.month
    df["Transaction_Day"] = df["Date of transaction"].dt.day

    df = df.drop(columns=["Date of transaction"])


# ============================================================
# 6. REMOVE ROWS WITH MISSING VALUES
# ============================================================

df = df.dropna()

print("\nFinal dataset:", df.shape)


# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 8. CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# 9. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 11. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 12. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ]
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\nTraining Fraud Detection Model...")

pipeline.fit(X_train, y_train)


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 15. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("FINAL FRAUD DETECTION MODEL")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "models/fraud_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/fraud_model.pkl")
