import os
import glob
import warnings
warnings.filterwarnings("ignore")

import kagglehub
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# ============================================================
# 1. DOWNLOAD DATASET
# ============================================================

print("Downloading dataset...")

path = kagglehub.dataset_download("kartik2112/fraud-detection")

print("Dataset path:", path)

csv_files = glob.glob(
    os.path.join(path, "**", "*.csv"),
    recursive=True
)

train_files = [
    f for f in csv_files
    if "fraudTrain" in os.path.basename(f)
]

test_files = [
    f for f in csv_files
    if "fraudTest" in os.path.basename(f)
]

if not train_files:
    raise FileNotFoundError("fraudTrain.csv not found.")

train_file = train_files[0]

print("Training file:", train_file)

df = pd.read_csv(train_file)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # Transaction Date Features
    # --------------------------------------------------------

    data["trans_date_trans_time"] = pd.to_datetime(
        data["trans_date_trans_time"]
    )

    data["transaction_hour"] = (
        data["trans_date_trans_time"].dt.hour
    )

    data["transaction_day"] = (
        data["trans_date_trans_time"].dt.day
    )

    data["transaction_month"] = (
        data["trans_date_trans_time"].dt.month
    )

    data["transaction_weekday"] = (
        data["trans_date_trans_time"].dt.dayofweek
    )

    # Weekend flag
    data["is_weekend"] = (
        data["transaction_weekday"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Customer Age
    # --------------------------------------------------------

    data["dob"] = pd.to_datetime(
        data["dob"],
        errors="coerce"
    )

    data["customer_age"] = (
        data["trans_date_trans_time"] - data["dob"]
    ).dt.days / 365.25

    data["customer_age"] = data["customer_age"].clip(
        18, 100
    )

    # --------------------------------------------------------
    # Log Transaction Amount
    # --------------------------------------------------------

    data["amt_log"] = np.log1p(
        data["amt"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Haversine Distance
    # --------------------------------------------------------

    def haversine(lat1, lon1, lat2, lon2):

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            +
            np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        return 6371 * 2 * np.arcsin(
            np.sqrt(a)
        )

    data["distance_km"] = haversine(
        data["lat"],
        data["long"],
        data["merch_lat"],
        data["merch_long"]
    )

    # --------------------------------------------------------
    # Drop unnecessary / high-cardinality ID columns
    # --------------------------------------------------------

    columns_to_drop = [
        "Unnamed: 0",
        "trans_date_trans_time",
        "dob",
        "cc_num",
        "first",
        "last",
        "street",
        "zip",
        "unix_time",
        "trans_num"
    ]

    data = data.drop(
        columns=[
            c for c in columns_to_drop
            if c in data.columns
        ],
        errors="ignore"
    )

    return data


print("\nCreating engineered features...")

df = create_features(df)

print("Feature-engineered shape:", df.shape)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("After duplicate removal:", df.shape)


# ============================================================
# 4. TARGET
# ============================================================

target = "is_fraud"

X = df.drop(columns=[target])
y = df[target]


# ============================================================
# 5. HANDLE CLASS IMBALANCE
# ============================================================

print("\nOriginal class distribution:")
print(y.value_counts())

fraud_data = df[df[target] == 1]
legitimate_data = df[df[target] == 0]

print("\nFraud transactions:", len(fraud_data))
print("Legitimate transactions:", len(legitimate_data))


# ------------------------------------------------------------
# Keep all fraud cases and sample legitimate transactions.
# This makes training considerably faster while retaining
# the minority class.
# ------------------------------------------------------------

legitimate_sample_size = min(
    len(legitimate_data),
    len(fraud_data) * 4
)

legitimate_sample = legitimate_data.sample(
    n=legitimate_sample_size,
    random_state=42
)

balanced_df = pd.concat(
    [fraud_data, legitimate_sample]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\nTraining dataset after balancing:")
print(balanced_df[target].value_counts())


X = balanced_df.drop(columns=[target])
y = balanced_df[target]


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. IDENTIFY NUMERICAL & CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 9. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 10. TRAIN MODELS
# ============================================================

results = []
trained_models = {}

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_val)

    probabilities = pipeline.predict_proba(
        X_val
    )[:, 1]

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 80)
print(
    results_df.sort_values(
        "F1 Score",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[
    best_model_name
]

print(
    f"\nBest Model based on F1 Score: "
    f"{best_model_name}"
)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/fraud_model.pkl"
)

joblib.dump(
    {
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    },
    "models/model_metadata.pkl"
)


# Save validation data for Streamlit visualizations

validation_output = X_val.copy()
validation_output["actual"] = y_val.values

for name, pipeline in trained_models.items():

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    validation_output[
        f"{safe_name}_prob"
    ] = pipeline.predict_proba(
        X_val
    )[:, 1]

    validation_output[
        f"{safe_name}_pred"
    ] = pipeline.predict(X_val)


validation_output.to_csv(
    "models/validation_predictions.csv",
    index=False
)


results_df.to_csv(
    "models/model_results.csv",
    index=False
)


print("\nModel files saved successfully!")
print("models/fraud_model.pkl")
print("models/model_metadata.pkl")
print("models/validation_predictions.csv")
print("models/model_results.csv")