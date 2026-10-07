from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "fraud_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "fraud_classifier_results.csv"


def main():
    print("Loading feature data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Transactions loaded: {len(df):,}")

    # Features available before knowing the fraud label.
    features = [
        "amount",
        "transactions_1h",
        "amount_24h",
        "failed_attempts_24h",
        "minutes_since_previous",
        "is_new_device",
        "is_new_location",
        "high_velocity_flag",
        "high_value_flag",
        "failed_attempt_flag",
        "rapid_transaction_flag",
        "multiple_risk_signals",
    ]

    target = "fraud_label"

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if target not in df.columns:
        raise ValueError(
            f"Missing target column: {target}"
        )

    X = df[features].copy()
    y = df[target].astype(int)

    print("\nClass distribution:")
    print(y.value_counts().to_string())

    # Stratified split keeps the suspicious/normal ratio
    # consistent between training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("\nTraining set:", len(X_train))
    print("Test set:", len(X_test))

    # Standardise numerical features so variables with
    # different scales do not dominate the model.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    # Predictions
    predictions = model.predict(X_test_scaled)

    # Probability of class 1
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Suspicious"],
            zero_division=0,
        )
    )

    # Apply the trained model to the entire dataset.
    X_all_scaled = scaler.transform(X)

    df["fraud_probability"] = model.predict_proba(
        X_all_scaled
    )[:, 1]

    df["classifier_prediction"] = model.predict(
        X_all_scaled
    )

    # Save the full dataset with model outputs.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nFull dataset predictions:")
    print(
        df["classifier_prediction"]
        .value_counts()
        .to_string()
    )

    print("\nAverage fraud probability:")
    print(
        f"{df['fraud_probability'].mean():.4f}"
    )

    print("\nHighest fraud probabilities:")

    top_transactions = (
        df[
            [
                "transaction_id",
                "scenario_type",
                "fraud_label",
                "fraud_probability",
                "classifier_prediction",
            ]
        ]
        .sort_values(
            "fraud_probability",
            ascending=False,
        )
        .head(10)
    )

    print(top_transactions.to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()