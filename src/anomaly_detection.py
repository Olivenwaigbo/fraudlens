from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "fraud_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "anomaly_detection_results.csv"


def main():
    print("Loading feature data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Transactions loaded: {len(df):,}")

    # Features used by the anomaly detector.
    # These describe transaction behaviour rather than directly
    # telling the model whether a transaction is fraudulent.
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

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[features].copy()

    print("\nFeatures used:")
    for feature in features:
        print(f"- {feature}")

    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly
    predictions = model.predict(X)

    # Lower scores indicate more unusual observations.
    anomaly_scores = model.decision_function(X)

    df["anomaly_prediction"] = predictions
    df["anomaly_score"] = anomaly_scores

    df["anomaly_detected"] = (
        df["anomaly_prediction"] == -1
    ).astype(int)

    # Convert anomaly score into a simpler operational level.
    df["anomaly_level"] = "LOW"

    df.loc[
        df["anomaly_detected"] == 1,
        "anomaly_level"
    ] = "MEDIUM"

    # Extremely unusual observations get HIGH.
    high_threshold = df.loc[
        df["anomaly_detected"] == 1,
        "anomaly_score"
    ].quantile(0.25)

    df.loc[
        (df["anomaly_detected"] == 1)
        & (df["anomaly_score"] <= high_threshold),
        "anomaly_level"
    ] = "HIGH"

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("ISOLATION FOREST RESULTS")
    print("=" * 60)

    print(f"\nTotal transactions: {len(df):,}")

    print(
        f"Detected anomalies: "
        f"{df['anomaly_detected'].sum():,}"
    )

    print(
        f"Normal transactions: "
        f"{(df['anomaly_detected'] == 0).sum():,}"
    )

    print("\nAnomaly level distribution:")
    print(
        df["anomaly_level"]
        .value_counts()
        .to_string()
    )

    print("\nDetection by scenario:")

    scenario_summary = (
        df.groupby("scenario_type")
        .agg(
            transactions=("transaction_id", "count"),
            anomalies=("anomaly_detected", "sum"),
            average_anomaly_score=("anomaly_score", "mean"),
        )
        .sort_values("anomalies", ascending=False)
    )

    print(scenario_summary.to_string())

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()