"""
FraudLens - Feature Engineering

Transforms raw synthetic payment transactions into model-ready
fraud detection features.

Input:
    data/transactions_test.csv

Output:
    data/fraud_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "transactions_test.csv"
OUTPUT_FILE = DATA_DIR / "fraud_features.csv"


# ---------------------------------------------------------
# Main feature engineering pipeline
# ---------------------------------------------------------

def create_features(df):
    """
    Create fraud detection features from raw transaction data.
    """

    data = df.copy()

    # -----------------------------------------------------
    # Datetime conversion
    # -----------------------------------------------------

    data["created_at"] = pd.to_datetime(
        data["created_at"]
    )

    data["previous_transaction_at"] = pd.to_datetime(
        data["previous_transaction_at"]
    )

    # -----------------------------------------------------
    # Transaction amount features
    # -----------------------------------------------------

    data["amount_log"] = np.log1p(
        data["amount"]
    )

    data["amount_to_24h_ratio"] = (
        data["amount"]
        / data["amount_24h"].replace(0, np.nan)
    )

    data["amount_to_24h_ratio"] = (
        data["amount_to_24h_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # -----------------------------------------------------
    # Velocity features
    # -----------------------------------------------------

    data["high_velocity_flag"] = (
        data["transactions_1h"] >= 10
    ).astype(int)

    data["rapid_transaction_flag"] = (
        data["minutes_since_previous"] <= 5
    ).astype(int)

    data["high_24h_activity_flag"] = (
        data["amount_24h"] >= 100000
    ).astype(int)

    # -----------------------------------------------------
    # Failed attempt features
    # -----------------------------------------------------

    data["failed_attempt_flag"] = (
        data["failed_attempts_24h"] >= 3
    ).astype(int)

    data["high_failed_attempt_flag"] = (
        data["failed_attempts_24h"] >= 7
    ).astype(int)

    # -----------------------------------------------------
    # Device features
    # -----------------------------------------------------

    data["new_device_flag"] = (
        data["is_new_device"] == 1
    ).astype(int)

    # -----------------------------------------------------
    # Location features
    # -----------------------------------------------------

    data["new_location_flag"] = (
        data["is_new_location"] == 1
    ).astype(int)

    # -----------------------------------------------------
    # Account age features
    # -----------------------------------------------------

    data["new_account_flag"] = (
        data["account_age_days"] <= 30
    ).astype(int)

    data["mature_account_flag"] = (
        data["account_age_days"] >= 365
    ).astype(int)

    # -----------------------------------------------------
    # Behavioural risk signal
    # -----------------------------------------------------

    data["behavioural_signal_count"] = (
        data["high_velocity_flag"]
        + data["rapid_transaction_flag"]
        + data["failed_attempt_flag"]
        + data["new_device_flag"]
        + data["new_location_flag"]
    )

    data["multiple_risk_signals"] = (
        data["behavioural_signal_count"] >= 2
    ).astype(int)

    data["strong_risk_signal"] = (
        data["behavioural_signal_count"] >= 3
    ).astype(int)

    # -----------------------------------------------------
    # Transaction history features
    # -----------------------------------------------------

    data["has_transaction_history"] = (
        data["previous_transaction_count"] > 0
    ).astype(int)

    data["previous_transaction_flag"] = (
        data["previous_transaction_count"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Time-based features
    # -----------------------------------------------------

    data["hour"] = (
        data["created_at"].dt.hour
    )

    data["day_of_week"] = (
        data["created_at"].dt.dayofweek
    )

    data["weekend_flag"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["night_transaction_flag"] = (
        (data["hour"] < 6)
        | (data["hour"] >= 23)
    ).astype(int)

    # -----------------------------------------------------
    # Amount risk bands
    # -----------------------------------------------------

    data["high_value_flag"] = (
        data["amount"] >= 100000
    ).astype(int)

    data["very_high_value_flag"] = (
        data["amount"] >= 200000
    ).astype(int)

    # -----------------------------------------------------
    # Combined behavioural indicators
    # -----------------------------------------------------

    data["velocity_and_failed_attempts"] = (
        (
            data["transactions_1h"] >= 10
        )
        & (
            data["failed_attempts_24h"] >= 3
        )
    ).astype(int)

    data["new_device_and_location"] = (
        (
            data["is_new_device"] == 1
        )
        & (
            data["is_new_location"] == 1
        )
    ).astype(int)

    data["high_value_and_new_device"] = (
        (
            data["amount"] >= 100000
        )
        & (
            data["is_new_device"] == 1
        )
    ).astype(int)

    data["high_value_and_new_location"] = (
        (
            data["amount"] >= 100000
        )
        & (
            data["is_new_location"] == 1
        )
    ).astype(int)

    # -----------------------------------------------------
    # Missing value protection
    # -----------------------------------------------------

    numeric_columns = data.select_dtypes(
        include=["number"]
    ).columns

    data[numeric_columns] = (
        data[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return data


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Creating FraudLens fraud detection features...")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    # Load test dataset
    df = pd.read_csv(INPUT_FILE)

    print(
        f"Input transactions: {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns)}"
    )

    # Create features
    feature_df = create_features(df)

    # Save
    feature_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    original_columns = len(df.columns)
    feature_columns = len(feature_df.columns)

    added_columns = (
        feature_columns - original_columns
    )

    print()
    print("Feature engineering completed successfully.")
    print()

    print(
        f"Transactions: {len(feature_df):,}"
    )

    print(
        f"Original columns: {original_columns}"
    )

    print(
        f"Final columns: {feature_columns}"
    )

    print(
        f"Features added: {added_columns}"
    )

    print(
        f"Missing values: "
        f"{feature_df.isna().sum().sum()}"
    )

    print()

    print("Risk signal distribution:")

    print(
        feature_df[
            "behavioural_signal_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("Suspicious scenario signals:")

    signal_columns = [
        "high_velocity_flag",
        "high_value_flag",
        "new_device_flag",
        "new_location_flag",
        "failed_attempt_flag",
        "multiple_risk_signals",
        "strong_risk_signal",
    ]

    for column in signal_columns:

        print(
            f"{column}: "
            f"{feature_df[column].sum():,}"
        )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()