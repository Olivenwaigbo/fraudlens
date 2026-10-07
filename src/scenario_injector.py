"""
FraudLens - Synthetic Fraud Scenario Injector

Creates a controlled test dataset by injecting suspicious behavioural
patterns into the clean synthetic transaction dataset.

The original transactions.csv is never modified.

Output:
    data/transactions_test.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
SCENARIOS_PER_TYPE = 30

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "transactions.csv"
OUTPUT_FILE = DATA_DIR / "transactions_test.csv"


# ---------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------

SCENARIO_TYPES = [
    "HIGH_VELOCITY",
    "UNUSUAL_AMOUNT",
    "NEW_DEVICE",
    "FAILED_ATTEMPTS",
    "NEW_LOCATION",
    "COMBINED_ANOMALY",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_unique_transaction_id(existing_ids, prefix="FRAUD"):
    """
    Generate a unique transaction ID.
    """
    counter = 1

    while True:
        transaction_id = f"{prefix}-{counter:06d}"

        if transaction_id not in existing_ids:
            return transaction_id

        counter += 1


def choose_base_rows(df, count, used_indexes, rng):
    """
    Select unique base transactions that have not already been used.
    """
    available = df.index.difference(list(used_indexes))

    if len(available) < count:
        raise ValueError("Not enough unused transactions available.")

    selected = rng.choice(available.to_numpy(), size=count, replace=False)

    return list(selected)


# ---------------------------------------------------------
# Scenario injection functions
# ---------------------------------------------------------

def inject_high_velocity(df, rows, rng):
    """
    Create transactions showing unusually high transaction velocity.
    """

    for idx in rows:
        df.loc[idx, "transactions_1h"] = int(rng.integers(12, 25))
        df.loc[idx, "amount_24h"] = round(
            float(df.loc[idx, "amount"]) * rng.uniform(8, 15),
            2,
        )
        df.loc[idx, "minutes_since_previous"] = int(
            rng.integers(1, 6)
        )

    return df


def inject_unusual_amount(df, rows, rng):
    """
    Create transactions where the amount is unusually large.
    """

    for idx in rows:
        original_amount = float(df.loc[idx, "amount"])

        df.loc[idx, "amount"] = round(
            max(original_amount * rng.uniform(8, 20), 150000),
            2,
        )

        df.loc[idx, "amount_24h"] = round(
            float(df.loc[idx, "amount"]) * rng.uniform(1.2, 2.5),
            2,
        )

    return df


def inject_new_device(df, rows, rng):
    """
    Create transactions from a device that is new for the customer.
    """

    device_types = ["Web", "Android", "iOS"]

    for idx in rows:
        current_device = df.loc[idx, "device_type"]

        alternatives = [
            device for device in device_types
            if device != current_device
        ]

        new_device = rng.choice(alternatives)

        df.loc[idx, "device_type"] = new_device
        df.loc[idx, "is_new_device"] = 1

    return df


def inject_failed_attempts(df, rows, rng):
    """
    Create transactions preceded by multiple failed attempts.
    """

    for idx in rows:
        df.loc[idx, "failed_attempts_24h"] = int(
            rng.integers(5, 12)
        )

        df.loc[idx, "transactions_1h"] = int(
            rng.integers(4, 10)
        )

    return df


def inject_new_location(df, rows, rng):
    """
    Create transactions from a location different from the
    customer's normal/home location.
    """

    locations = [
        "Lagos",
        "Abuja",
        "Kano",
        "Ibadan",
        "Port Harcourt",
        "Benin City",
        "Enugu",
        "Kaduna",
    ]

    for idx in rows:
        home_location = df.loc[idx, "home_location"]

        alternatives = [
            location
            for location in locations
            if location != home_location
        ]

        new_location = rng.choice(alternatives)

        df.loc[idx, "location"] = new_location
        df.loc[idx, "is_new_location"] = 1

    return df


def inject_combined_anomaly(df, rows, rng):
    """
    Create transactions containing several suspicious signals
    at the same time.
    """

    device_types = ["Web", "Android", "iOS"]

    locations = [
        "Lagos",
        "Abuja",
        "Kano",
        "Ibadan",
        "Port Harcourt",
        "Benin City",
        "Enugu",
        "Kaduna",
    ]

    for idx in rows:

        # 1. Very high transaction velocity
        df.loc[idx, "transactions_1h"] = int(
            rng.integers(15, 30)
        )

        # 2. Large transaction amount
        df.loc[idx, "amount"] = round(
            max(
                float(df.loc[idx, "amount"])
                * rng.uniform(10, 25),
                200000,
            ),
            2,
        )

        # 3. Large 24-hour transaction value
        df.loc[idx, "amount_24h"] = round(
            float(df.loc[idx, "amount"])
            * rng.uniform(2, 4),
            2,
        )

        # 4. Multiple failed attempts
        df.loc[idx, "failed_attempts_24h"] = int(
            rng.integers(7, 15)
        )

        # 5. Very short time since previous transaction
        df.loc[idx, "minutes_since_previous"] = int(
            rng.integers(1, 3)
        )

        # 6. New device
        current_device = df.loc[idx, "device_type"]

        alternatives = [
            device
            for device in device_types
            if device != current_device
        ]

        df.loc[idx, "device_type"] = rng.choice(alternatives)
        df.loc[idx, "is_new_device"] = 1

        # 7. New location
        home_location = df.loc[idx, "home_location"]

        alternatives = [
            location
            for location in locations
            if location != home_location
        ]

        df.loc[idx, "location"] = rng.choice(alternatives)
        df.loc[idx, "is_new_location"] = 1

    return df


# ---------------------------------------------------------
# Main process
# ---------------------------------------------------------

def main():

    print("Creating FraudLens suspicious transaction scenarios...")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Base dataset not found: {INPUT_FILE}"
        )

    # Load clean dataset
    df = pd.read_csv(INPUT_FILE)

    print(f"Base transactions: {len(df):,}")

    rng = np.random.default_rng(SEED)

    # Make a copy so the original dataset remains untouched
    test_df = df.copy()

    used_indexes = set()

    scenario_rows = []

    # -----------------------------------------------------
    # Inject each scenario type
    # -----------------------------------------------------

    for scenario in SCENARIO_TYPES:

        rows = choose_base_rows(
            test_df,
            SCENARIOS_PER_TYPE,
            used_indexes,
            rng,
        )

        used_indexes.update(rows)

        if scenario == "HIGH_VELOCITY":
            test_df = inject_high_velocity(
                test_df,
                rows,
                rng,
            )

        elif scenario == "UNUSUAL_AMOUNT":
            test_df = inject_unusual_amount(
                test_df,
                rows,
                rng,
            )

        elif scenario == "NEW_DEVICE":
            test_df = inject_new_device(
                test_df,
                rows,
                rng,
            )

        elif scenario == "FAILED_ATTEMPTS":
            test_df = inject_failed_attempts(
                test_df,
                rows,
                rng,
            )

        elif scenario == "NEW_LOCATION":
            test_df = inject_new_location(
                test_df,
                rows,
                rng,
            )

        elif scenario == "COMBINED_ANOMALY":
            test_df = inject_combined_anomaly(
                test_df,
                rows,
                rng,
            )

        # Mark the scenario
        for idx in rows:
            scenario_rows.append(
                {
                    "index": idx,
                    "scenario_type": scenario,
                }
            )

    # -----------------------------------------------------
    # Add scenario metadata
    # -----------------------------------------------------

    test_df["scenario_type"] = "BASELINE"
    test_df["fraud_label"] = 0

    for item in scenario_rows:
        idx = item["index"]

        test_df.loc[idx, "scenario_type"] = item[
            "scenario_type"
        ]

        test_df.loc[idx, "fraud_label"] = 1

    # -----------------------------------------------------
    # Create unique IDs for injected transactions
    # -----------------------------------------------------

    existing_ids = set(test_df["transaction_id"])

    for item in scenario_rows:

        idx = item["index"]

        original_id = test_df.loc[idx, "transaction_id"]

        # Keep the original transaction ID but make it clear
        # that this row belongs to the synthetic test scenario.
        new_id = f"{original_id}-S"

        if new_id in existing_ids:
            new_id = make_unique_transaction_id(
                existing_ids
            )

        test_df.loc[idx, "transaction_id"] = new_id

        existing_ids.add(new_id)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    test_df = test_df.reset_index(drop=True)

    # Ensure scenario labels are strings
    test_df["scenario_type"] = (
        test_df["scenario_type"].astype(str)
    )

    # Ensure fraud labels are integers
    test_df["fraud_label"] = (
        test_df["fraud_label"].astype(int)
    )

    # Sort by timestamp to preserve transaction chronology
    test_df["created_at"] = pd.to_datetime(
        test_df["created_at"]
    )

    test_df = test_df.sort_values(
        by=["created_at", "transaction_id"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    test_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("FraudLens test dataset created successfully.")
    print()

    print(f"Transactions: {len(test_df):,}")
    print(f"Columns: {len(test_df.columns)}")
    print(
        f"Suspicious transactions: "
        f"{test_df['fraud_label'].sum():,}"
    )
    print(
        f"Baseline transactions: "
        f"{(test_df['fraud_label'] == 0).sum():,}"
    )

    print()
    print("Scenario distribution:")
    print(
        test_df["scenario_type"]
        .value_counts()
        .to_string()
    )

    print()
    print("Fraud label distribution:")
    print(
        test_df["fraud_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()