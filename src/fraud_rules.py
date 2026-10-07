"""
FraudLens - Rule-Based Fraud Detection

Applies transparent behavioural rules to payment transactions.

Input:
    data/fraud_features.csv

Output:
    data/rule_detection_results.csv
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "fraud_features.csv"
OUTPUT_FILE = DATA_DIR / "rule_detection_results.csv"


# ---------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------

RULE_DEFINITIONS = {
    "HIGH_VELOCITY": {
        "column": "high_velocity_flag",
        "points": 25,
        "reason": (
            "Unusually high transaction velocity detected."
        ),
    },

    "HIGH_VALUE": {
        "column": "high_value_flag",
        "points": 20,
        "reason": (
            "Transaction amount is unusually high."
        ),
    },

    "NEW_DEVICE": {
        "column": "new_device_flag",
        "points": 15,
        "reason": (
            "Transaction was made from a new device."
        ),
    },

    "NEW_LOCATION": {
        "column": "new_location_flag",
        "points": 15,
        "reason": (
            "Transaction occurred from a new location."
        ),
    },

    "FAILED_ATTEMPTS": {
        "column": "failed_attempt_flag",
        "points": 20,
        "reason": (
            "Multiple failed payment attempts detected."
        ),
    },

    "RAPID_TRANSACTION": {
        "column": "rapid_transaction_flag",
        "points": 15,
        "reason": (
            "Transaction occurred shortly after "
            "the previous transaction."
        ),
    },

    "MULTIPLE_SIGNALS": {
        "column": "multiple_risk_signals",
        "points": 10,
        "reason": (
            "Multiple independent behavioural risk "
            "signals are present."
        ),
    },
}


# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------

def classify_risk(score):
    """
    Convert a numeric risk score into a risk level.
    """

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# Apply rules
# ---------------------------------------------------------

def apply_rules(df):
    """
    Apply transparent fraud detection rules.
    """

    data = df.copy()

    # -----------------------------------------------------
    # Score
    # -----------------------------------------------------

    data["rule_risk_score"] = 0

    # Store triggered rule names
    data["rule_signals"] = ""

    # Store human-readable explanations
    data["rule_reasons"] = ""

    # -----------------------------------------------------
    # Apply each rule
    # -----------------------------------------------------

    for rule_name, rule in RULE_DEFINITIONS.items():

        column = rule["column"]
        points = rule["points"]
        reason = rule["reason"]

        triggered = (
            data[column] == 1
        )

        # Add score
        data.loc[
            triggered,
            "rule_risk_score"
        ] += points

        # Add rule name
        data.loc[
            triggered,
            "rule_signals"
        ] = (
            data.loc[
                triggered,
                "rule_signals"
            ]
            + rule_name
            + "; "
        )

        # Add explanation
        data.loc[
            triggered,
            "rule_reasons"
        ] = (
            data.loc[
                triggered,
                "rule_reasons"
            ]
            + reason
            + " "
        )

    # -----------------------------------------------------
    # Cap score
    # -----------------------------------------------------

    data["rule_risk_score"] = (
        data["rule_risk_score"]
        .clip(upper=100)
    )

    # -----------------------------------------------------
    # Risk level
    # -----------------------------------------------------

    data["rule_risk_level"] = (
        data["rule_risk_score"]
        .apply(classify_risk)
    )

    # -----------------------------------------------------
    # Rule detection flag
    # -----------------------------------------------------

    data["rule_detected"] = (
        data["rule_risk_score"] >= 30
    ).astype(int)

    return data


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(
        "Running FraudLens rule-based fraud detection..."
    )

    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    # Load features
    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input transactions: {len(df):,}"
    )

    # Apply rules
    results = apply_rules(df)

    # Save
    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print(
        "Rule detection completed successfully."
    )

    print()

    print(
        "Risk level distribution:"
    )

    print(
        results[
            "rule_risk_level"
        ]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"],
            fill_value=0,
        )
        .to_string()
    )

    print()

    print(
        "Detected transactions:"
    )

    print(
        results[
            "rule_detected"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Average rule risk score:"
    )

    print(
        f"{results['rule_risk_score'].mean():.2f}"
    )

    print()

    print(
        "Suspicious scenario detection:"
    )

    scenario_summary = (
        results
        .groupby("scenario_type")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            detected=(
                "rule_detected",
                "sum",
            ),
            average_score=(
                "rule_risk_score",
                "mean",
            ),
        )
        .sort_values(
            "average_score",
            ascending=False,
        )
    )

    print(
        scenario_summary.to_string()
    )

    print()

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()