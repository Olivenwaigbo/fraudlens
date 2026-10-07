from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RULE_FILE = BASE_DIR / "data" / "rule_detection_results.csv"
ANOMALY_FILE = BASE_DIR / "data" / "anomaly_detection_results.csv"
CLASSIFIER_FILE = BASE_DIR / "data" / "fraud_classifier_results.csv"
FEATURE_FILE = BASE_DIR / "data" / "fraud_features.csv"

OUTPUT_FILE = BASE_DIR / "data" / "risk_scoring_results.csv"


def min_max_scale(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


def main():

    print("Loading detection results...")

    rules = pd.read_csv(RULE_FILE)
    anomalies = pd.read_csv(ANOMALY_FILE)
    classifier = pd.read_csv(CLASSIFIER_FILE)

    print(f"Rule records: {len(rules):,}")
    print(f"Anomaly records: {len(anomalies):,}")
    print(f"Classifier records: {len(classifier):,}")

    # ---------------------------------------------------------------
    # Select columns from each detection layer
    # ---------------------------------------------------------------

    rule_columns = [
        "transaction_id",
        "rule_risk_score",
        "rule_detected",
        "rule_signals",
        "rule_reasons",
    ]

    anomaly_columns = [
        "transaction_id",
        "anomaly_prediction",
        "anomaly_score",
        "anomaly_detected",
        "anomaly_level",
    ]

    classifier_columns = [
        "transaction_id",
        "fraud_probability",
        "classifier_prediction",
    ]

    rules = rules[rule_columns]
    anomalies = anomalies[anomaly_columns]
    classifier = classifier[classifier_columns]

    # ---------------------------------------------------------------
    # Merge the three detection layers
    # ---------------------------------------------------------------

    df = rules.merge(
        anomalies,
        on="transaction_id",
        how="inner",
    )

    df = df.merge(
        classifier,
        on="transaction_id",
        how="inner",
    )

    print(f"\nMerged records: {len(df):,}")

    if len(df) != len(rules):
        raise ValueError(
            "Some transaction records were lost during merging."
        )

    # ---------------------------------------------------------------
    # Convert each detection signal to a 0–100 scale
    # ---------------------------------------------------------------

    # Rule engine already produces a 0–100 risk score.
    df["rule_component"] = (
        df["rule_risk_score"]
        .clip(0, 100)
    )

    # Isolation Forest:
    # Lower anomaly scores represent more unusual observations.
    anomaly_scaled = min_max_scale(
        -df["anomaly_score"]
    )

    df["anomaly_component"] = (
        anomaly_scaled * 100
    )

    # Logistic Regression probability is between 0 and 1.
    df["classifier_component"] = (
        df["fraud_probability"] * 100
    )

    # ---------------------------------------------------------------
    # Combined risk score
    # ---------------------------------------------------------------

    df["risk_score"] = (
        (df["rule_component"] * 0.40)
        + (df["anomaly_component"] * 0.25)
        + (df["classifier_component"] * 0.35)
    )

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    # ---------------------------------------------------------------
    # Risk levels
    # ---------------------------------------------------------------

    df["risk_level"] = "LOW"

    df.loc[
        df["risk_score"] >= 30,
        "risk_level"
    ] = "MEDIUM"

    df.loc[
        df["risk_score"] >= 60,
        "risk_level"
    ] = "HIGH"

    # ---------------------------------------------------------------
    # Investigation priority
    # ---------------------------------------------------------------

    df["investigation_priority"] = "ROUTINE"

    df.loc[
        df["risk_score"] >= 30,
        "investigation_priority"
    ] = "REVIEW"

    df.loc[
        df["risk_score"] >= 60,
        "investigation_priority"
    ] = "URGENT_REVIEW"

    # ---------------------------------------------------------------
    # Evidence summary
    # ---------------------------------------------------------------

    def build_evidence(row):

        evidence = []

        if row["rule_detected"] == 1:
            evidence.append("rule_signal")

        if row["anomaly_detected"] == 1:
            evidence.append("behavioral_anomaly")

        if row["classifier_prediction"] == 1:
            evidence.append("classifier_signal")

        if not evidence:
            return "none"

        return ", ".join(evidence)

    df["risk_evidence"] = df.apply(
        build_evidence,
        axis=1,
    )

    # ---------------------------------------------------------------
    # Add scenario information
    # ---------------------------------------------------------------

    source_features = pd.read_csv(
        FEATURE_FILE,
        usecols=[
            "transaction_id",
            "scenario_type",
            "fraud_label",
        ],
    )

    df = df.merge(
        source_features,
        on="transaction_id",
        how="left",
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("FRAUDLENS RISK SCORING")
    print("=" * 60)

    print("\nRisk level distribution:")
    print(
        df["risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nInvestigation priority:")
    print(
        df["investigation_priority"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nAverage risk score: "
        f"{df['risk_score'].mean():.2f}"
    )

    print(
        f"Maximum risk score: "
        f"{df['risk_score'].max():.2f}"
    )

    print("\nRisk scoring by scenario:")

    scenario_summary = (
        df.groupby("scenario_type")
        .agg(
            transactions=("transaction_id", "count"),
            average_risk_score=("risk_score", "mean"),
            high_risk=(
                "risk_level",
                lambda x: (x == "HIGH").sum(),
            ),
            medium_risk=(
                "risk_level",
                lambda x: (x == "MEDIUM").sum(),
            ),
        )
        .sort_values(
            "average_risk_score",
            ascending=False,
        )
    )

    print(
        scenario_summary.to_string()
    )

    print("\nHighest-risk transactions:")

    top_risk = (
        df[
            [
                "transaction_id",
                "scenario_type",
                "fraud_label",
                "risk_score",
                "risk_level",
                "risk_evidence",
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_risk.to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()