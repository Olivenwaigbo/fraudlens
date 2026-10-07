"""
FraudLens - Synthetic Payment Dataset Generator

Generates a clean baseline payment transaction dataset.

The dataset is intentionally designed so that suspicious behavioural
signals are relatively uncommon in normal transactions.

Output:
    data/transactions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
NUM_TRANSACTIONS = 20_000
NUM_CUSTOMERS = 2_000
NUM_MERCHANTS = 1_000

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

OUTPUT_FILE = DATA_DIR / "transactions.csv"


# ---------------------------------------------------------
# Reference data
# ---------------------------------------------------------

PAYMENT_METHODS = [
    "Card",
    "Bank Transfer",
    "Wallet",
    "USSD",
]

PAYMENT_METHOD_PROBS = [
    0.45,
    0.25,
    0.15,
    0.15,
]

LOCATIONS = [
    "Lagos",
    "Abuja",
    "Kano",
    "Ibadan",
    "Port Harcourt",
    "Benin City",
    "Enugu",
    "Kaduna",
]

DEVICE_TYPES = [
    "Web",
    "Android",
    "iOS",
]

MERCHANT_CATEGORIES = [
    "Food & Delivery",
    "Utilities",
    "Transport",
    "Education",
    "Healthcare",
    "Entertainment",
    "Retail",
    "E-commerce",
]

PROVIDERS = [
    "Provider A",
    "Provider B",
    "Provider C",
    "Provider D",
]

BANKS = [
    "Bank Alpha",
    "Bank Beta",
    "Bank Gamma",
    "Bank Delta",
    "Bank Epsilon",
]


# ---------------------------------------------------------
# Main generator
# ---------------------------------------------------------

def main():

    print("Generating FraudLens synthetic payment dataset...")
    print()

    rng = np.random.default_rng(SEED)

    # -----------------------------------------------------
    # Customers
    # -----------------------------------------------------

    customers = [
        f"CUST-{i:05d}"
        for i in range(1, NUM_CUSTOMERS + 1)
    ]

    customer_home_locations = {
        customer: rng.choice(LOCATIONS)
        for customer in customers
    }

    customer_primary_devices = {
        customer: rng.choice(DEVICE_TYPES)
        for customer in customers
    }

    customer_account_age = {
        customer: int(
            rng.integers(30, 1_800)
        )
        for customer in customers
    }

    # -----------------------------------------------------
    # Generate customer transaction counts
    #
    # Every customer receives at least one transaction.
    # -----------------------------------------------------

    base_counts = np.ones(
        NUM_CUSTOMERS,
        dtype=int,
    )

    remaining = NUM_TRANSACTIONS - NUM_CUSTOMERS

    additional_counts = rng.multinomial(
        remaining,
        np.ones(NUM_CUSTOMERS) / NUM_CUSTOMERS,
    )

    transaction_counts = (
        base_counts + additional_counts
    )

    # -----------------------------------------------------
    # Build transactions
    # -----------------------------------------------------

    records = []

    global_counter = 1

    for customer_index, customer_id in enumerate(customers):

        customer_count = transaction_counts[
            customer_index
        ]

        home_location = customer_home_locations[
            customer_id
        ]

        primary_device = customer_primary_devices[
            customer_id
        ]

        account_age = customer_account_age[
            customer_id
        ]

        # Starting timestamp for customer
        current_time = pd.Timestamp(
            "2026-01-01 08:00:00"
        ) + pd.Timedelta(
            minutes=int(
                rng.integers(0, 60 * 24 * 14)
            )
        )

        previous_transaction_time = None

        for transaction_number in range(
            customer_count
        ):

            # -------------------------------------------------
            # Time between transactions
            #
            # Normal transactions generally occur hours/days
            # apart rather than seconds apart.
            # -------------------------------------------------

            if previous_transaction_time is None:

                minutes_since_previous = 999999

            else:

                minutes_since_previous = int(
                    rng.integers(60, 60 * 72)
                )

                current_time = (
                    previous_transaction_time
                    + pd.Timedelta(
                        minutes=minutes_since_previous
                    )
                )

            # -------------------------------------------------
            # Normal location behaviour
            #
            # Around 93% of baseline transactions occur from
            # the customer's normal/home location.
            # -------------------------------------------------

            if rng.random() < 0.93:

                location = home_location
                is_new_location = 0

            else:

                alternatives = [
                    value
                    for value in LOCATIONS
                    if value != home_location
                ]

                location = rng.choice(
                    alternatives
                )

                is_new_location = 1

            # -------------------------------------------------
            # Normal device behaviour
            #
            # Around 95% of baseline transactions use the
            # customer's primary device type.
            # -------------------------------------------------

            if rng.random() < 0.95:

                device_type = primary_device
                is_new_device = 0

            else:

                alternatives = [
                    value
                    for value in DEVICE_TYPES
                    if value != primary_device
                ]

                device_type = rng.choice(
                    alternatives
                )

                is_new_device = 1

            # -------------------------------------------------
            # Transaction amount
            #
            # Most transactions are relatively small.
            # Very large values are intentionally uncommon.
            # -------------------------------------------------

            amount = round(
                float(
                    rng.lognormal(
                        mean=np.log(8_000),
                        sigma=0.75,
                    )
                ),
                2,
            )

            # Keep baseline values within a sensible range.
            amount = min(
                max(amount, 500),
                120_000,
            )

            # -------------------------------------------------
            # Failed attempts
            #
            # Most normal transactions have zero failed
            # attempts. A small percentage have 1-2.
            # -------------------------------------------------

            failed_attempt_probability = rng.random()

            if failed_attempt_probability < 0.88:

                failed_attempts = 0

            elif failed_attempt_probability < 0.98:

                failed_attempts = int(
                    rng.integers(1, 3)
                )

            else:

                failed_attempts = int(
                    rng.integers(3, 5)
                )

            # -------------------------------------------------
            # Normal velocity
            #
            # Keep normal transactions mostly below the
            # suspicious high-velocity threshold.
            # -------------------------------------------------

            transactions_1h = int(
                rng.choice(
                    [1, 2, 3, 4, 5],
                    p=[
                        0.30,
                        0.30,
                        0.20,
                        0.15,
                        0.05,
                    ],
                )
            )

            # -------------------------------------------------
            # 24-hour amount
            # -------------------------------------------------

            historical_multiplier = float(
                rng.uniform(1.5, 5.0)
            )

            amount_24h = round(
                amount * historical_multiplier,
                2,
            )

            # -------------------------------------------------
            # Transaction ID
            # -------------------------------------------------

            transaction_id = (
                f"TXN-{global_counter:08d}"
            )

            merchant_id = (
                f"MER-{int(rng.integers(1, NUM_MERCHANTS + 1)):04d}"
            )

            # -------------------------------------------------
            # Payment attributes
            # -------------------------------------------------

            payment_method = rng.choice(
                PAYMENT_METHODS,
                p=PAYMENT_METHOD_PROBS,
            )

            provider = rng.choice(
                PROVIDERS
            )

            bank = rng.choice(
                BANKS
            )

            merchant_category = rng.choice(
                MERCHANT_CATEGORIES
            )

            # -------------------------------------------------
            # Previous transaction metadata
            # -------------------------------------------------

            if previous_transaction_time is None:

                previous_transaction_at = pd.Timestamp(
                    "1900-01-01 00:00:00"
                )

                previous_transaction_count = 0

            else:

                previous_transaction_at = (
                    previous_transaction_time
                )

                previous_transaction_count = (
                    transaction_number
                )

            # -------------------------------------------------
            # Create record
            # -------------------------------------------------

            records.append(
                {
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                    "merchant_id": merchant_id,
                    "merchant_category": merchant_category,
                    "provider": provider,
                    "bank": bank,
                    "payment_method": payment_method,
                    "currency": "NGN",
                    "amount": amount,
                    "created_at": current_time,
                    "location": location,
                    "device_type": device_type,
                    "account_age_days": account_age,
                    "home_location": home_location,
                    "primary_device_type": primary_device,
                    "is_new_device": is_new_device,
                    "is_new_location": is_new_location,
                    "previous_transaction_at": previous_transaction_at,
                    "minutes_since_previous": minutes_since_previous,
                    "previous_transaction_count": previous_transaction_count,
                    "failed_attempts_24h": failed_attempts,
                    "transactions_1h": transactions_1h,
                    "amount_24h": amount_24h,
                    "fraud_label": 0,
                }
            )

            previous_transaction_time = current_time

            global_counter += 1

    # ---------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(records)

    # Ensure timestamps are properly formatted
    df["created_at"] = pd.to_datetime(
        df["created_at"]
    )

    df["previous_transaction_at"] = pd.to_datetime(
        df["previous_transaction_at"]
    )

    # Sort chronologically
    df = df.sort_values(
        by=[
            "customer_id",
            "created_at",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    assert len(df) == NUM_TRANSACTIONS

    assert df["transaction_id"].is_unique

    assert df.isna().sum().sum() == 0

    assert df["fraud_label"].sum() == 0

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("Dataset generated successfully.")
    print(
        f"Transactions: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Fraud labels: {df['fraud_label'].sum():,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print("New device rate:")
    print(
        f"{df['is_new_device'].mean() * 100:.2f}%"
    )

    print()
    print("New location rate:")
    print(
        f"{df['is_new_location'].mean() * 100:.2f}%"
    )

    print()
    print("Failed attempts distribution:")
    print(
        df["failed_attempts_24h"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Transactions per hour distribution:")
    print(
        df["transactions_1h"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Dataset preview:")
    print(
        df.head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()