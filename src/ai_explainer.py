from pathlib import Path
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# FRAUDLENS AI EXPLAINER
# ============================================================
# Purpose:
# Generate a human-readable investigation summary from the
# FraudLens model outputs.
#
# Important:
# - FraudLens uses synthetic data.
# - Model outputs are investigative signals.
# - The system does NOT establish that fraud occurred.
# - The AI does NOT make operational decisions.
#
# Usage:
#   python src\ai_explainer.py                 (default transaction)
#   python src\ai_explainer.py TXN-00001903-S  (specific transaction)
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRANSACTION_FILE = BASE_DIR / "data" / "transactions_test.csv"
RULE_FILE = BASE_DIR / "data" / "rule_detection_results.csv"
ANOMALY_FILE = BASE_DIR / "data" / "anomaly_detection_results.csv"
CLASSIFIER_FILE = BASE_DIR / "data" / "fraud_classifier_results.csv"
RISK_FILE = BASE_DIR / "data" / "risk_scoring_results.csv"

OUTPUT_FILE = BASE_DIR / "data" / "ai_investigations.csv"

DEFAULT_TRANSACTION_ID = "TXN-00001903-S"

AI_MODEL = "openai/gpt-oss-120b"

SOURCE_AI = "ai_validated"
SOURCE_FALLBACK = "deterministic_fallback"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY was not found.")
    print("The deterministic fallback will be used.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ============================================================
# BASIC FORMATTING
# ============================================================

def clean_value(value):
    """Return a clean readable value."""

    if value is None or pd.isna(value):
        return "Not available"

    return str(value).strip()


def to_float(value, default=0.0):
    """Safely convert a value to float."""

    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def format_number(value, decimals=2):
    """Format a numeric value."""

    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return clean_value(value)


def format_integer(value):
    """Format a numeric value as a whole number."""

    try:
        return f"{int(float(value)):,}"
    except (ValueError, TypeError):
        return clean_value(value)


def format_indicator(value):
    """Convert binary/boolean indicators into Yes/No."""

    if value is None or pd.isna(value):
        return "Not available"

    try:
        numeric = int(float(value))

        if numeric == 1:
            return "Yes"

        if numeric == 0:
            return "No"

    except (ValueError, TypeError):
        pass

    text = str(value).strip().lower()

    if text in {"true", "yes"}:
        return "Yes"

    if text in {"false", "no"}:
        return "No"

    return str(value).strip()


def is_yes(value):
    return format_indicator(value) == "Yes"


# ============================================================
# RULE SIGNAL CLEANING
# ============================================================

def clean_rule_signals(value):
    """
    Normalize rule signal formatting.

    HIGH_VELOCITY;HIGH_VALUE;NEW_DEVICE,NEW_LOCATION
    becomes
    HIGH_VELOCITY, HIGH_VALUE, NEW_DEVICE, NEW_LOCATION
    """

    if value is None or pd.isna(value):
        return "None recorded"

    text = str(value).strip()

    if not text:
        return "None recorded"

    text = text.replace(";", ",").replace("|", ",")

    parts = []

    for part in text.split(","):
        part = part.strip()

        if part and part not in parts:
            parts.append(part)

    if not parts:
        return "None recorded"

    return ", ".join(parts)


def signal_list(signals_text):
    """Return the cleaned signal string as a Python list."""

    if signals_text == "None recorded":
        return []

    return [s.strip() for s in signals_text.split(",") if s.strip()]


# ============================================================
# AI TEXT CLEANING
# ============================================================

def clean_ai_text(text):
    """
    Light, safe cleanup of model output.

    NOTE: No comma or word-splitting rewrites here. A blanket
    comma rule would corrupt numbers such as 200,000.00.
    """

    if not text:
        return ""

    text = str(text).strip()

    # Remove markdown emphasis.
    text = text.replace("**", "").replace("__", "")

    # Collapse line breaks and repeated whitespace.
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    transactions = pd.read_csv(TRANSACTION_FILE)
    rules = pd.read_csv(RULE_FILE)
    anomalies = pd.read_csv(ANOMALY_FILE)
    classifier = pd.read_csv(CLASSIFIER_FILE)
    risk = pd.read_csv(RISK_FILE)

    return transactions, rules, anomalies, classifier, risk


# ============================================================
# FIND TRANSACTION
# ============================================================

def get_transaction_row(dataframe, transaction_id, filename):

    if "transaction_id" not in dataframe.columns:
        raise ValueError(f"{filename} does not contain transaction_id.")

    result = dataframe[dataframe["transaction_id"] == transaction_id]

    if result.empty:
        raise ValueError(f"{transaction_id} was not found in {filename}.")

    return result.iloc[0]


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence(transaction_id):

    (
        transactions,
        rules,
        anomalies,
        classifier,
        risk,
    ) = load_data()

    transaction = get_transaction_row(
        transactions, transaction_id, TRANSACTION_FILE.name
    )

    # Verify that the other model files contain the transaction.
    get_transaction_row(rules, transaction_id, RULE_FILE.name)
    get_transaction_row(anomalies, transaction_id, ANOMALY_FILE.name)
    get_transaction_row(classifier, transaction_id, CLASSIFIER_FILE.name)

    risk_result = get_transaction_row(risk, transaction_id, RISK_FILE.name)

    evidence = {

        # Transaction
        "transaction_id": transaction_id,
        "amount": transaction.get("amount"),
        "merchant_category": transaction.get("merchant_category"),
        "location": transaction.get("location"),
        "home_location": transaction.get("home_location"),
        "device_type": transaction.get("device_type"),
        "primary_device_type": transaction.get("primary_device_type"),
        "is_new_device": transaction.get("is_new_device"),
        "is_new_location": transaction.get("is_new_location"),
        "account_age_days": transaction.get("account_age_days"),
        "minutes_since_previous": transaction.get("minutes_since_previous"),
        "previous_transaction_count": transaction.get(
            "previous_transaction_count"
        ),
        "failed_attempts_24h": transaction.get("failed_attempts_24h"),
        "transactions_1h": transaction.get("transactions_1h"),
        "amount_24h": transaction.get("amount_24h"),

        # Rule engine
        "rule_risk_score": risk_result.get("rule_risk_score"),
        "rule_detected": risk_result.get("rule_detected"),
        "rule_signals": clean_rule_signals(risk_result.get("rule_signals")),
        "rule_reasons": clean_value(risk_result.get("rule_reasons")),

        # Anomaly detector
        "anomaly_prediction": risk_result.get("anomaly_prediction"),
        "anomaly_score": risk_result.get("anomaly_score"),
        "anomaly_detected": risk_result.get("anomaly_detected"),
        "anomaly_level": risk_result.get("anomaly_level"),

        # Supervised classifier
        "fraud_probability": risk_result.get("fraud_probability"),
        "classifier_prediction": risk_result.get("classifier_prediction"),

        # Combined risk
        "risk_score": risk_result.get("risk_score"),
        "risk_level": risk_result.get("risk_level"),
        "investigation_priority": risk_result.get("investigation_priority"),
        "risk_evidence": clean_value(risk_result.get("risk_evidence")),

        # Ground-truth fields (never shown to the AI or in the report)
        "fraud_label": risk_result.get("fraud_label"),
        "scenario_type": risk_result.get("scenario_type"),
    }

    return evidence


# ============================================================
# VALIDATE EVIDENCE
# ============================================================

def validate_evidence(evidence):

    required_fields = [
        "transaction_id",
        "amount",
        "transactions_1h",
        "minutes_since_previous",
        "failed_attempts_24h",
        "amount_24h",
        "is_new_device",
        "is_new_location",
        "device_type",
        "primary_device_type",
        "location",
        "home_location",
        "rule_risk_score",
        "rule_signals",
        "anomaly_score",
        "anomaly_level",
        "fraud_probability",
        "classifier_prediction",
        "risk_score",
        "risk_level",
        "investigation_priority",
    ]

    missing = []

    for field in required_fields:
        value = evidence.get(field)

        if value is None or pd.isna(value):
            missing.append(field)

    if missing:
        raise ValueError(
            "Critical FraudLens evidence is missing: " + ", ".join(missing)
        )


# ============================================================
# BUILD AI PROMPT
# ============================================================

def build_ai_prompt(evidence):

    return f"""
You are writing an investigation interpretation for FraudLens,
an experimental payment-risk investigation prototype using
synthetic data.

Write exactly TWO concise sentences.

Sentence 1:
Describe the observable indicators that contributed to the
elevated investigation risk. Use specific values from the
evidence (for example the number of failed attempts, the device
and location differences, or the transaction velocity) rather
than generic wording.

Sentence 2:
State that the signals support human-led review but do not
establish that a real-world fraud event occurred.

STRICT RULES:

- Use ONLY the evidence supplied below.
- Never say fraud occurred.
- Never call the transaction fraudulent.
- Never say fraud is confirmed.
- Never infer customer intent.
- Never invent facts.
- Do not recommend blocking, declining, freezing, reversing,
  approving, releasing, or contacting anyone.
- Do not mention OTP, 3DS, chargebacks, disputes, or merchant
  verification.
- Do not describe the raw anomaly score as high or low.
- If mentioning anomaly detection, say that the anomaly detector
  recorded {clean_value(evidence["anomaly_level"])} anomaly level.
- Do not describe the combined risk score as maximum.
- Do not describe the synthetic classifier probability as a
  real-world probability.
- Keep the language neutral and factual.
- Output plain text only, with no markdown.

EVIDENCE

Transaction: {evidence["transaction_id"]}
Amount: {format_number(evidence["amount"])} NGN
Transactions in past hour: {format_integer(evidence["transactions_1h"])}
Minutes since previous transaction: {format_number(evidence["minutes_since_previous"], 0)}
Previous transaction count: {format_integer(evidence["previous_transaction_count"])}
Failed attempts in past 24 hours: {format_integer(evidence["failed_attempts_24h"])}
Amount in past 24 hours: {format_number(evidence["amount_24h"])} NGN
New device: {format_indicator(evidence["is_new_device"])}
New location: {format_indicator(evidence["is_new_location"])}
Device: {clean_value(evidence["device_type"])}
Primary device: {clean_value(evidence["primary_device_type"])}
Location: {clean_value(evidence["location"])}
Home location: {clean_value(evidence["home_location"])}
Merchant category: {clean_value(evidence["merchant_category"])}
Account age: {format_integer(evidence["account_age_days"])} days
Rule signals: {evidence["rule_signals"]}
Rule risk score: {format_number(evidence["rule_risk_score"])}
Anomaly level: {clean_value(evidence["anomaly_level"])}
Raw anomaly score: {format_number(evidence["anomaly_score"], 6)}
Synthetic classifier probability: {format_number(evidence["fraud_probability"], 4)}
Classifier prediction: {clean_value(evidence["classifier_prediction"])}
Combined risk score: {format_number(evidence["risk_score"])}
Risk level: {clean_value(evidence["risk_level"])}
Investigation priority: {clean_value(evidence["investigation_priority"])}
""".strip()


# ============================================================
# VALIDATE AI INTERPRETATION
# ============================================================

def validate_ai_interpretation(text):

    if not text:
        print("AI validation rejected empty response.")
        return False

    lowered = text.lower()

    forbidden_phrases = [
        "fraud occurred",
        "fraudulent transaction",
        "confirmed fraud",
        "definitely fraud",
        "this is fraud",
        "criminal intent",
        "block the transaction",
        "decline the transaction",
        "freeze the account",
        "reverse the transaction",
        "approve the transaction",
        "release the transaction",
        "high anomaly score",
        "low anomaly score",
        "maximum",
    ]

    for phrase in forbidden_phrases:
        if phrase in lowered:
            print(f"AI validation rejected phrase: '{phrase}'")
            return False

    # The disclaimer sentence must actually be present.
    if "human" not in lowered:
        print("AI validation rejected response: no human-review statement.")
        return False

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]

    if len(sentences) > 3:
        print(
            "AI validation rejected response because it "
            f"contains {len(sentences)} sentences."
        )
        return False

    return True


# ============================================================
# DETERMINISTIC FALLBACK (evidence-driven, not hard-coded)
# ============================================================

def join_natural(items):
    """['a', 'b', 'c'] -> 'a, b, and c'"""

    if not items:
        return ""

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_indicator_phrases(evidence):
    """Turn the recorded evidence into specific, factual phrases."""

    signals = signal_list(evidence["rule_signals"])
    phrases = []

    if "HIGH_VELOCITY" in signals:
        phrases.append(
            f"{format_integer(evidence['transactions_1h'])} transactions "
            "in the past hour"
        )

    if "RAPID_TRANSACTION" in signals:
        minutes = format_number(evidence["minutes_since_previous"], 0)
        unit = "minute" if minutes == "1" else "minutes"
        phrases.append(f"only {minutes} {unit} since the previous transaction")

    if "FAILED_ATTEMPTS" in signals:
        phrases.append(
            f"{format_integer(evidence['failed_attempts_24h'])} failed "
            "attempts in the past 24 hours"
        )

    if "NEW_DEVICE" in signals or is_yes(evidence["is_new_device"]):
        phrases.append(
            f"a new device ({clean_value(evidence['device_type'])} versus "
            f"primary {clean_value(evidence['primary_device_type'])})"
        )

    if "NEW_LOCATION" in signals or is_yes(evidence["is_new_location"]):
        phrases.append(
            f"a new location ({clean_value(evidence['location'])} versus "
            f"home {clean_value(evidence['home_location'])})"
        )

    if "HIGH_VALUE" in signals:
        phrases.append(
            f"a high transaction amount ({format_number(evidence['amount'])} NGN)"
        )

    return phrases


def deterministic_interpretation(evidence):

    phrases = build_indicator_phrases(evidence)

    if phrases:
        first = (
            "The transaction shows elevated investigation risk based on "
            f"{join_natural(phrases)}."
        )
    else:
        first = (
            "The transaction shows elevated investigation risk based on "
            f"the recorded rule signals ({evidence['rule_signals']})."
        )

    second = (
        f"The anomaly detector recorded {clean_value(evidence['anomaly_level'])} "
        "anomaly level, and these signals support human-led review but do "
        "not establish that a real-world fraud event occurred."
    )

    return f"{first} {second}"


# ============================================================
# GENERATE AI INTERPRETATION
# ============================================================

def generate_ai_interpretation(evidence):
    """
    Returns (text, source) where source is SOURCE_AI or SOURCE_FALLBACK.
    """

    def fallback(reason):
        print(reason)
        print("Using deterministic fallback.")
        return deterministic_interpretation(evidence), SOURCE_FALLBACK

    if client is None:
        return fallback("Groq client unavailable.")

    prompt = build_ai_prompt(evidence)

    try:

        # gpt-oss models are reasoning models: hidden reasoning tokens
        # count against the completion limit. A small max_tokens value
        # can be fully consumed by reasoning, leaving an EMPTY answer.
        # Fix: low reasoning effort + a generous completion limit.
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful financial-risk investigation "
                        "writing assistant. Follow the evidence and "
                        "constraints exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=1024,
            reasoning_effort="low",
        )

        choice = response.choices[0]
        raw_text = choice.message.content
        finish_reason = getattr(choice, "finish_reason", "unknown")

        if not raw_text or not raw_text.strip():
            usage = getattr(response, "usage", None)
            print(
                "AI returned empty content "
                f"(finish_reason={finish_reason}, usage={usage})."
            )

        ai_text = clean_ai_text(raw_text)

        if validate_ai_interpretation(ai_text):
            print("AI interpretation validated.")
            return ai_text, SOURCE_AI

        print(f"Rejected AI text was: {ai_text!r}")
        return fallback("AI interpretation failed validation.")

    except Exception as error:
        return fallback(f"AI interpretation failed: {error}")


# ============================================================
# OBSERVED PATTERN (deterministic, clearly labelled hypothesis)
# ============================================================

def build_pattern_note(evidence):

    age = to_float(evidence["account_age_days"])
    changes = []

    if is_yes(evidence["is_new_device"]):
        changes.append("device")

    if is_yes(evidence["is_new_location"]):
        changes.append("location")

    failed = to_float(evidence["failed_attempts_24h"])
    txn_1h = to_float(evidence["transactions_1h"])

    if age >= 180 and changes and (failed >= 3 or txn_1h >= 10):
        return (
            f"The account is established ({format_integer(age)} days old), "
            f"yet this transaction combines a {' and '.join(changes)} change "
            f"with {format_integer(failed)} failed attempts in 24 hours and "
            f"{format_integer(txn_1h)} transactions in the past hour. "
            "A reviewer may wish to test the hypothesis that the account "
            "was accessed by someone other than the account holder. "
            "This is a hypothesis to test, not a finding."
        )

    return ""


# ============================================================
# BUILD FINAL INVESTIGATION
# ============================================================

def build_final_investigation(evidence, ai_interpretation, source):

    signals = signal_list(evidence["rule_signals"])

    if source == SOURCE_AI:
        interpretation_label = (
            f"AI interpretation (model: {AI_MODEL}, validated):"
        )
    else:
        interpretation_label = (
            "Signal interpretation (generated by deterministic template; "
            "AI output was unavailable or rejected):"
        )

    lines = [

        "INVESTIGATION SUMMARY",
        "",
        (
            f"FraudLens flagged transaction {evidence['transaction_id']} "
            "for human investigation. The recorded combined risk score is "
            f"{format_number(evidence['risk_score'])}, with risk level "
            f"{clean_value(evidence['risk_level'])} and investigation "
            f"priority {clean_value(evidence['investigation_priority'])}."
        ),
        "",
        interpretation_label,
        ai_interpretation,
    ]

    pattern_note = build_pattern_note(evidence)

    if pattern_note:
        lines += [
            "",
            "OBSERVED PATTERN",
            "",
            pattern_note,
        ]

    lines += [

        "",
        "RISK EVIDENCE",
        "",
        f"- Transaction amount: {format_number(evidence['amount'])} NGN",
        f"- Transactions in the past hour: {format_integer(evidence['transactions_1h'])}",
        f"- Minutes since previous transaction: {format_number(evidence['minutes_since_previous'], 0)}",
        f"- Previous transaction count: {format_integer(evidence['previous_transaction_count'])}",
        f"- Failed attempts in the past 24 hours: {format_integer(evidence['failed_attempts_24h'])}",
        f"- Total amount in the past 24 hours: {format_number(evidence['amount_24h'])} NGN",
        f"- New device indicator: {format_indicator(evidence['is_new_device'])}",
        f"- New location indicator: {format_indicator(evidence['is_new_location'])}",
        f"- Device type: {clean_value(evidence['device_type'])} (primary: {clean_value(evidence['primary_device_type'])})",
        f"- Location: {clean_value(evidence['location'])} (home: {clean_value(evidence['home_location'])})",
        f"- Merchant category: {clean_value(evidence['merchant_category'])}",
        f"- Account age: {format_integer(evidence['account_age_days'])} days",
        f"- Rule engine signals: {evidence['rule_signals']}",

        "",
        "MODEL SIGNALS",
        "",
        (
            f"- Rule engine: score {format_number(evidence['rule_risk_score'])} "
            f"with {len(signals)} recorded signal(s), listed above."
        ),
        (
            f"- Isolation Forest: anomaly_detected = "
            f"{format_indicator(evidence['anomaly_detected'])}, "
            f"anomaly_level = {clean_value(evidence['anomaly_level'])}, "
            f"anomaly_score = {format_number(evidence['anomaly_score'], 6)}."
        ),
        (
            f"- Logistic regression (synthetic labelled data): "
            f"fraud_probability output = {format_number(evidence['fraud_probability'], 4)}, "
            f"classifier prediction = {clean_value(evidence['classifier_prediction'])}."
        ),
        (
            f"- Combined score: {format_number(evidence['risk_score'])}, "
            f"risk level {clean_value(evidence['risk_level'])}, "
            f"priority {clean_value(evidence['investigation_priority'])}."
        ),

        "",
        "WHAT IS UNKNOWN",
        "",
        "- Whether the customer authorized the transaction.",
        (
            "- The purpose of the transaction beyond the recorded merchant "
            f"category ({clean_value(evidence['merchant_category'])})."
        ),
        "- Whether relevant context exists outside the evidence supplied to FraudLens.",
        "- Whether the model signals correspond to an actual fraud event.",

        "",
        "RECOMMENDED NEXT STEPS",
        "",
        "1. Review the failed-attempt log for the past 24 hours (timing, channel, and error type).",
        "2. Compare the device and IP history for this account against the new device and location.",
        "3. Review the sequence of recent transactions, including amounts, timing, and merchants.",
        "4. Verify the activity with the account holder through a trusted, previously registered channel, following the payment or fraud-review process.",
        "5. Record the reviewer's decision and rationale separately from these model outputs.",

        "",
        "CONFIDENCE NOTE",
        "",
        (
            "FraudLens is an experimental project operating on synthetic "
            "data. Its rule engine, anomaly detector, supervised classifier, "
            "and combined score produce investigative signals, not proof, "
            "and should be interpreted within a human-led review process. "
            "Classifier probabilities at or near 1.0 on synthetic data can "
            "reflect easily separable synthetic patterns and should not be "
            "read as real-world certainty."
        ),
    ]

    return "\n".join(lines).strip()


# ============================================================
# VALIDATE FINAL INVESTIGATION
# ============================================================

def validate_final_investigation(text):

    if not text:
        return False

    required_sections = [
        "INVESTIGATION SUMMARY",
        "RISK EVIDENCE",
        "MODEL SIGNALS",
        "WHAT IS UNKNOWN",
        "RECOMMENDED NEXT STEPS",
        "CONFIDENCE NOTE",
    ]

    for section in required_sections:
        if section not in text:
            return False

    lowered = text.lower()

    forbidden_phrases = [
        "fraud occurred",
        "confirmed fraud",
        "definitely fraud",
        "this is fraud",
        "fraudulent transaction",
    ]

    for phrase in forbidden_phrases:
        if phrase in lowered:
            return False

    return True


# ============================================================
# SAVE INVESTIGATION
# ============================================================

def save_investigation(
    transaction_id,
    evidence,
    ai_interpretation,
    interpretation_source,
    final_investigation,
):

    new_row = pd.DataFrame(
        [
            {
                "transaction_id": transaction_id,
                "risk_score": evidence["risk_score"],
                "risk_level": evidence["risk_level"],
                "investigation_priority": evidence["investigation_priority"],
                "interpretation_source": interpretation_source,
                "ai_interpretation": ai_interpretation,
                "full_investigation": final_investigation,
            }
        ]
    )

    if OUTPUT_FILE.exists():

        existing = pd.read_csv(OUTPUT_FILE)

        existing = existing[existing["transaction_id"] != transaction_id]

        updated = pd.concat([existing, new_row], ignore_index=True)

    else:
        updated = new_row

    updated.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"AI investigation saved for {transaction_id}.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def investigate_transaction(transaction_id):

    print(f"Generating AI investigation for {transaction_id}...")

    evidence = build_evidence(transaction_id)

    validate_evidence(evidence)

    ai_interpretation, source = generate_ai_interpretation(evidence)

    final_investigation = build_final_investigation(
        evidence, ai_interpretation, source
    )

    if not validate_final_investigation(final_investigation):
        raise ValueError("Final investigation failed validation.")

    save_investigation(
        transaction_id,
        evidence,
        ai_interpretation,
        source,
        final_investigation,
    )

    print()
    print(final_investigation)
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    target_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSACTION_ID

    investigate_transaction(target_id)