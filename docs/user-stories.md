# FraudLens — User Stories

## Payment/Fraud Analyst

### US-001 — View Risk Overview

As a payment/fraud analyst,

I want to see an overview of transaction risk,

so that I can understand the current risk distribution.

**Acceptance Criteria**

- Total transactions are displayed.
- Risk levels are displayed.
- High-risk transactions can be identified.
- Detected anomalies are displayed.

---

### US-002 — Investigate a Transaction

As a payment/fraud analyst,

I want to search for a transaction,

so that I can investigate its risk signals.

**Acceptance Criteria**

- A transaction can be selected.
- Transaction details are displayed.
- Risk score is displayed.
- Risk level is displayed.
- Triggered signals are displayed.

---

### US-003 — Understand Why a Transaction Was Flagged

As a payment/fraud analyst,

I want to understand the evidence behind a risk assessment,

so that I can decide what requires further investigation.

**Acceptance Criteria**

- Relevant behavioral signals are displayed.
- Model output is displayed.
- The source transaction data is traceable.
- Unknown information is clearly identified.

---

### US-004 — Review Behavioral History

As a payment/fraud analyst,

I want to see relevant transaction history,

so that I can understand the customer's recent behavior.

**Acceptance Criteria**

- Recent transactions are displayed.
- Transaction amounts are visible.
- Transaction frequency can be assessed.
- Relevant behavioral features are displayed.

---

### US-005 — Receive an AI Investigation Summary

As a payment/fraud analyst,

I want an AI-generated explanation of the available evidence,

so that I can review a transaction more efficiently.

**Acceptance Criteria**

- The explanation uses available evidence.
- Unsupported claims are avoided.
- Unknown information is explicitly identified.
- The AI does not declare a transaction fraudulent.

---

## Product Manager

### US-006 — Evaluate Detection Performance

As a product manager,

I want to see model evaluation metrics,

so that I can understand how the detection approach performs.

**Acceptance Criteria**

- Precision is displayed.
- Recall is displayed.
- F1 score is displayed.
- ROC-AUC is displayed.
- Confusion matrix is available.

---

### US-007 — Understand Risk Patterns

As a product manager,

I want to see the most common risk signals,

so that I can identify recurring behavioral patterns.

**Acceptance Criteria**

- Risk signals are summarized.
- High-frequency signals can be identified.
- Risk distribution can be reviewed.

---

## Engineering / Risk Operations

### US-008 — Trace a Risk Assessment

As an engineering or risk operations user,

I want to understand how a risk assessment was generated,

so that I can investigate model behavior.

**Acceptance Criteria**

- Input features are visible.
- Rule-based signals are visible.
- Model outputs are visible.
- Risk score inputs are traceable.

---

### US-009 — Identify Potential False Positives

As an engineering or risk operations user,

I want to review transactions flagged as high risk,

so that I can investigate potentially incorrect classifications.

**Acceptance Criteria**

- High-risk transactions can be filtered.
- Model predictions are visible.
- Relevant behavioral features are visible.
- The underlying synthetic label can be reviewed during evaluation.

---

## System

### US-010 — Protect Sensitive Data

As a system owner,

I want the prototype to use synthetic data,

so that the project does not require real customer payment information.

**Acceptance Criteria**

- No real customer data is required.
- Synthetic transactions can reproduce risk scenarios.
- Data generation is reproducible.
