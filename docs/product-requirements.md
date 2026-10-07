# FraudLens — Product Requirements Document

## 1. Product Overview

**Product Name:** FraudLens

**Product Category:** Payment Fraud Detection & Risk Intelligence

**Product Type:** Experimental payment risk intelligence platform

### Product Description

FraudLens analyzes synthetic payment transaction data to identify suspicious behavioral patterns, generate risk signals, and support human investigation.

The system combines rule-based detection, anomaly detection, supervised machine learning, risk scoring, and AI-assisted investigation.

FraudLens is an experimental decision-support system. It does not autonomously determine that a transaction is fraudulent, block transactions, or make financial decisions.

---

## 2. Problem Statement

Payment platforms process large volumes of transactions. Individual transactions may appear normal while combinations of behavioral signals can indicate that a transaction requires further investigation.

Examples of potentially suspicious patterns include:

- unusually high transaction amounts
- unusually high transaction velocity
- multiple failed attempts followed by a successful transaction
- transactions from a new device
- transactions from an unusual location
- combinations of several unusual behavioral signals

Payment and fraud operations teams need tools that help them identify these patterns and understand the evidence behind them.

FraudLens explores how data science and AI can support this investigation process.

---

## 3. Target Users

### Primary User — Payment/Fraud Analyst

Needs to:

- identify suspicious transactions
- understand why transactions were flagged
- review behavioral signals
- prioritize transactions for investigation
- understand the evidence behind a risk score

### Secondary User — Product Manager

Needs to:

- understand emerging payment risk patterns
- monitor detection performance
- evaluate fraud detection features
- identify opportunities for improving payment products

### Secondary User — Engineering / Risk Operations

Needs to:

- understand model signals
- investigate false positives
- trace how risk scores were generated
- monitor detection behavior

---

## 4. Product Hypothesis

If payment teams can combine transaction-level signals, behavioral features, anomaly detection, and explainable risk scoring in one investigation workflow, they can identify potentially suspicious payment activity more efficiently.

FraudLens will test this hypothesis using synthetic payment data.

---

## 5. MVP Goals

The MVP should allow users to:

1. View synthetic payment transactions.
2. Generate behavioral features from transaction history.
3. Detect suspicious transaction patterns.
4. Identify behavioral anomalies.
5. Classify transactions using a supervised machine learning model.
6. Generate a risk score.
7. Assign a risk level.
8. View the signals responsible for a risk assessment.
9. Investigate an individual transaction.
10. Generate an AI-assisted investigation explanation.
11. Monitor detection and model performance through a dashboard.

---

## 6. Non-Goals

FraudLens will not:

- process real money
- connect to real banks
- connect to real payment processors
- use real customer data
- automatically block transactions
- automatically approve transactions
- make autonomous financial decisions
- declare that a person committed fraud
- replace human fraud investigators
- perform real-world criminal investigations

---

## 7. Core Features

### 7.1 Transaction Monitoring

Display synthetic payment transactions and their relevant behavioral information.

---

### 7.2 Behavioral Feature Engineering

Generate features such as:

- transaction velocity
- transaction amount deviation
- failed attempts
- transaction frequency
- new device indicator
- new location indicator
- account age
- recent transaction volume
- recent transaction value

---

### 7.3 Rule-Based Risk Signals

Identify predefined suspicious patterns.

Examples:

- high transaction velocity
- unusually large transaction
- multiple failed attempts
- new device
- new location
- combined behavioral signals

---

### 7.4 Anomaly Detection

Use unsupervised machine learning to identify transactions that behave unusually compared with the broader transaction population.

Initial algorithm:

**Isolation Forest**

---

### 7.5 Fraud Classification

Use supervised machine learning on the synthetic fraud labels.

Initial model:

**Logistic Regression**

The model will be evaluated using appropriate classification metrics.

---

### 7.6 Risk Scoring

Combine available risk signals and model outputs into an experimental risk score.

Risk levels:

- LOW
- MEDIUM
- HIGH

The risk score is an investigation signal and not a definitive fraud determination.

---

### 7.7 Transaction Investigation

Users should be able to select an individual transaction and view:

- transaction details
- behavioral features
- risk score
- risk level
- triggered risk signals
- anomaly result
- model prediction
- relevant transaction history

---

### 7.8 AI Investigation

The AI layer will transform established evidence into a structured investigation summary.

The AI must:

- use only available evidence
- avoid inventing facts
- clearly distinguish known information from unknown information
- explain why the transaction was flagged
- recommend operational investigation steps
- avoid declaring a transaction fraudulent
- avoid making autonomous financial decisions

---

### 7.9 Operations Dashboard

The dashboard should provide:

- total transactions
- high-risk transactions
- medium-risk transactions
- detected anomalies
- risk distribution
- top risk signals
- fraud model performance
- transaction investigation

---

## 8. Synthetic Fraud Scenarios

FraudLens will use controlled synthetic scenarios.

### Scenario 1 — High Velocity

Multiple transactions occur within a short period.

### Scenario 2 — Unusual Amount

A transaction is substantially larger than the customer's historical transaction pattern.

### Scenario 3 — New Device

A transaction occurs using a device that has not previously been associated with the customer.

### Scenario 4 — Multiple Failed Attempts

Several failed transactions occur before a successful transaction.

### Scenario 5 — New Location

A transaction occurs from a location that differs from the customer's established transaction pattern.

### Scenario 6 — Combined Behavioral Anomaly

Multiple risk signals occur together.

The scenarios are synthetic and are intended only for model experimentation and product demonstration.

---

## 9. Success Metrics

### Detection Metrics

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Product Metrics

- percentage of transactions successfully assigned a risk level
- percentage of high-risk transactions with explainable evidence
- investigation workflow completion
- percentage of risk signals traceable to source data

### AI Quality

- evidence adherence
- unsupported claim rate
- clarity of investigation summary
- identification of unknown information

---

## 10. Product Principles

### Evidence First

Every risk assessment should be traceable to available transaction evidence.

### Human in the Loop

FraudLens supports investigation rather than replacing human judgment.

### Traceability

Users should be able to understand how a risk assessment was generated.

### Explainability

Model outputs should be presented in a form that users can understand.

### Privacy by Design

The project uses synthetic data and does not require real customer information.

### Responsible Experimentation

Fraud detection results are risk signals for investigation, not definitive determinations of wrongdoing.

---

## 11. Technical Approach

### Data

Synthetic payment transaction data generated with Python.

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Isolation Forest
- Logistic Regression

### Application

- Streamlit

### API

- FastAPI

### AI

- Groq
- Llama model

### Storage

CSV-based experimental data pipeline.

---

## 12. Future Opportunities

Potential future extensions include:

- real-time transaction scoring
- graph-based fraud detection
- customer behavioral profiles
- merchant risk intelligence
- device intelligence
- model monitoring
- explainable machine learning
- case management
- analyst feedback loops
- fraud pattern discovery