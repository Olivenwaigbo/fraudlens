# FraudLens — System Architecture

## Overview

FraudLens is an experimental payment fraud detection and risk intelligence platform.

The system follows a layered architecture:

Transaction Data
→ Feature Engineering
→ Risk Detection
→ Risk Scoring
→ Investigation
→ AI Explanation

---

## Architecture

```text
                  ┌─────────────────────┐
                  │  Synthetic Payment  │
                  │        Data         │
                  └──────────┬──────────┘
                             │
                             ↓
                  ┌─────────────────────┐
                  │ Feature Engineering │
                  └──────────┬──────────┘
                             │
              ┌──────────────┴──────────────┐
              ↓                             ↓
    ┌──────────────────┐          ┌──────────────────┐
    │ Rule-Based       │          │ Anomaly Detection│
    │ Risk Signals     │          │ Isolation Forest │
    └────────┬─────────┘          └────────┬─────────┘
             │                             │
             └──────────────┬──────────────┘
                            ↓
                  ┌─────────────────────┐
                  │ Fraud Classification│
                  │ Logistic Regression │
                  └──────────┬──────────┘
                             │
                             ↓
                  ┌─────────────────────┐
                  │    Risk Scoring     │
                  └──────────┬──────────┘
                             │
                             ↓
                  ┌─────────────────────┐
                  │   Investigation     │
                  └──────────┬──────────┘
                             │
                             ↓
                  ┌─────────────────────┐
                  │   AI Explanation    │
                  │  Groq / Llama Model │
                  └─────────────────────┘