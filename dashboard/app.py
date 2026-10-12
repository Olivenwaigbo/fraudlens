import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudLens | Payment Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Model used for the AI follow-up chat, served by Groq.
# Override with the FRAUDLENS_AI_MODEL env var / Streamlit secret if needed.
AI_MODEL = os.getenv("FRAUDLENS_AI_MODEL", "openai/gpt-oss-20b")

# ============================================================
# PALETTE
# ============================================================

PURPLE_900 = "#2E1065"
PURPLE_700 = "#6D28D9"
PURPLE_600 = "#7C3AED"
PURPLE_500 = "#8B5CF6"
PURPLE_300 = "#C4B5FD"
PURPLE_100 = "#EDE9FE"
PURPLE_50 = "#FAF7FF"

LEVEL_COLORS = {
    "HIGH": PURPLE_900,
    "MEDIUM": PURPLE_500,
    "LOW": PURPLE_300,
}
PURPLE_SCALE = [[0, PURPLE_100], [0.5, PURPLE_500], [1, PURPLE_900]]


# ============================================================
# STYLING
# ============================================================

st.markdown(
    f"""
    <style>
        .stApp {{ background: #ffffff; }}
        .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PURPLE_50} 0%, {PURPLE_100} 100%);
            border-right: 1px solid {PURPLE_100};
        }}

        /* Hero banner */
        .hero {{
            background: linear-gradient(120deg, {PURPLE_900} 0%, {PURPLE_700} 55%, {PURPLE_500} 100%);
            color: white;
            padding: 1.6rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 30px rgba(109, 40, 217, 0.25);
        }}
        .hero h1 {{ color: white; margin: 0; font-size: 2.1rem; font-weight: 750; }}
        .hero p {{ color: {PURPLE_100}; margin: 0.3rem 0 0 0; font-size: 1rem; }}

        .section-title {{
            font-size: 1.2rem; font-weight: 700; color: {PURPLE_900};
            margin: 1.4rem 0 0.7rem 0; padding-left: 0.7rem;
            border-left: 4px solid {PURPLE_600};
        }}

        /* KPI cards */
        .kpi {{
            background: white; border: 1px solid {PURPLE_100};
            border-top: 4px solid {PURPLE_600};
            border-radius: 12px; padding: 1rem 1.1rem;
            box-shadow: 0 2px 10px rgba(124, 58, 237, 0.07);
            transition: transform .15s ease, box-shadow .15s ease;
        }}
        .kpi:hover {{ transform: translateY(-3px); box-shadow: 0 8px 22px rgba(124, 58, 237, 0.18); }}
        .kpi-label {{ color: #6b7280; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }}
        .kpi-value {{ color: {PURPLE_900}; font-size: 1.9rem; font-weight: 750; line-height: 1.2; }}
        .kpi-sub {{ color: {PURPLE_600}; font-size: .8rem; }}

        /* Badges */
        .badge {{
            display: inline-block; padding: .2rem .7rem; border-radius: 999px;
            font-weight: 700; font-size: .8rem; letter-spacing: .04em;
        }}
        .badge-HIGH {{ background: {PURPLE_900}; color: white; }}
        .badge-MEDIUM {{ background: {PURPLE_500}; color: white; }}
        .badge-LOW {{ background: {PURPLE_100}; color: {PURPLE_700}; }}

        .signal-box {{
            background: {PURPLE_50}; border: 1px solid {PURPLE_100};
            border-radius: 10px; padding: .8rem 1rem; margin-bottom: .6rem;
        }}
        .signal-box strong {{ color: {PURPLE_700}; }}

        .confidence-box {{
            background: {PURPLE_50}; border-left: 5px solid {PURPLE_600};
            padding: 1rem 1.2rem; border-radius: 8px; margin: 1rem 0;
        }}

        div[data-testid="stMetric"] {{
            background: white; border: 1px solid {PURPLE_100};
            padding: .9rem; border-radius: 12px;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 999px; border: 1px solid {PURPLE_300};
            color: {PURPLE_700}; background: white; font-weight: 600;
            transition: all .15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: {PURPLE_600}; color: white; border-color: {PURPLE_600};
        }}

        .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
        .stTabs [aria-selected="true"] {{ color: {PURPLE_700}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    return (
        pd.read_csv(DATA_DIR / "transactions.csv"),
        pd.read_csv(DATA_DIR / "fraud_features.csv"),
        pd.read_csv(DATA_DIR / "rule_detection_results.csv"),
        pd.read_csv(DATA_DIR / "anomaly_detection_results.csv"),
        pd.read_csv(DATA_DIR / "fraud_classifier_results.csv"),
        pd.read_csv(DATA_DIR / "risk_scoring_results.csv"),
        pd.read_csv(DATA_DIR / "ai_investigations.csv"),
    )


try:
    (transactions, features, rules, anomalies,
     classifier, risk, investigations) = load_data()
except Exception as e:
    st.error(f"Unable to load FraudLens data: {e}")
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

for _df in [transactions, features, rules, anomalies, classifier, risk, investigations]:
    if "transaction_id" in _df.columns:
        _df["transaction_id"] = _df["transaction_id"].astype(str)

if "created_at" in transactions.columns:
    transactions["created_at"] = pd.to_datetime(transactions["created_at"], errors="coerce")

dashboard = risk.copy()

transaction_columns = [
    "transaction_id", "customer_id", "merchant_id", "merchant_category",
    "provider", "bank", "payment_method", "currency", "amount", "created_at",
    "location", "device_type", "account_age_days", "home_location",
    "primary_device_type", "is_new_device", "is_new_location",
    "previous_transaction_at", "minutes_since_previous",
    "previous_transaction_count", "failed_attempts_24h", "transactions_1h",
    "amount_24h",
]
transaction_columns = [c for c in transaction_columns if c in transactions.columns]

dashboard = dashboard.merge(
    transactions[transaction_columns], on="transaction_id",
    how="left", suffixes=("", "_transaction"),
)

ai_columns = [
    "transaction_id", "investigation", "combined_risk_score",
    "combined_risk_level", "investigation_priority", "ai_interpretation",
    "full_investigation", "interpretation_source",
]
ai_columns = [c for c in ai_columns if c in investigations.columns]

dashboard = dashboard.merge(
    investigations[ai_columns], on="transaction_id",
    how="left", suffixes=("", "_ai"),
)

for col in [
    "amount", "risk_score", "rule_risk_score", "anomaly_score",
    "fraud_probability", "rule_component", "anomaly_component",
    "classifier_component", "combined_risk_score",
]:
    if col in dashboard.columns:
        dashboard[col] = pd.to_numeric(dashboard[col], errors="coerce")

if "created_at" in dashboard.columns:
    dashboard["created_at"] = pd.to_datetime(dashboard["created_at"], errors="coerce")

if "risk_level" in dashboard.columns:
    dashboard["risk_level"] = dashboard["risk_level"].astype(str).str.upper()


# ============================================================
# HELPERS
# ============================================================

def format_currency(value, currency=""):
    if pd.isna(value):
        return "N/A"
    try:
        return f"{currency} {float(value):,.2f}".strip()
    except (ValueError, TypeError):
        return str(value)


def safe_text(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "Not available"
    text = str(value).strip()
    return text or "Not available"


def kpi(label, value, sub=""):
    return (
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def badge(level):
    level = str(level).upper()
    if level not in LEVEL_COLORS:
        return safe_text(level)
    return f'<span class="badge badge-{level}">{level}</span>'


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def display_signal(label, value):
    st.markdown(
        f'<div class="signal-box"><strong>{label}</strong><br>{value}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=PURPLE_900),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor=PURPLE_100)
    fig.update_yaxes(gridcolor=PURPLE_100)
    return fig


def gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score) if pd.notna(score) else 0,
            number=dict(font=dict(color=PURPLE_900)),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=PURPLE_700),
                bgcolor="white",
                steps=[
                    dict(range=[0, 33], color=PURPLE_100),
                    dict(range=[33, 66], color=PURPLE_300),
                    dict(range=[66, 100], color="#A78BFA"),
                ],
            ),
        )
    )
    return style_fig(fig, 250)


def component_radar(record):
    cats = ["Rules", "Anomaly", "Classifier"]
    vals = [
        record.get("rule_component", 0),
        record.get("anomaly_component", 0),
        record.get("classifier_component", 0),
    ]
    vals = [0 if pd.isna(v) else float(v) for v in vals]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1],
            fill="toself", line=dict(color=PURPLE_700),
            fillcolor="rgba(139, 92, 246, 0.35)",
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return style_fig(fig, 250)


# ============================================================
# AI FOLLOW-UP ENGINE
# ============================================================

AI_SYSTEM_PROMPT = """You are the FraudLens Investigation Assistant, helping a human
fraud analyst review payment transactions. FraudLens is an experimental project on
SYNTHETIC data. Its rule engine, anomaly detector, classifier and combined risk score
produce investigative signals, not proof of fraud.

Rules:
- Answer only from the context provided below. If something is not in the context, say so.
- Be concise, concrete and neutral. Never accuse a customer of fraud; talk about risk signals.
- When useful, suggest practical next investigation steps for a human reviewer.
- Note that classifier probabilities near 1.0 on synthetic data may reflect easily
  separable patterns, not real-world certainty.

CONTEXT:
"""


def transaction_context(record):
    fields = [
        "transaction_id", "amount", "currency", "customer_id", "merchant_id",
        "merchant_category", "provider", "bank", "payment_method", "location",
        "home_location", "device_type", "primary_device_type", "is_new_device",
        "is_new_location", "account_age_days", "minutes_since_previous",
        "previous_transaction_count", "failed_attempts_24h", "transactions_1h",
        "amount_24h", "scenario_type", "risk_score", "risk_level",
        "investigation_priority", "rule_risk_score", "rule_detected",
        "rule_signals", "anomaly_score", "anomaly_detected", "anomaly_level",
        "fraud_probability", "classifier_prediction", "rule_component",
        "anomaly_component", "classifier_component", "risk_evidence",
        "ai_interpretation",
    ]
    lines = [f"{f}: {safe_text(record.get(f))}" for f in fields if f in record.index]
    full = record.get("full_investigation")
    if isinstance(full, str) and full.strip():
        lines.append("full_investigation: " + full.strip()[:3000])
    return "\n".join(lines)


def portfolio_context(df):
    lines = [f"Transactions in current filter: {len(df)}"]
    if "risk_level" in df.columns:
        lines.append("Risk level counts: " + df["risk_level"].value_counts().to_dict().__str__())
    if "risk_score" in df.columns:
        lines.append(f"Average risk score: {df['risk_score'].mean():.2f}")
        cols = [c for c in ["transaction_id", "amount", "risk_score", "risk_level",
                            "scenario_type", "investigation_priority"] if c in df.columns]
        lines.append("Top 10 by risk score:\n" +
                     df.sort_values("risk_score", ascending=False).head(10)[cols].to_string(index=False))
    if "scenario_type" in df.columns:
        lines.append("Scenario counts: " + df["scenario_type"].value_counts().to_dict().__str__())
    return "\n".join(lines)


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def local_answer(question, record):
    """Offline fallback used when no API key is configured."""
    q = question.lower()
    if record is None:
        return ("Portfolio-level questions need the live AI assistant. Add an "
                "`GROQ_API_KEY` (env var or Streamlit secrets) to enable it.")

    score = record.get("risk_score")
    level = safe_text(record.get("risk_level"))
    comps = {
        "rule engine": record.get("rule_component"),
        "anomaly detector": record.get("anomaly_component"),
        "classifier": record.get("classifier_component"),
    }
    comps = {k: v for k, v in comps.items() if pd.notna(v)}
    top = max(comps, key=comps.get) if comps else None

    if any(w in q for w in ["next", "do", "action", "recommend", "step"]):
        return (
            "Suggested review steps:\n"
            "1. Verify the customer's recent activity and whether the device/location change is expected.\n"
            "2. Check failed attempts and short-window velocity for card-testing patterns.\n"
            "3. Contact the customer through a verified channel if the amount is unusual.\n"
            "4. Document the outcome so the models can be tuned against reviewer decisions."
        )
    if any(w in q for w in ["device", "location", "new"]):
        return (
            f"New device: **{safe_text(record.get('is_new_device'))}**, "
            f"new location: **{safe_text(record.get('is_new_location'))}**. "
            f"Transaction location: {safe_text(record.get('location'))} "
            f"(home: {safe_text(record.get('home_location'))})."
        )
    if any(w in q for w in ["rule", "signal", "trigger"]):
        return f"Rule signals: {safe_text(record.get('rule_signals'))}."
    if any(w in q for w in ["confiden", "sure", "reliable", "trust"]):
        return (
            f"Classifier probability is {safe_text(record.get('fraud_probability'))}. "
            "On synthetic data, values near 1.0 can reflect easily separable patterns, "
            "so treat this as a signal for human review, not certainty."
        )
    return (
        f"This transaction scored **{score:.1f}** ({level}). "
        + (f"The strongest contributor is the **{top}**. " if top else "")
        + f"Evidence: {safe_text(record.get('risk_evidence'))}"
    )


def ask_ai(history, record, scope_df):
    key = get_api_key()
    if not key:
        return local_answer(history[-1]["content"], record) + (
            "\n\n_Offline mode: set `GROQ_API_KEY` for full AI answers._"
        )
    try:
        from groq import Groq

        client = Groq(api_key=key)
        context = transaction_context(record) if record is not None else portfolio_context(scope_df)
        response = client.chat.completions.create(
            model=AI_MODEL,
            max_tokens=800,
            temperature=0.2,
            messages=(
                [{"role": "system", "content": AI_SYSTEM_PROMPT + context}]
                + [{"role": m["role"], "content": m["content"]} for m in history]
            ),
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return "The AI assistant returned an empty answer.\n\n" + local_answer(
                history[-1]["content"], record
            )
        return answer
    except Exception as e:
        return f"The AI assistant could not respond ({e}).\n\n" + local_answer(
            history[-1]["content"], record
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ FraudLens")
    st.caption("Payment Risk & Investigation Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Investigations", "Transaction Explorer", "Model Signals", "AI Explainer"],
    )

    st.divider()
    st.markdown("### Filters")

    risk_levels = ["All"]
    if "risk_level" in dashboard.columns:
        risk_levels += sorted(dashboard["risk_level"].dropna().unique().tolist())
    selected_risk = st.selectbox("Risk level", risk_levels)

    priorities = ["All"]
    if "investigation_priority" in dashboard.columns:
        priorities += sorted(dashboard["investigation_priority"].dropna().astype(str).unique().tolist())
    selected_priority = st.selectbox("Investigation priority", priorities)

    scenarios = ["All"]
    if "scenario_type" in dashboard.columns:
        scenarios += sorted(dashboard["scenario_type"].dropna().astype(str).unique().tolist())
    selected_scenario = st.selectbox("Scenario", scenarios)

    score_range = None
    if "risk_score" in dashboard.columns and dashboard["risk_score"].notna().any():
        lo = float(dashboard["risk_score"].min())
        hi = float(dashboard["risk_score"].max())
        if lo < hi:
            score_range = st.slider("Risk score range", lo, hi, (lo, hi))

    amount_range = None
    if "amount" in dashboard.columns and dashboard["amount"].notna().any():
        alo = float(dashboard["amount"].min())
        ahi = float(dashboard["amount"].max())
        if alo < ahi:
            amount_range = st.slider("Amount range", alo, ahi, (alo, ahi))


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = dashboard.copy()

if selected_risk != "All":
    filtered = filtered[filtered["risk_level"] == selected_risk]
if selected_priority != "All":
    filtered = filtered[filtered["investigation_priority"].astype(str) == selected_priority]
if selected_scenario != "All":
    filtered = filtered[filtered["scenario_type"].astype(str) == selected_scenario]
if score_range:
    filtered = filtered[filtered["risk_score"].between(*score_range)]
if amount_range:
    filtered = filtered[filtered["amount"].between(*amount_range)]

with st.sidebar:
    st.divider()
    st.caption(f"Showing **{len(filtered):,}** of {len(dashboard):,} transactions")
    st.download_button(
        "⬇️ Download filtered data",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="fraudlens_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ FraudLens</h1>
        <p>Payment Risk & Investigation Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("⚠️ Confidence & Model Interpretation Note"):
    st.markdown(
        """
        FraudLens is an experimental project built on synthetic data.
        Its rule engine, anomaly detector, supervised classifier, and
        combined risk score generate investigative signals, not
        definitive proof of fraud.

        Results should therefore be interpreted within a human-led
        review process.

        Classifier probabilities at or near 1.0 on synthetic data may
        reflect easily separable patterns in the dataset and should
        not be interpreted as real-world certainty or production-level
        model confidence.
        """
    )

if filtered.empty:
    st.warning("No transactions match the selected filters. Try widening them in the sidebar.")
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    section("Risk Overview")

    levels = filtered["risk_level"] if "risk_level" in filtered.columns else pd.Series(dtype=str)
    total = len(filtered)
    high = int((levels == "HIGH").sum())
    medium = int((levels == "MEDIUM").sum())
    avg_risk = filtered["risk_score"].mean() if "risk_score" in filtered.columns else 0
    urgent = 0
    if "investigation_priority" in filtered.columns:
        urgent = int(filtered["investigation_priority"].astype(str).str.upper().str.contains("URGENT").sum())
    exposure = (
        filtered.loc[levels == "HIGH", "amount"].sum()
        if "amount" in filtered.columns else 0
    )

    k = st.columns(5)
    k[0].markdown(kpi("Transactions", f"{total:,}", "in current view"), unsafe_allow_html=True)
    k[1].markdown(kpi("High risk", f"{high:,}", f"{high / total:.1%} of view"), unsafe_allow_html=True)
    k[2].markdown(kpi("Medium risk", f"{medium:,}", f"{medium / total:.1%} of view"), unsafe_allow_html=True)
    k[3].markdown(kpi("Avg risk score", f"{avg_risk:.1f}", "out of 100"), unsafe_allow_html=True)
    k[4].markdown(kpi("Urgent reviews", f"{urgent:,}", f"High-risk value {exposure:,.0f}"), unsafe_allow_html=True)

    section("Distribution")
    left, right = st.columns(2)

    with left:
        counts = levels.value_counts().reset_index()
        counts.columns = ["Risk Level", "Transactions"]
        fig = px.pie(
            counts, names="Risk Level", values="Transactions", hole=0.55,
            color="Risk Level", color_discrete_map=LEVEL_COLORS,
            title="Risk level mix",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with right:
        if "investigation_priority" in filtered.columns:
            pr = filtered["investigation_priority"].astype(str).value_counts().reset_index()
            pr.columns = ["Priority", "Transactions"]
            fig = px.bar(
                pr, x="Transactions", y="Priority", orientation="h",
                color="Transactions", color_continuous_scale=PURPLE_SCALE,
                title="Investigation priority",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered, x="risk_score", nbins=20,
            color="risk_level", color_discrete_map=LEVEL_COLORS,
            title="Combined risk score distribution",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with right:
        if "amount" in filtered.columns:
            hover = [c for c in ["transaction_id", "scenario_type"] if c in filtered.columns]
            fig = px.scatter(
                filtered, x="amount", y="risk_score",
                color="risk_level", color_discrete_map=LEVEL_COLORS,
                hover_data=hover, opacity=0.75,
                title="Amount vs risk score (hover for details)",
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

    if "created_at" in filtered.columns and filtered["created_at"].notna().any():
        section("Risk over time")
        trend = (
            filtered.dropna(subset=["created_at"])
            .assign(day=lambda d: d["created_at"].dt.date)
            .groupby("day")
            .agg(avg_risk=("risk_score", "mean"), transactions=("transaction_id", "count"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_bar(x=trend["day"], y=trend["transactions"], name="Transactions",
                    marker_color=PURPLE_100)
        fig.add_scatter(x=trend["day"], y=trend["avg_risk"], name="Avg risk score",
                        yaxis="y2", line=dict(color=PURPLE_700, width=3))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    if "scenario_type" in filtered.columns:
        section("Scenarios by average risk")
        sc = (
            filtered.groupby("scenario_type")
            .agg(avg_risk=("risk_score", "mean"), count=("transaction_id", "count"))
            .reset_index().sort_values("avg_risk")
        )
        fig = px.bar(sc, x="avg_risk", y="scenario_type", orientation="h",
                     color="avg_risk", color_continuous_scale=PURPLE_SCALE,
                     hover_data=["count"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    section("Highest-Risk Transactions")
    cols = [c for c in ["transaction_id", "amount", "risk_score", "risk_level",
                        "investigation_priority", "fraud_probability", "anomaly_score"]
            if c in filtered.columns]
    st.dataframe(
        filtered.sort_values("risk_score", ascending=False).head(10)[cols],
        use_container_width=True, hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.1f"),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "fraud_probability": st.column_config.NumberColumn(format="%.4f"),
            "anomaly_score": st.column_config.NumberColumn(format="%.4f"),
        },
    )


# ============================================================
# INVESTIGATIONS
# ============================================================

elif page == "Investigations":

    section("Investigation Queue")
    st.write("Transactions are ordered by combined risk score to support human-led investigation.")

    c1, c2 = st.columns([2, 1])
    search = c1.text_input("🔎 Search by transaction ID", placeholder="e.g. TXN-1042")
    top_n = c2.slider("Rows to show", 10, max(len(filtered), 11), min(100, len(filtered)))

    cols = [c for c in ["transaction_id", "amount", "risk_score", "risk_level",
                        "investigation_priority", "rule_component", "anomaly_component",
                        "classifier_component", "scenario_type"] if c in filtered.columns]

    queue = filtered.sort_values("risk_score", ascending=False)
    if search:
        queue = queue[queue["transaction_id"].str.contains(search, case=False, na=False)]
    queue = queue.head(top_n)[cols]

    st.caption(f"{len(queue):,} cases in queue")
    st.dataframe(
        queue, use_container_width=True, hide_index=True, height=520,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.1f"),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "rule_component": st.column_config.NumberColumn("Rules", format="%.2f"),
            "anomaly_component": st.column_config.NumberColumn("Anomaly", format="%.2f"),
            "classifier_component": st.column_config.NumberColumn("Classifier", format="%.2f"),
        },
    )

    if not queue.empty:
        section("Case spotlight")
        spotlight = st.selectbox("Pick a case to preview", queue["transaction_id"].tolist())
        rec = filtered[filtered["transaction_id"] == spotlight].iloc[0]
        a, b = st.columns([1, 1])
        with a:
            st.plotly_chart(gauge(rec.get("risk_score")), use_container_width=True)
        with b:
            st.plotly_chart(component_radar(rec), use_container_width=True)
        st.info("Open **AI Explainer** to ask follow-up questions about this case.")


# ============================================================
# TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":

    section("Transaction Explorer")

    ids = filtered["transaction_id"].dropna().astype(str).tolist()
    selected = st.selectbox("Select a transaction", ids)
    record = filtered[filtered["transaction_id"] == selected].iloc[0]

    st.divider()

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge(record.get("risk_score")), use_container_width=True)
    with g2:
        st.plotly_chart(component_radar(record), use_container_width=True)
    with g3:
        st.metric("Transaction ID", safe_text(record.get("transaction_id")))
        st.metric("Amount", format_currency(record.get("amount"), record.get("currency", "")))
        st.markdown("**Risk level**")
        st.markdown(badge(record.get("risk_level")), unsafe_allow_html=True)

    tab_details, tab_behaviour, tab_signals, tab_evidence = st.tabs(
        ["📄 Details", "🧭 Behaviour", "📡 Detection signals", "🧾 Evidence"]
    )

    with tab_details:
        details = {
            "Customer ID": "customer_id", "Merchant ID": "merchant_id",
            "Merchant Category": "merchant_category", "Provider": "provider",
            "Bank": "bank", "Payment Method": "payment_method",
            "Currency": "currency", "Location": "location",
            "Device Type": "device_type",
        }
        d1, d2 = st.columns(2)
        for i, (label, col) in enumerate(details.items()):
            (d1 if i % 2 == 0 else d2).write(f"**{label}:** {safe_text(record.get(col))}")

    with tab_behaviour:
        behaviour = {
            "New Device": "is_new_device", "New Location": "is_new_location",
            "Account Age (days)": "account_age_days",
            "Minutes Since Previous": "minutes_since_previous",
            "Previous Transactions": "previous_transaction_count",
            "Failed Attempts (24h)": "failed_attempts_24h",
            "Transactions (1h)": "transactions_1h", "Amount (24h)": "amount_24h",
        }
        b1, b2 = st.columns(2)
        for i, (label, col) in enumerate(behaviour.items()):
            (b1 if i % 2 == 0 else b2).write(f"**{label}:** {safe_text(record.get(col))}")

    with tab_signals:
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("#### Rule Engine")
            display_signal("Risk Score", safe_text(record.get("rule_risk_score")))
            display_signal("Detected", safe_text(record.get("rule_detected")))
            display_signal("Signals", safe_text(record.get("rule_signals")))
        with s2:
            st.markdown("#### Anomaly Detector")
            display_signal("Anomaly Score", safe_text(record.get("anomaly_score")))
            display_signal("Detected", safe_text(record.get("anomaly_detected")))
            display_signal("Anomaly Level", safe_text(record.get("anomaly_level")))
        with s3:
            st.markdown("#### Supervised Classifier")
            display_signal("Fraud Probability", safe_text(record.get("fraud_probability")))
            display_signal("Prediction", safe_text(record.get("classifier_prediction")))
            display_signal("Classifier Component", safe_text(record.get("classifier_component")))

    with tab_evidence:
        evidence = record.get("risk_evidence")
        if pd.isna(evidence) or not str(evidence).strip():
            st.info("No combined risk evidence was recorded for this transaction.")
        else:
            st.info(str(evidence))
        st.caption("Want to dig deeper? Ask follow-up questions in the **AI Explainer** page.")


# ============================================================
# MODEL SIGNALS
# ============================================================

elif page == "Model Signals":

    section("Model Signals")
    st.write("FraudLens combines three independent detection approaches into a combined investigative risk score.")

    comp_cols = {
        "Rules": "rule_component",
        "Anomaly": "anomaly_component",
        "Classifier": "classifier_component",
    }
    present = {k: v for k, v in comp_cols.items() if v in filtered.columns}

    k = st.columns(len(present) or 1)
    for col, (label, name) in zip(k, present.items()):
        col.markdown(kpi(f"Avg {label} component", f"{filtered[name].mean():.2f}"), unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        section("Detection rates")
        rates = {}
        for label, column in {
            "Rule Detected": "rule_detected",
            "Anomaly Detected": "anomaly_detected",
            "Classifier Prediction": "classifier_prediction",
        }.items():
            if column in filtered.columns:
                vals = filtered[column].astype(str).str.lower()
                rates[label] = vals.isin(["true", "1", "yes", "fraud", "high"]).mean() * 100
        if rates:
            rdf = pd.DataFrame({"Model": list(rates), "Detection rate (%)": list(rates.values())})
            fig = px.bar(rdf, x="Model", y="Detection rate (%)",
                         color="Detection rate (%)", color_continuous_scale=PURPLE_SCALE)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with right:
        section("Average contribution by risk level")
        if present and "risk_level" in filtered.columns:
            avg = filtered.groupby("risk_level")[list(present.values())].mean().reset_index()
            avg = avg.rename(columns={v: k for k, v in present.items()})
            melted = avg.melt(id_vars="risk_level", var_name="Component", value_name="Average")
            fig = px.bar(melted, x="risk_level", y="Average", color="Component",
                         barmode="stack",
                         color_discrete_sequence=[PURPLE_900, PURPLE_500, PURPLE_300])
            st.plotly_chart(style_fig(fig), use_container_width=True)

    section("Do the models agree?")
    if {"anomaly_score", "fraud_probability"}.issubset(filtered.columns):
        hover = [c for c in ["transaction_id", "scenario_type"] if c in filtered.columns]
        fig = px.scatter(
            filtered, x="anomaly_score", y="fraud_probability",
            color="risk_score", color_continuous_scale=PURPLE_SCALE,
            hover_data=hover, opacity=0.8,
            title="Anomaly score vs classifier probability (colour = combined risk)",
        )
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        st.caption(
            "Points where one model is high and the other low are worth a manual look: "
            "they show cases where a single approach would have missed the signal."
        )

    section("Top 15 transactions: what drove the score")
    if present:
        top = filtered.sort_values("risk_score", ascending=False).head(15)
        melted = top.melt(
            id_vars="transaction_id", value_vars=list(present.values()),
            var_name="Component", value_name="Value",
        )
        melted["Component"] = melted["Component"].map({v: k for k, v in present.items()})
        fig = px.bar(melted, x="Value", y="transaction_id", color="Component",
                     orientation="h",
                     color_discrete_sequence=[PURPLE_900, PURPLE_500, PURPLE_300])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    section("Model output comparison")
    cols = [c for c in ["transaction_id", "rule_risk_score", "anomaly_score",
                        "fraud_probability", "rule_component", "anomaly_component",
                        "classifier_component", "risk_score", "risk_level"]
            if c in filtered.columns]
    st.dataframe(
        filtered.sort_values("risk_score", ascending=False).head(25)[cols],
        use_container_width=True, hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.1f"),
        },
    )


# ============================================================
# AI EXPLAINER (with follow-up chat)
# ============================================================

elif page == "AI Explainer":

    section("AI Investigation Assistant")
    st.write(
        "The AI Explainer translates FraudLens model signals into an investigation "
        "narrative for human reviewers. Ask follow-up questions below."
    )

    scope = st.radio(
        "What do you want to discuss?",
        ["One transaction", "Whole filtered portfolio"],
        horizontal=True,
    )

    record = None
    chat_key = "chat_portfolio"
    suggestions = [
        "Which scenarios look most risky?",
        "What patterns stand out in the top cases?",
        "Where should the team focus first?",
    ]

    if scope == "One transaction":
        ids = filtered["transaction_id"].dropna().astype(str).tolist()
        selected = st.selectbox("Select transaction", ids, key="ai_transaction")
        record = filtered[filtered["transaction_id"] == selected].iloc[0]
        chat_key = f"chat_{selected}"

        summary = record.get("investigation")
        if pd.isna(summary) or not str(summary).strip():
            summary = record.get("ai_interpretation")
        if pd.isna(summary) or not str(summary).strip():
            summary = "No AI investigation summary is available for this transaction."

        st.markdown("### Investigation Summary")
        st.info(str(summary))

        c1, c2, c3 = st.columns(3)
        c1.metric("Combined Risk Score", f"{record.get('risk_score', 0):.2f}")
        c2.markdown("**Risk Level**")
        c2.markdown(badge(record.get("risk_level")), unsafe_allow_html=True)
        c3.metric("Investigation Priority", safe_text(record.get("investigation_priority")))

        with st.expander("🧠 AI interpretation", expanded=True):
            interp = record.get("ai_interpretation")
            if pd.isna(interp) or not str(interp).strip():
                st.info("No AI interpretation is available.")
            else:
                st.markdown(str(interp))

        with st.expander("📂 Full investigation record"):
            full = record.get("full_investigation")
            if pd.isna(full) or not str(full).strip():
                st.info("No full investigation record is available.")
            else:
                st.text_area("Investigation record", str(full), height=350, disabled=True)

        st.caption(f"Interpretation source: {safe_text(record.get('interpretation_source'))}")

        suggestions = [
            "Why was this flagged?",
            "What should I check next?",
            "How reliable is this score?",
            "Is the device or location suspicious?",
        ]

    # ---------------- Chat ----------------
    st.markdown("### 💬 Ask a follow-up")

    history = st.session_state.setdefault(chat_key, [])

    top_l, top_r = st.columns([4, 1])
    top_l.caption(
        "Live AI answers are enabled."
        if get_api_key()
        else "Offline mode: add `GROQ_API_KEY` to enable full AI answers."
    )
    if top_r.button("Clear chat", key=f"clear_{chat_key}"):
        st.session_state[chat_key] = []
        st.rerun()

    for msg in history:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
            st.markdown(msg["content"])

    pending = None
    chip_cols = st.columns(len(suggestions))
    for i, (col, text) in enumerate(zip(chip_cols, suggestions)):
        if col.button(text, key=f"sug_{chat_key}_{i}", use_container_width=True):
            pending = text

    typed = st.chat_input("Ask anything about this case, e.g. 'What would make this look legitimate?'")
    question = typed or pending

    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Analysing signals..."):
                answer = ask_ai(history, record, filtered)
            st.markdown(answer)
        history.append({"role": "assistant", "content": answer})

    st.markdown(
        """
        <div class="confidence-box">
            <strong>Confidence Note</strong><br><br>
            FraudLens is an experimental project built on synthetic data. Its rule
            engine, anomaly detector, supervised classifier, and combined risk score
            generate investigative signals, not definitive proof of fraud.<br><br>
            Results should therefore be interpreted within a human-led review process.<br><br>
            Classifier probabilities at or near 1.0 on synthetic data may reflect easily
            separable patterns in the dataset and should not be interpreted as
            real-world certainty or production-level model confidence.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "FraudLens • Experimental payment-risk intelligence prototype "
    "• Synthetic data • Human-led investigation"
)