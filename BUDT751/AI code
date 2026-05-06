"""
Loan Application Predictor
--------------------------
A Streamlit app for loan officers to input applicant data
and receive an ML-based approve/deny classification.

Tech stack: Streamlit, scikit-learn, pandas, numpy
Usage: streamlit run loan_predictor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

APP_TITLE = "Loan Application Decision Support"
APP_SUBTITLE = "AI-assisted risk classification for loan officers"

# Feature definitions — edit these to match your real data schema
FEATURE_CONFIG = {
    "annual_income":       {"label": "Annual Income ($)",         "min": 10_000,  "max": 500_000, "default": 60_000,  "step": 1_000},
    "loan_amount":         {"label": "Loan Amount ($)",           "min": 1_000,   "max": 100_000, "default": 15_000,  "step": 500},
    "loan_term_months":    {"label": "Loan Term (months)",        "min": 12,      "max": 84,      "default": 36,      "step": 12},
    "credit_score":        {"label": "Credit Score",              "min": 300,     "max": 850,     "default": 680,     "step": 1},
    "debt_to_income":      {"label": "Debt-to-Income Ratio (%)",  "min": 0.0,     "max": 80.0,    "default": 28.0,    "step": 0.5},
    "employment_years":    {"label": "Years Employed",            "min": 0,       "max": 40,      "default": 5,       "step": 1},
    "num_open_accounts":   {"label": "Open Credit Accounts",      "min": 0,       "max": 30,      "default": 4,       "step": 1},
    "num_delinquencies":   {"label": "Past Delinquencies (2yr)",  "min": 0,       "max": 10,      "default": 0,       "step": 1},
}

# Risk thresholds — tune these based on your institution's risk appetite
APPROVAL_THRESHOLD   = 0.55   # Probability above which we classify as approved
HIGH_RISK_THRESHOLD  = 0.35   # Below this = high risk flag


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# Replace this section with your real data loader
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model on Lending Club data...")
def train_model():
    df = pd.read_csv("accepted_2007_to_2018Q4.csv", low_memory=False)

    # Keep only final loan outcomes
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

    # Target: 1 = good/approved-like outcome, 0 = bad/default-like outcome
    df["approved"] = (df["loan_status"] == "Fully Paid").astype(int)

    # Convert Lending Club columns to your app's feature names
    df["annual_income"] = df["annual_inc"]
    df["loan_amount"] = df["loan_amnt"]
    df["loan_term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    df["credit_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["debt_to_income"] = df["dti"]

    df["employment_years"] = (
        df["emp_length"]
        .astype(str)
        .str.replace("10+ years", "10", regex=False)
        .str.replace("< 1 year", "0", regex=False)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df["num_open_accounts"] = df["open_acc"]
    df["num_delinquencies"] = df["delinq_2yrs"]

    feature_cols = list(FEATURE_CONFIG.keys())

    df = df[feature_cols + ["approved"]]

    # Fill missing values instead of dropping everything
    df["employment_years"] = df["employment_years"].fillna(0)
    df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median())
    df["debt_to_income"] = df["debt_to_income"].fillna(df["debt_to_income"].median())
    df = df.dropna()

    # Optional: sample for faster Streamlit startup
    df = df.sample(n=min(100000, len(df)), random_state=42)

    X = df[feature_cols]
    y = df["approved"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, accuracy, feature_cols
      
# ─────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────

def predict_loan(model, scaler, feature_cols: list, applicant_data: dict) -> dict:
    """
    Runs inference on a single applicant record.
    
    Args:
        model:           Trained sklearn classifier
        scaler:          Fitted StandardScaler
        feature_cols:    Ordered list of feature names
        applicant_data:  Dict of {feature_name: value}
    
    Returns:
        dict with keys: approved (bool), probability (float),
                        risk_level (str), feature_importances (dict)
    """
    input_df = pd.DataFrame([applicant_data])[feature_cols]
    input_scaled = scaler.transform(input_df)

    approval_prob = model.predict_proba(input_scaled)[0][1]  # P(approved)
    approved = approval_prob >= APPROVAL_THRESHOLD

    # Risk tier
    if approval_prob >= APPROVAL_THRESHOLD:
        risk_level = "Low Risk" if approval_prob > 0.75 else "Moderate Risk"
    else:
        risk_level = "High Risk" if approval_prob < HIGH_RISK_THRESHOLD else "Elevated Risk"

    # Feature importances for this model (global, not local — swap for SHAP in v2)
    importances = dict(zip(feature_cols, model.feature_importances_))

    return {
        "approved":            approved,
        "probability":         round(float(approval_prob), 4),
        "risk_level":          risk_level,
        "feature_importances": importances,
    }


# ─────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────

def render_header():
    """Renders the app header and model metadata."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏦",
        layout="wide"
    )
    st.title(f"🏦 {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.divider()


def render_input_form() -> dict:
    """
    Renders the applicant data input form in the sidebar.
    
    Returns:
        dict of feature_name -> value
    """
    st.sidebar.header("📋 Applicant Information")
    st.sidebar.caption("Enter the applicant's financial profile below.")

    inputs = {}
    for feature_key, config in FEATURE_CONFIG.items():
        if isinstance(config["default"], float) or isinstance(config["step"], float):
            inputs[feature_key] = st.sidebar.number_input(
                label=config["label"],
                min_value=float(config["min"]),
                max_value=float(config["max"]),
                value=float(config["default"]),
                step=float(config["step"]),
                key=feature_key
            )
        else:
            inputs[feature_key] = st.sidebar.number_input(
                label=config["label"],
                min_value=int(config["min"]),
                max_value=int(config["max"]),
                value=int(config["default"]),
                step=int(config["step"]),
                key=feature_key
            )

    return inputs


def render_decision_card(result: dict):
    """
    Renders the main decision output card.
    
    Args:
        result: Output dict from predict_loan()
    """
    approved     = result["approved"]
    probability  = result["probability"]
    risk_level   = result["risk_level"]

    # Decision banner
    if approved:
        st.success(f"### ✅ RECOMMENDATION: APPROVE  —  {risk_level}")
    else:
        st.error(f"### ❌ RECOMMENDATION: DENY  —  {risk_level}")

    st.caption(
        "⚠️ This is a decision-support tool. The final credit decision "
        "remains with the loan officer and must comply with all applicable regulations."
    )

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Approval Probability",
            value=f"{probability:.1%}",
        )
    with col2:
        st.metric(
            label="Risk Classification",
            value=risk_level
        )
    with col3:
        st.metric(
            label="Decision Threshold",
            value=f"{APPROVAL_THRESHOLD:.0%}"
        )

    # Probability gauge bar
    st.write("")
    st.write("**Approval Confidence Score**")
    gauge_color = "normal" if approved else "inverse"
    st.progress(
        value=probability,
        text=f"{probability:.1%} probability of approval"
    )


def render_feature_importance(result: dict, applicant_data: dict):
    """
    Renders feature importance chart and applicant data summary.
    
    Args:
        result:         Output dict from predict_loan()
        applicant_data: Raw input dict from the form
    """
    st.divider()
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 Key Risk Drivers")
        st.caption("Global feature importances from the trained model.")

        importance_df = pd.DataFrame(
            list(result["feature_importances"].items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        # Map raw feature names to readable labels
        label_map = {k: v["label"] for k, v in FEATURE_CONFIG.items()}
        importance_df["Feature"] = importance_df["Feature"].map(label_map)

        st.bar_chart(importance_df.set_index("Feature")["Importance"])

    with col_right:
        st.subheader("📁 Application Summary")
        st.caption("Data submitted for this application.")

        summary_rows = []
        for key, value in applicant_data.items():
            label = FEATURE_CONFIG[key]["label"]
            if "Income" in label or "Amount" in label:
                formatted = f"${value:,.0f}"
            elif "Ratio" in label:
                formatted = f"{value:.1f}%"
            else:
                formatted = str(value)
            summary_rows.append({"Field": label, "Value": formatted})

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True
        )


def render_model_info(accuracy: float):
    """
    Renders model performance metadata in an expandable section.
    
    Args:
        accuracy: Test-set accuracy from training
    """
    with st.expander("🔬 Model Information & Compliance Notes", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Model Type:** Gradient Boosting Classifier  
            **Training Samples:** Lending Club historical loan records  
            **Test Set Accuracy:** `{accuracy:.1%}`  
            **Approval Threshold:** `{APPROVAL_THRESHOLD:.0%}`  
            """)
        with col2:
            st.markdown("""
            **Compliance Reminders:**
            - This tool does not make final credit decisions
            - All decisions subject to Equal Credit Opportunity Act (ECOA)
            - Adverse action notices required for denials
            - Model must be periodically audited for disparate impact
            """)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    """Entry point — orchestrates the full Streamlit app."""

    render_header()

    # Load / train model (cached after first run)
    try:
        model, scaler, accuracy, feature_cols = train_model()
    except Exception as e:
        st.error(f"❌ Model failed to load: {e}")
        st.stop()

    # Sidebar: model health indicator
    st.sidebar.divider()
    st.sidebar.success(f"✅ Model ready  |  Accuracy: {accuracy:.1%}")
    st.sidebar.caption("Model retrained on each app restart.")

    # Main area: input form
    applicant_data = render_input_form()

    # Predict button
    st.sidebar.divider()
    run_prediction = st.sidebar.button(
        "🔍 Evaluate Application",
        type="primary",
        use_container_width=True
    )

    if run_prediction:
        try:
            with st.spinner("Running risk assessment..."):
                result = predict_loan(model, scaler, feature_cols, applicant_data)

            render_decision_card(result)
            render_feature_importance(result, applicant_data)
            render_model_info(accuracy)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info("Check that all input fields are filled correctly.")

    else:
        # Empty state
        st.info(
            "👈 Fill in the applicant's information in the sidebar, "
            "then click **Evaluate Application** to generate a risk assessment."
        )
        render_model_info(accuracy)


if __name__ == "__main__":
    main()

