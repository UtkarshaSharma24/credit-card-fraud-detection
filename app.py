
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudShield AI | Transaction Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f5f7fb;
        }

        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #263244;
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb !important;
        }

        .brand {
            padding: 8px 0 22px 0;
        }

        .brand-title {
            font-size: 25px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .brand-subtitle {
            color: #9ca3af;
            font-size: 13px;
            margin-top: 4px;
        }

        .hero {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            padding: 28px 30px;
            border-radius: 18px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 8px 30px rgba(17, 24, 39, 0.12);
        }

        .hero h1 {
            margin: 0;
            font-size: 34px;
        }

        .hero p {
            margin: 8px 0 0 0;
            color: #d1d5db;
            font-size: 15px;
        }

        .kpi {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 15px;
            padding: 18px;
            min-height: 105px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.04);
        }

        .kpi-label {
            color: #6b7280;
            font-size: 13px;
            font-weight: 600;
        }

        .kpi-value {
            color: #111827;
            font-size: 27px;
            font-weight: 800;
            margin-top: 5px;
        }

        .kpi-note {
            color: #6b7280;
            font-size: 12px;
            margin-top: 3px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 750;
            color: #111827;
            margin: 10px 0 12px 0;
        }

        .risk-card {
            padding: 24px;
            border-radius: 16px;
            background: white;
            border: 1px solid #e5e7eb;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        }

        .risk-number {
            font-size: 42px;
            font-weight: 850;
            margin: 5px 0;
        }

        .risk-label {
            color: #6b7280;
            font-size: 13px;
        }

        .status-box {
            padding: 18px 20px;
            border-radius: 14px;
            margin: 12px 0;
            font-weight: 700;
            font-size: 17px;
        }

        .safe {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .danger {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }

        .warning {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5e7eb;
            padding: 12px;
            border-radius: 14px;
        }

        .small-muted {
            color: #6b7280;
            font-size: 12px;
        }

        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 12px;
            padding: 30px 0 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")


@st.cache_data
def load_results():
    return pd.read_csv("models/model_results.csv")


@st.cache_data
def load_validation():
    return pd.read_csv("models/validation_predictions.csv")


try:
    model = load_model()
    results_df = load_results()
    validation_df = load_validation()
except Exception as e:
    st.error("Model files could not be loaded.")
    st.code(str(e))
    st.stop()

# ============================================================
# HELPERS
# ============================================================

def safe_model_name(name):
    return str(name).lower().replace(" ", "_")


def metric_value(model_name, metric):
    row = results_df[results_df["Model"] == model_name]
    if row.empty:
        return None
    return float(row.iloc[0][metric])


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def build_transaction(
    amount,
    latitude,
    longitude,
    city_population,
    merchant_latitude,
    merchant_longitude,
    transaction_date,
    hour,
    customer_age,
    gender,
    category,
    merchant,
    city,
    state,
    job,
):
    distance_km = calculate_distance_km(
        latitude,
        longitude,
        merchant_latitude,
        merchant_longitude,
    )

    return pd.DataFrame(
        {
            "amt": [amount],
            "lat": [latitude],
            "long": [longitude],
            "city_pop": [city_population],
            "merch_lat": [merchant_latitude],
            "merch_long": [merchant_longitude],
            "transaction_hour": [hour],
            "transaction_day": [transaction_date.day],
            "transaction_month": [transaction_date.month],
            "transaction_weekday": [transaction_date.weekday()],
            "is_weekend": [1 if transaction_date.weekday() >= 5 else 0],
            "customer_age": [customer_age],
            "amt_log": [np.log1p(amount)],
            "distance_km": [distance_km],
            "gender": [gender],
            "category": [category],
            "merchant": [merchant],
            "city": [city],
            "state": [state],
            "job": [job],
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">🛡️ FraudShield AI</div>
            <div class="brand-subtitle">Transaction Risk Monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Select a module",
        [
            "🏠 Overview",
            "🔍 Transaction Scanner",
            "📊 Model Analytics",
            "📈 ROC Analysis",
            "🎯 Precision-Recall",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    best_model = results_df.loc[
        results_df["F1 Score"].idxmax(), "Model"
    ]

    st.markdown("**Production model**")
    st.success(f"✓ {best_model}")

    st.markdown(
        '<div class="small-muted">Local ML inference • No transaction data is uploaded</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>Credit Card Fraud Monitoring</h1>
            <p>AI-assisted transaction screening for identifying suspicious payment activity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_model = results_df.loc[
        results_df["F1 Score"].idxmax(), "Model"
    ]

    fraud_count = int(validation_df["actual"].sum())
    total_count = len(validation_df)
    fraud_rate = fraud_count / total_count if total_count else 0
    best_auc = results_df["ROC-AUC"].max()
    best_f1 = results_df["F1 Score"].max()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Models Evaluated</div>
                <div class="kpi-value">{len(results_df)}</div>
                <div class="kpi-note">Classification models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Best Model</div>
                <div class="kpi-value" style="font-size:21px;">{best_model}</div>
                <div class="kpi-note">Highest F1-score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Best ROC-AUC</div>
                <div class="kpi-value">{best_auc:.3f}</div>
                <div class="kpi-note">Class discrimination</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Validation Fraud Rate</div>
                <div class="kpi-value">{fraud_rate:.1%}</div>
                <div class="kpi-note">{fraud_count:,} fraud cases</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="section-title">Model snapshot</div>', unsafe_allow_html=True)

        display_df = results_df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].map(lambda x: f"{x:.2%}")
        display_df["Precision"] = display_df["Precision"].map(lambda x: f"{x:.2%}")
        display_df["Recall"] = display_df["Recall"].map(lambda x: f"{x:.2%}")
        display_df["F1 Score"] = display_df["F1 Score"].map(lambda x: f"{x:.2%}")
        display_df["ROC-AUC"] = display_df["ROC-AUC"].map(lambda x: f"{x:.3f}")
        display_df["PR-AUC"] = display_df["PR-AUC"].map(lambda x: f"{x:.3f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown('<div class="section-title">Detection focus</div>', unsafe_allow_html=True)
        st.info(
            "For fraud detection, Recall and Precision are important because "
            "the system should catch suspicious transactions while limiting unnecessary alerts."
        )

        st.metric("Best F1 Score", f"{best_f1:.2%}")
        st.metric("Validation Samples", f"{total_count:,}")

    st.markdown(
        '<div class="footer">FraudShield AI • Machine Learning Fraud Detection Demo</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# TRANSACTION SCANNER
# ============================================================

elif page == "🔍 Transaction Scanner":

    st.markdown(
        """
        <div class="hero">
            <h1>Transaction Scanner</h1>
            <p>Enter transaction information to generate an AI-based fraud risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Demo environment: prediction is generated locally from the trained machine-learning pipeline."
    )

    with st.form("transaction_form"):

        st.markdown("### 💳 Payment details")

        c1, c2, c3 = st.columns(3)

        with c1:
            amount = st.number_input(
                "Transaction amount ($)",
                min_value=0.01,
                value=125.00,
                step=10.00,
                format="%.2f",
            )

        with c2:
            category = st.selectbox(
                "Transaction category",
                [
                    "grocery_pos",
                    "shopping_net",
                    "misc_net",
                    "shopping_pos",
                    "entertainment",
                    "gas_transport",
                    "food_dining",
                    "personal_care",
                    "health_fitness",
                    "home",
                    "kids_pets",
                    "travel",
                ],
            )

        with c3:
            transaction_date = st.date_input(
                "Transaction date"
            )

        st.markdown("### 👤 Customer details")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            customer_age = st.number_input(
                "Customer age",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )

        with c2:
            gender = st.selectbox("Gender", ["M", "F"])

        with c3:
            city = st.text_input("Customer city", value="New York")

        with c4:
            state = st.text_input("Customer state", value="NY")

        c1, c2 = st.columns(2)

        with c1:
            job = st.text_input(
                "Customer occupation",
                value="Engineer",
            )

        with c2:
            city_population = st.number_input(
                "City population",
                min_value=0,
                value=50000,
                step=1000,
            )

        st.markdown("### 📍 Location & merchant")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Customer location**")
            latitude = st.number_input(
                "Customer latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7128,
                format="%.5f",
            )
            longitude = st.number_input(
                "Customer longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-74.0060,
                format="%.5f",
            )

        with c2:
            st.markdown("**Merchant location**")
            merchant_latitude = st.number_input(
                "Merchant latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7306,
                format="%.5f",
            )
            merchant_longitude = st.number_input(
                "Merchant longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-73.9352,
                format="%.5f",
            )

        merchant = st.text_input(
            "Merchant",
            value="Online Merchant",
        )

        st.markdown("### 🕐 Transaction timing")

        hour = st.slider(
            "Transaction hour",
            min_value=0,
            max_value=23,
            value=14,
            help="24-hour format. Example: 14 = 2 PM.",
        )

        submitted = st.form_submit_button(
            "🔎 Analyze Transaction",
            use_container_width=True,
            type="primary",
        )

    if submitted:

        transaction = build_transaction(
            amount=amount,
            latitude=latitude,
            longitude=longitude,
            city_population=city_population,
            merchant_latitude=merchant_latitude,
            merchant_longitude=merchant_longitude,
            transaction_date=transaction_date,
            hour=hour,
            customer_age=customer_age,
            gender=gender,
            category=category,
            merchant=merchant,
            city=city,
            state=state,
            job=job,
        )

        try:
            probability = float(model.predict_proba(transaction)[0][1])
            prediction = int(model.predict(transaction)[0])

            distance = float(transaction["distance_km"].iloc[0])

            st.divider()
            st.markdown("### Risk assessment")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="risk-label">FRAUD RISK SCORE</div>
                        <div class="risk-number">{probability:.1%}</div>
                        <div class="risk-label">Model confidence signal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="risk-label">TRANSACTION AMOUNT</div>
                        <div class="risk-number">${amount:,.2f}</div>
                        <div class="risk-label">{category.replace("_", " ").title()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r3:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="risk-label">CUSTOMER–MERCHANT DISTANCE</div>
                        <div class="risk-number">{distance:.1f} km</div>
                        <div class="risk-label">Calculated from coordinates</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if prediction == 1:
                st.markdown(
                    """
                    <div class="status-box danger">
                        🚨 Suspicious transaction detected — manual review is recommended.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="status-box safe">
                        ✓ Transaction appears legitimate based on the trained model.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Risk interpretation
            if probability >= 0.80:
                st.markdown(
                    '<div class="status-box danger">High risk: strong fraud signal.</div>',
                    unsafe_allow_html=True,
                )
            elif probability >= 0.50:
                st.markdown(
                    '<div class="status-box warning">Medium risk: additional verification may be appropriate.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-box safe">Low risk: no strong fraud signal detected.</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("View engineered transaction features"):
                st.dataframe(
                    transaction.T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )

        except Exception as e:
            st.error("Prediction could not be completed.")
            st.code(str(e))
            st.info(
                "If this error mentions missing or unexpected features, the training "
                "pipeline and the Streamlit input schema need to be aligned."
            )

# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":

    st.markdown(
        """
        <div class="hero">
            <h1>Model Analytics</h1>
            <p>Compare classification models and inspect their validation performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
    ]

    selected_metric = st.selectbox(
        "Metric to compare",
        metrics,
        index=3,
    )

    chart_df = results_df[["Model", selected_metric]].copy()
    chart_df = chart_df.set_index("Model")

    st.bar_chart(chart_df)

    st.markdown("### Detailed comparison")

    formatted = results_df.copy()
    for col in metrics:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")

    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Confusion Matrix")

    selected_model = st.selectbox(
        "Select model",
        results_df["Model"].tolist(),
    )

    safe_name = safe_model_name(selected_model)
    pred_column = f"{safe_name}_pred"

    if pred_column in validation_df.columns:
        y_true = validation_df["actual"]
        y_pred = validation_df[pred_column]

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Legitimate", "Fraud"],
        ).plot(ax=ax)

        ax.set_title(f"{selected_model} — Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(f"Prediction column '{pred_column}' was not found.")

# ============================================================
# ROC CURVE
# ============================================================

elif page == "📈 ROC Analysis":

    st.markdown(
        """
        <div class="hero">
            <h1>ROC Analysis</h1>
            <p>Compare how well each model separates legitimate and fraudulent transactions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in results_df.iterrows():
        model_name = row["Model"]
        probability_column = f"{safe_model_name(model_name)}_prob"

        if probability_column in validation_df.columns:
            fpr, tpr, _ = roc_curve(
                validation_df["actual"],
                validation_df[probability_column],
            )

            ax.plot(
                fpr,
                tpr,
                label=f"{model_name} (AUC={row['ROC-AUC']:.3f})",
            )

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "A higher ROC-AUC generally indicates stronger overall class discrimination."
    )

# ============================================================
# PRECISION-RECALL
# ============================================================

elif page == "🎯 Precision-Recall":

    st.markdown(
        """
        <div class="hero">
            <h1>Precision–Recall Analysis</h1>
            <p>Evaluate the trade-off between catching fraud and limiting false alerts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in results_df.iterrows():
        model_name = row["Model"]
        probability_column = f"{safe_model_name(model_name)}_prob"

        if probability_column in validation_df.columns:
            precision, recall, _ = precision_recall_curve(
                validation_df["actual"],
                validation_df[probability_column],
            )

            ax.plot(
                recall,
                precision,
                label=f"{model_name} (AP={row['PR-AUC']:.3f})",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve Comparison")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Precision–Recall analysis is particularly useful for fraud datasets "
        "because fraudulent transactions are typically much less common than legitimate ones."
    )

st.markdown(
    '<div class="footer">FraudShield AI • Educational / Portfolio Project</div>',
    unsafe_allow_html=True,
)
