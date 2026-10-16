
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CardioPredict AI | Heart Health Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "heart_disease_model.joblib"
META_PATH = ARTIFACTS / "model_metadata.json"


# =========================================================
# 2. CUSTOM DARK HEALTHCARE UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #03090D;
    --panel: #09151B;
    --panel2: #0D2025;
    --border: #20373D;
    --green: #00D995;
    --teal: #20F1D0;
    --white: #F4F8FA;
    --muted: #91A3AC;
}

.stApp {
    background: var(--bg);
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1750px;
    padding: 1.2rem 2rem 3rem;
}

h1, h2, h3, h4, p, label, .stMarkdown {
    color: var(--white);
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}

p {
    line-height: 1.7;
}

[data-testid="stSidebar"] {
    background: #061117;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--white);
}

[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

/* BRAND */
.brand {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1.2px;
    color: #FFFFFF;
}

.brand span {
    color: var(--green);
}

.brand-subtitle {
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--muted);
    margin-top: 4px;
}

.top-status {
    display: inline-block;
    color: var(--green);
    border: 1px solid #1D594A;
    background: #08241D;
    border-radius: 25px;
    padding: 8px 12px;
    font-size: 11px;
}

/* HERO */
.hero {
    background:
        radial-gradient(ellipse at 88% 5%, #103A34 0%, transparent 45%),
        linear-gradient(125deg, #07151B, #0A2024 60%, #071117);
    border: 1px solid #1B3938;
    border-radius: 18px;
    padding: 30px;
    margin: 12px 0 22px;
    box-shadow: 0 12px 40px rgba(0,0,0,.22);
}

.hero-kicker {
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

.hero-title {
    font-size: clamp(35px, 3.3vw, 57px);
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -2px;
    color: #FFFFFF;
    margin-bottom: 16px;
}

.hero-title span {
    color: var(--green);
}

.hero-description {
    font-size: 14px;
    color: #B0C0C6;
    line-height: 1.8;
    max-width: 570px;
}

.hero-heart {
    text-align: center;
    color: var(--teal);
    font-size: 110px;
    line-height: 1.4;
    text-shadow: 0 0 35px #00D99555;
}

.hero-heart-caption {
    color: #C1D8D8;
    text-align: center;
    font-size: 13px;
    font-weight: 700;
}

/* DASHBOARD CARDS */
.metric-card {
    background: linear-gradient(145deg, #0D2025, #09151B);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
    min-height: 115px;
}

.metric-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .8px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 800;
    margin: 9px 0 4px;
}

.metric-footer {
    color: var(--green);
    font-size: 10px;
}

.panel {
    background: linear-gradient(145deg, #0C1C22, #071117);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
    margin-bottom: 12px;
}

.panel-title {
    color: #F4F8FA;
    font-size: 14px;
    font-weight: 700;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 11px;
    margin-top: 5px;
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    margin: 12px 0 6px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 22px;
}

.feature-card {
    border-top: 1px solid var(--border);
    padding: 14px 7px;
    min-height: 98px;
}

.feature-icon {
    color: var(--green);
    font-size: 23px;
    margin-bottom: 7px;
}

.feature-title {
    color: #F4F8FA;
    font-size: 12px;
    font-weight: 700;
}

.feature-description {
    color: var(--muted);
    font-size: 11px;
    margin-top: 5px;
}

/* INPUTS */
[data-testid="stForm"] {
    background: #09151B;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 22px;
}

[data-testid="stWidgetLabel"] p {
    color: #DBE6EA !important;
    font-size: 12px;
    font-weight: 600;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] {
    background: #0D2025;
    border-color: #294148;
    color: #FFFFFF;
    border-radius: 8px;
}

input {
    color: #FFFFFF !important;
}

.section-label {
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 7px 0 15px;
}

.stButton > button,
.stFormSubmitButton > button {
    min-height: 43px;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: var(--green);
    color: #03110D;
    border: 1px solid var(--green);
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: #37EDB4;
    border-color: #37EDB4;
    color: #03110D;
}

/* RESULTS */
.result-card {
    background: linear-gradient(145deg, #0D2025, #09151B);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 21px;
    min-height: 120px;
}

.result-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.result-value {
    color: var(--green);
    font-size: 25px;
    font-weight: 800;
    margin-top: 12px;
}

.disclaimer {
    background: #201A0E;
    border: 1px solid #5D4926;
    border-left: 4px solid #E4AC42;
    border-radius: 10px;
    padding: 15px;
    margin-top: 23px;
    color: #F3DDAA;
    font-size: 12px;
    line-height: 1.8;
}

.disclaimer strong {
    color: #F3DDAA;
}

.footer {
    text-align: center;
    color: #70838C;
    font-size: 11px;
    padding-top: 20px;
}

[data-testid="stMetric"] {
    background: #0A171D;
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 14px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--green) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

@media (max-width: 900px) {
    .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 22px;
    }

    .hero-title {
        font-size: 37px;
    }

    .hero-heart {
        font-size: 75px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD SAVED MODEL
# =========================================================
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    trained_model = joblib.load(MODEL_PATH)
    metadata = {}

    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

    return trained_model, metadata


try:
    model, metadata = load_artifacts()
except Exception as error:
    st.error("Could not load your trained model.")
    st.write("Check that both artifact files are in the artifacts folder.")
    st.code(str(error))
    st.stop()


# =========================================================
# 4. NAVIGATION
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

brand_col, status_col = st.columns([4, 1])

with brand_col:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:5px 0 14px">
        <div style="font-size:36px;color:#00D995">♡</div>
        <div>
            <div class="brand">
                CardioPredict <span>AI</span>
            </div>
            <div class="brand-subtitle">
                INTELLIGENT HEART HEALTH ANALYTICS
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with status_col:
    st.markdown("""
    <div style="text-align:right;padding-top:14px">
        <span class="top-status">● MODEL READY</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div style="height:1px;background:#20373D;margin-bottom:15px"></div>',
    unsafe_allow_html=True
)

n1, n2, n3, n4, spacer = st.columns(
    [1, 1.4, 1.1, 1.6, 4.4]
)

with n1:
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()

with n2:
    if st.button("Patient Assessment", use_container_width=True):
        st.session_state.page = "Assessment"
        st.rerun()

with n3:
    if st.button("Model Insights", use_container_width=True):
        st.session_state.page = "Model Info"
        st.rerun()

with n4:
    if st.button("Lab Report Analysis", use_container_width=True):
        st.session_state.page = "Lab Report"
        st.rerun()

page = st.session_state.page


# =========================================================
# 5. HOME DASHBOARD
# =========================================================
if page == "Home":

    left, right = st.columns([0.91, 1.09], gap="large")

    with left:
        st.markdown("""
        <div style="padding-top:20px">
            <div class="hero-kicker">
                NEXT-GENERATION HEALTH ANALYTICS
            </div>
            <div class="hero-title">
                Smarter Care.<br>
                Faster <span>Decisions.</span>
            </div>
            <div class="hero-description">
                AI-powered exploration of clinical measurements
                through machine learning. Review model evaluation,
                explore patient features, and understand classification results.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button(
            "Start Patient Assessment  →",
            type="primary",
            use_container_width=True
        ):
            st.session_state.page = "Assessment"
            st.rerun()

        st.write("")

        f1, f2 = st.columns(2)

        with f1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⌁</div>
                <div class="feature-title">AI Diagnostics</div>
                <div class="feature-description">
                    Explore model predictions
                </div>
            </div>
            """, unsafe_allow_html=True)

        with f2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⌘</div>
                <div class="feature-title">Health Analytics</div>
                <div class="feature-description">
                    Visualize evaluation data
                </div>
            </div>
            """, unsafe_allow_html=True)

        f1, f2 = st.columns(2)

        with f1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">◇</div>
                <div class="feature-title">Risk Classification</div>
                <div class="feature-description">
                    Review predicted classes
                </div>
            </div>
            """, unsafe_allow_html=True)

        with f2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">▥</div>
                <div class="feature-title">Model Insights</div>
                <div class="feature-description">
                    Understand model performance
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:18px;font-size:10px;
                    letter-spacing:1.3px;color:#70838C">
            CODEALPHA · MACHINE LEARNING TASK 4
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Dashboard header
        st.markdown("""
        <div class="panel" style="margin-top:10px">
            <div style="display:flex;justify-content:space-between;gap:10px">
                <div class="panel-title">
                    CardioPredict AI Dashboard
                </div>
                <div style="font-size:11px;color:#00D995">
                    ● Active
                </div>
            </div>
            <div class="panel-subtitle">
                Model performance overview
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dashboard metrics
        c1, c2, c3 = st.columns(3)

        cards = [
            ("DATA RECORDS", "303", "13 input features"),
            ("TEST ACCURACY", "88.5%", "61 test records"),
            ("TEST ROC-AUC", "0.966", "Logistic Regression"),
        ]

        for col, (label, value, footer) in zip([c1, c2, c3], cards):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-footer">{footer}</div>
                </div>
                """, unsafe_allow_html=True)

        # Heart visual and model information
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("""
            <div class="panel" style="text-align:center;min-height:220px">
                <div class="panel-title">Heart Health Analytics</div>
                <div class="hero-heart">♡</div>
                <div class="hero-heart-caption">
                    Machine Learning Pipeline
                </div>
                <div class="panel-subtitle">
                    Educational classification demonstration
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="panel" style="min-height:220px">
                <div class="panel-title">Model Summary</div>
                <div class="panel-subtitle">
                    Selected using training cross-validation
                </div>
                <div style="margin-top:20px">
                    <div style="font-size:11px;color:#91A3AC">
                        SELECTED ALGORITHM
                    </div>
                    <div style="font-size:19px;font-weight:800;
                                color:#00D995;margin-top:7px">
                        Logistic Regression
                    </div>
                    <div style="font-size:11px;color:#91A3AC;margin-top:18px">
                        MEAN CROSS-VALIDATION ROC-AUC
                    </div>
                    <div style="font-size:25px;font-weight:800;
                                color:#20F1D0;margin-top:6px">
                        0.902
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Compact chart
        st.markdown("""
        <div class="panel-title" style="margin:15px 0 5px">
            Model Performance
        </div>
        <div class="panel-subtitle">
            Accuracy on the held-out test set
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 2.15))
        fig.patch.set_facecolor("#09151B")
        ax.set_facecolor("#09151B")

        bars = ax.barh(
            ["Logistic Regression", "Random Forest", "SVM"],
            [88.5, 88.5, 88.5],
            color=["#00D995", "#168B7D", "#28718B"],
            height=0.5,
        )

        ax.set_xlim(0, 100)
        ax.tick_params(colors="#D7E3E7", labelsize=9)
        ax.set_xlabel("Accuracy (%)", color="#91A3AC", fontsize=9)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", color="#20373D", alpha=0.7)
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.1f%%", padding=4,
                     color="#F4F8FA", fontsize=8)
        fig.tight_layout(pad=0.7)
        st.pyplot(fig)
        plt.close(fig)

        # Dataset distribution
        st.markdown("""
        <div class="panel-title" style="margin:15px 0 5px">
            Dataset Distribution
        </div>
        <div class="panel-subtitle">
            Target classes in the complete dataset
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 1.7))
        fig.patch.set_facecolor("#09151B")
        ax.set_facecolor("#09151B")

        ax.barh(
            ["No disease", "Disease present"],
            [164, 139],
            color=["#28718B", "#00D995"],
            height=0.48,
        )
        ax.tick_params(colors="#D7E3E7", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", color="#20373D", alpha=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.7)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("""
    <div class="disclaimer">
        <strong>Medical disclaimer:</strong>
        CardioPredict AI is an educational machine learning project.
        It is not a medical diagnostic system. Model scores are not
        clinically validated individual disease probabilities.
        Consult a qualified healthcare professional for medical concerns.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 6. PATIENT ASSESSMENT
# =========================================================
elif page == "Assessment":

    st.markdown(
        '<div class="section-title">Patient Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="section-subtitle">
        Enter the 13 features expected by the trained model.
        Use the original dataset definitions and category encodings.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Default values are for interface demonstration only. "
        "They are not medical recommendations."
    )

    with st.form("patient_assessment_form"):

        st.markdown(
            '<div class="section-label">01 / PERSONAL INFORMATION</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age (years)", min_value=1, max_value=120, value=50
            )

        with c2:
            sex = st.selectbox(
                "Sex (dataset code)",
                [0, 1],
                help="Original UCI coding: 0 = female, 1 = male."
            )

        st.divider()

        st.markdown(
            '<div class="section-label">02 / CLINICAL MEASUREMENTS</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            cp = st.selectbox("Chest pain type (cp)", [1, 2, 3, 4])

        with c2:
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)",
                min_value=50, max_value=250, value=120
            )

        with c3:
            chol = st.number_input(
                "Serum cholesterol (mg/dl)",
                min_value=50, max_value=700, value=220
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            fbs = st.selectbox(
                "Fasting blood sugar > 120 (fbs)", [0, 1]
            )

        with c2:
            restecg = st.selectbox(
                "Resting ECG (restecg)", [0, 1, 2]
            )

        with c3:
            thalach = st.number_input(
                "Maximum heart rate achieved",
                min_value=50, max_value=250, value=150
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            exang = st.selectbox(
                "Exercise-induced angina (exang)", [0, 1]
            )

        with c2:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0, max_value=10.0,
                value=1.0, step=0.1
            )

        with c3:
            slope = st.selectbox(
                "ST segment slope (slope)", [1, 2, 3]
            )

        c1, c2 = st.columns(2)

        with c1:
            ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])

        with c2:
            thal = st.selectbox("Thal category (thal)", [3, 6, 7])

        submitted = st.form_submit_button(
            "Run Model Assessment",
            type="primary",
            use_container_width=True
        )

    if submitted:

        patient = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }])

        try:
            prediction = int(model.predict(patient)[0])
            probabilities = model.predict_proba(patient)[0]
            classes = list(model.classes_)

            positive_score = float(
                probabilities[classes.index(1)]
            )
            negative_score = float(
                probabilities[classes.index(0)]
            )

            st.divider()
            st.markdown("### Assessment Results")

            if prediction == 1:
                st.error(
                    "Model classification: Disease present. "
                    "This is not a medical diagnosis."
                )
            else:
                st.success(
                    "Model classification: No disease. "
                    "This does not rule out heart disease."
                )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED CLASS</div>
                    <div class="result-value">
                        {"Disease present" if prediction == 1 else "No disease"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">POSITIVE CLASS MODEL SCORE</div>
                    <div class="result-value">{positive_score:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Class score breakdown")

            scores = pd.DataFrame({
                "Class": ["No disease", "Disease present"],
                "Score": [negative_score, positive_score],
            }).set_index("Class")

            st.bar_chart(scores, horizontal=True)

            with st.expander("View submitted clinical features"):
                st.dataframe(
                    patient,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Model scores are not clinically validated individual "
                "disease probabilities."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

    st.markdown("""
    <div class="disclaimer">
        <strong>Medical disclaimer:</strong>
        This educational application cannot diagnose or rule out
        heart disease. Consult a qualified healthcare professional
        for medical concerns.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 7. MODEL INSIGHTS
# =========================================================
elif page == "Model Info":

    st.markdown(
        '<div class="section-title">Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="section-subtitle">
        Model configuration and recorded evaluation results.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Test Accuracy", "88.5%")

    with c2:
        st.metric("Disease-class Recall", "92.9%")

    with c3:
        st.metric("Test ROC-AUC", "0.966")

    left, right = st.columns(2)

    with left:
        st.markdown("### Training Configuration")
        st.write("Algorithm: Logistic Regression")
        st.write("Dataset: UCI Heart Disease")
        st.write("Records: 303")
        st.write("Input features: 13")
        st.write("Train/test split: 80% / 20%")
        st.write("Cross-validation: 5-fold stratified")

    with right:
        st.markdown("### Model Comparison")

        metrics = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "SVM",
            ],
            "CV ROC-AUC": [0.902, 0.896, 0.888],
            "Accuracy": [0.885, 0.885, 0.885],
            "Recall": [0.929, 0.964, 0.964],
            "Test ROC-AUC": [0.966, 0.950, 0.961],
        })

        st.dataframe(
            metrics.style.format({
                "CV ROC-AUC": "{:.3f}",
                "Accuracy": "{:.1%}",
                "Recall": "{:.1%}",
                "Test ROC-AUC": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Saved Metadata")

    if metadata:
        st.json(metadata)
    else:
        st.info("No metadata file found.")

    st.markdown("""
    <div class="disclaimer">
        <strong>Limitations:</strong>
        The model was trained on a small historical dataset.
        These metrics do not establish clinical safety, calibration,
        generalizability, or suitability for medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

elif page == "Lab Report":
    from lab_report import render_lab_report
    render_lab_report()


# =========================================================
# 8. FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    CardioPredict AI · CodeAlpha Task 4 · Educational demonstration
</div>
""", unsafe_allow_html=True)
