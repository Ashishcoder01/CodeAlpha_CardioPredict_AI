from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent
LAB_MODEL_PATH = ROOT / "artifacts" / "lab_cvd_model.joblib"
LAB_META_PATH = ROOT / "artifacts" / "lab_cvd_metadata.json"

FEATURES = [
    "age", "sex", "bmi", "systolic_bp", "diastolic_bp",
    "total_cholesterol", "hdl", "diabetes"
]


@st.cache_resource
def load_lab_model():
    if not LAB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Lab model not found at {LAB_MODEL_PATH}. "
            "Copy lab_cvd_model.joblib into the artifacts folder."
        )
    model = joblib.load(LAB_MODEL_PATH)
    metadata = {}
    if LAB_META_PATH.exists():
        import json
        with open(LAB_META_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    return model, metadata


def extract_report_text(uploaded_file):
    """Extract selectable PDF text; use OCR for images/scanned pages if available."""
    filename = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    text = ""

    if filename.endswith(".pdf"):
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            text = "\n".join(page.get_text("text") for page in pdf)
        if text.strip():
            return text, "PDF text extraction"
        # Scanned PDF: try OCR if Tesseract is installed.
        try:
            import pytesseract
            from io import BytesIO
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(raw, dpi=180)
            text = "\n".join(pytesseract.image_to_string(page) for page in pages)
            return text, "OCR on scanned PDF"
        except Exception:
            return "", (
                "This PDF appears to be scanned. OCR is unavailable. "
                "Install Tesseract OCR and pdf2image/Poppler, or enter values manually."
            )

    try:
        import pytesseract
        from io import BytesIO
        image = Image.open(BytesIO(raw)).convert("RGB")
        return pytesseract.image_to_string(image), "Image OCR"
    except Exception as exc:
        return "", (
            "Image OCR could not run. Install Tesseract OCR and pytesseract, "
            f"or enter values manually. Details: {exc}"
        )


def _first_number(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (ValueError, AttributeError):
                pass
    return None


def parse_report_values(text):
    """Best-effort extraction only. Values must be checked by the user."""
    patterns = {
        "age": [
            r"\b(?:age)\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?)?\b"
        ],
        "bmi": [
            r"\bBMI\b\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)",
            r"\bbody mass index\b\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)"
        ],
        "systolic_bp": [
            r"\b(?:systolic(?: blood pressure)?|SYS|SBP)\b\s*[:\-]?\s*(\d{2,3})"
        ],
        "diastolic_bp": [
            r"\b(?:diastolic(?: blood pressure)?|DIA|DBP)\b\s*[:\-]?\s*(\d{2,3})"
        ],
        "total_cholesterol": [
            r"\b(?:total cholesterol|cholesterol total|serum cholesterol|TC)\b"
            r"\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)"
        ],
        "hdl": [
            r"\b(?:HDL(?: cholesterol)?|HDL-C)\b\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)"
        ],
    }
    return {
        key: _first_number(value_patterns, text)
        for key, value_patterns in patterns.items()
    }



def interpret_measurements(age, bmi, systolic_bp, diastolic_bp, total_cholesterol, hdl, sex, diabetes):
    """Return cautious, rule-based explanations for common adult measurements."""
    findings = []

    # BMI is a screening measure, not a diagnosis. Adult thresholds only.
    if bmi < 18.5:
        findings.append(("BMI", f"{bmi:.1f} kg/m²", "Below the usual adult range",
                         "A low BMI can be associated with undernutrition or other causes; BMI alone cannot explain why.",
                         "Discuss unexpected weight loss or persistently low weight with a healthcare professional."))
    elif bmi < 25:
        findings.append(("BMI", f"{bmi:.1f} kg/m²", "Usual adult range",
                         "This falls in the commonly used adult BMI range of 18.5–24.9.",
                         "Consider this alongside overall health; BMI is only a screening measure."))
    elif bmi < 30:
        findings.append(("BMI", f"{bmi:.1f} kg/m²", "Above usual adult range",
                         "BMI is in the overweight screening category; it does not directly measure body fat or diagnose a condition.",
                         "If relevant, discuss sustainable lifestyle options and overall cardiovascular risk with a clinician."))
    else:
        findings.append(("BMI", f"{bmi:.1f} kg/m²", "High adult BMI range",
                         "BMI is in the obesity screening category. BMI alone does not diagnose health problems.",
                         "A clinician can interpret this with waist measurement, history, and other health factors."))

    # Common adult blood-pressure categories; one reading is not a diagnosis.
    if systolic_bp >= 180 or diastolic_bp >= 120:
        findings.append(("Blood pressure", f"{systolic_bp:.0f}/{diastolic_bp:.0f} mmHg", "Very high reading — prompt action",
                         "This reading is in a very high range. A single measurement does not establish a diagnosis.",
                         "Sit quietly and repeat after at least 1 minute. If it remains this high, seek urgent medical advice; if accompanied by chest pain, shortness of breath, weakness, numbness, vision change, or difficulty speaking, seek emergency care."))
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        findings.append(("Blood pressure", f"{systolic_bp:.0f}/{diastolic_bp:.0f} mmHg", "High reading",
                         "This is in a high adult blood-pressure range. Readings can vary and should be confirmed with proper repeated measurements.",
                         "Repeat measurements after resting and arrange a healthcare review, especially if readings remain elevated."))
    elif systolic_bp >= 130 or diastolic_bp >= 80:
        findings.append(("Blood pressure", f"{systolic_bp:.0f}/{diastolic_bp:.0f} mmHg", "Above optimal range",
                         "This falls in a range considered elevated under commonly used US adult categories; guidelines can differ.",
                         "Check readings over time using a correctly fitted cuff and discuss persistent elevation with a clinician."))
    elif systolic_bp >= 120 and diastolic_bp < 80:
        findings.append(("Blood pressure", f"{systolic_bp:.0f}/{diastolic_bp:.0f} mmHg", "Elevated systolic reading",
                         "The systolic value is above the optimal range while the diastolic value is below 80.",
                         "Monitor readings over time; one reading alone is not a diagnosis."))
    else:
        findings.append(("Blood pressure", f"{systolic_bp:.0f}/{diastolic_bp:.0f} mmHg", "Within usual adult range",
                         "This reading is below the commonly used high blood-pressure thresholds.",
                         "Continue routine monitoring as advised, especially if you have symptoms or a history of blood-pressure problems."))

    if total_cholesterol >= 240:
        tc_status, tc_meaning, tc_action = ("High", "Total cholesterol is in a high range; this value alone does not show the full lipid profile or diagnose disease.", "Review the complete lipid panel and overall risk with a healthcare professional.")
    elif total_cholesterol >= 200:
        tc_status, tc_meaning, tc_action = ("Borderline high", "Total cholesterol is above the commonly used desirable threshold.", "Interpret alongside LDL, HDL, triglycerides, age, history, and other risk factors.")
    else:
        tc_status, tc_meaning, tc_action = ("Desirable range", "Total cholesterol is below 200 mg/dL, a commonly used desirable threshold for adults.", "This does not by itself rule out cardiovascular disease; review the full lipid profile if available.")
    findings.append(("Total cholesterol", f"{total_cholesterol:.0f} mg/dL", tc_status, tc_meaning, tc_action))

    low_hdl_threshold = 40 if sex == 1 else 50
    if hdl < low_hdl_threshold:
        hdl_status = "Low"
        hdl_meaning = f"HDL is below the commonly used adult threshold of {low_hdl_threshold} mg/dL for this sex category. HDL is one part of the overall lipid picture."
        hdl_action = "Review the full lipid panel and cardiovascular risk factors with a clinician; do not interpret HDL in isolation."
    elif hdl >= 60:
        hdl_status = "Higher HDL level"
        hdl_meaning = "HDL is at or above 60 mg/dL, often considered a favorable marker, but it does not cancel out other risk factors."
        hdl_action = "Interpret with the complete lipid panel and overall health history."
    else:
        hdl_status = "Not flagged by this simple threshold"
        hdl_meaning = "This HDL value is not below the simple threshold used here; interpretation depends on the complete lipid profile and clinical context."
        hdl_action = "Review alongside LDL, triglycerides, and other risk factors if available."
    findings.append(("HDL cholesterol", f"{hdl:.0f} mg/dL", hdl_status, hdl_meaning, hdl_action))

    if diabetes:
        findings.append(("Diabetes status", "Diabetes or borderline selected", "Risk factor selected",
                         "The model input indicates diabetes or borderline diabetes; this is a user-provided category, not a lab-based diagnosis.",
                         "Use your clinician's confirmed diagnosis and latest glucose/A1C results for medical decisions."))
    else:
        findings.append(("Diabetes status", "No known diabetes selected", "No condition flagged from this input",
                         "This input does not establish that diabetes is absent; the app has not tested blood glucose or A1C.",
                         "Follow routine screening guidance from a healthcare professional."))

    return findings

def render_lab_report():
    st.markdown(
        '<div class="section-title">Lab Report Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Upload a lab report, review extracted '
        'measurements, and run the separate research model.</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This educational model estimates association with historical, "
        "self-reported cardiovascular disease in NHANES data. It is not a "
        "diagnostic tool or a validated estimate of future disease."
    )

    try:
        model, metadata = load_lab_model()
    except Exception as exc:
        st.error("The lab-report model could not be loaded.")
        st.code(str(exc))
        st.caption(
            "Your original heart-disease model is separate and is not modified."
        )
        return

    uploaded = st.file_uploader(
        "Upload a lab report",
        type=["pdf", "png", "jpg", "jpeg"],
        help="PDF text extraction and image OCR are best-effort. Always verify values."
    )

    extracted_text = ""
    parsed = {}
    if uploaded is not None:
        if uploaded.size > 12 * 1024 * 1024:
            st.error("Please upload a file smaller than 12 MB.")
            return

        with st.spinner("Reading report..."):
            extracted_text, method = extract_report_text(uploaded)

        st.caption(f"Extraction method: {method}")
        if extracted_text.strip():
            with st.expander("Review extracted report text"):
                st.text_area(
                    "Extracted text (may contain errors)",
                    value=extracted_text,
                    height=220,
                    key="lab_extracted_text"
                )
            parsed = parse_report_values(extracted_text)
        else:
            st.warning(
                "No readable text was extracted. You can still enter the "
                "measurements manually below."
            )

    st.markdown("#### Confirm patient and laboratory values")
    st.caption(
        "Use age in years, blood pressure in mmHg, and cholesterol in mg/dL. "
        "The model does not automatically convert units. Correct every extracted value."
    )

    with st.form("lab_report_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", min_value=18, max_value=120,
                value=int(parsed["age"]) if parsed.get("age") is not None and 18 <= parsed["age"] <= 120 else 50
            )
            sex = st.selectbox(
                "Sex (NHANES code)", options=[1, 2],
                format_func=lambda x: "Male (1)" if x == 1 else "Female (2)"
            )
            bmi = st.number_input(
                "BMI (kg/m²)", min_value=10.0, max_value=80.0, value=(
                    float(parsed["bmi"]) if parsed.get("bmi") is not None and 10 <= parsed["bmi"] <= 80 else 25.0
                ), step=0.1
            )

        with c2:
            systolic_bp = st.number_input(
                "Systolic BP (mmHg)", min_value=60.0, max_value=280.0,
                value=float(parsed["systolic_bp"]) if parsed.get("systolic_bp") is not None and 60 <= parsed["systolic_bp"] <= 280 else 120.0,
                step=1.0
            )
            diastolic_bp = st.number_input(
                "Diastolic BP (mmHg)", min_value=30.0, max_value=180.0,
                value=float(parsed["diastolic_bp"]) if parsed.get("diastolic_bp") is not None and 30 <= parsed["diastolic_bp"] <= 180 else 80.0,
                step=1.0
            )
            total_cholesterol = st.number_input(
                "Total cholesterol (mg/dL)", min_value=50.0, max_value=700.0,
                value=float(parsed["total_cholesterol"]) if parsed.get("total_cholesterol") is not None and 50 <= parsed["total_cholesterol"] <= 700 else 190.0,
                step=1.0
            )

        with c3:
            hdl = st.number_input(
                "HDL cholesterol (mg/dL)", min_value=5.0, max_value=200.0,
                value=float(parsed["hdl"]) if parsed.get("hdl") is not None and 5 <= parsed["hdl"] <= 200 else 50.0,
                step=1.0
            )
            diabetes = st.selectbox(
                "Diabetes status",
                options=[0, 1],
                format_func=lambda x: "No known diabetes (0)" if x == 0 else "Diabetes or borderline (1)"
            )
            confirmed = st.checkbox(
                "I reviewed and confirmed these values and units.",
                value=False
            )

        submitted = st.form_submit_button(
            "Analyze Lab Values",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not confirmed:
            st.warning("Please confirm that you reviewed the values before continuing.")
            return
        if systolic_bp <= diastolic_bp:
            st.error("Systolic blood pressure must be greater than diastolic blood pressure.")
            return

        row = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "total_cholesterol": total_cholesterol,
            "hdl": hdl,
            "diabetes": diabetes
        }], columns=FEATURES)

        try:
            predicted_class = int(model.predict(row)[0])
            classes = list(model.classes_)
            probabilities = model.predict_proba(row)[0]
            score = float(probabilities[classes.index(1)])

            st.divider()
            st.markdown("### Model Output")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="result-card"><div class="result-label">'
                            'PREDICTED HISTORICAL-ASSOCIATION CLASS</div><div class="result-value">'
                            + ("Positive class" if predicted_class == 1 else "Negative class")
                            + '</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(
                    '<div class="result-card"><div class="result-label">'
                    'MODEL SCORE (NOT A CLINICAL PROBABILITY)</div><div class="result-value">'
                    f'{score:.1%}</div></div>',
                    unsafe_allow_html=True
                )

            st.caption(
                "The class represents a model association with historical self-reported "
                "cardiovascular disease, not a diagnosis or future-risk prediction. "
                "This score is not clinically calibrated."
            )
            with st.expander("View submitted model inputs"):
                st.dataframe(row, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Measurement explanations")
            st.caption(
                "These are general adult screening thresholds, not diagnoses. Reference ranges can vary by lab, "
                "country, pregnancy status, age, medicines, and medical history. Use the reference interval printed "
                "on your report where applicable."
            )
            findings = interpret_measurements(
                age, bmi, systolic_bp, diastolic_bp, total_cholesterol, hdl, sex, diabetes
            )
            for name, value, status, meaning, next_step in findings:
                with st.container(border=True):
                    st.markdown(f"**{name} — {value}**")
                    st.markdown(f"**Screening flag:** {status}")
                    st.write(meaning)
                    st.markdown(f"**Suggested next step:** {next_step}")

            st.markdown("### What this report can and cannot tell you")
            st.write(
                "The flags above compare the values you confirmed with simple, commonly used adult thresholds. "
                "They do not identify the cause of an abnormal result and do not confirm that you have a disease. "
                "The separate machine-learning output estimates association with historical self-reported cardiovascular "
                "disease in its training dataset; it is not a personal future-risk estimate and should not guide treatment."
            )
        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

    st.markdown(
        '<div class="disclaimer"><strong>Medical and privacy notice:</strong> '
        'This tool is for educational research only and must not be used to diagnose, '
        'rule out disease, or make treatment decisions. Do not upload reports unless '
        'you are comfortable processing their contents in this local application. '
        'Consult a qualified healthcare professional for medical concerns.</div>',
        unsafe_allow_html=True
    )
