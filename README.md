# CardioPredict AI

### Heart Disease Prediction & Lab Report Insights

CardioPredict AI is a Streamlit-based machine-learning project that demonstrates structured heart-disease prediction and lab-report analysis through a dark, healthcare-focused interface.

<p align="center">
  <a href="https://cardiopredict-ai1.streamlit.app/"><strong>Open the Live App</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/Ashishcoder01/CodeAlpha_CardioPredict_AI"><strong>GitHub Repository</strong></a>
</p>

**CodeAlpha Machine Learning Internship — Task 4: Disease Prediction from Medical Data**

---

## Overview

CardioPredict AI brings two separate educational ML workflows together in one Streamlit application:

1. **Heart Disease Prediction** — enter structured patient features and review the output from a Logistic Regression model trained on the UCI Heart Disease dataset.
2. **Lab Report Analysis** — upload a supported laboratory report, review extracted values, edit them when needed, and view cautious explanations of selected measurements and the output of a separate lab-data model.

The app also includes a home dashboard and a Model Insights page. It is intended for learning and demonstration—not clinical decision-making.

## Live Demo

**Application:** https://cardiopredict-ai1.streamlit.app/

The application is deployed with Streamlit Community Cloud. Availability can depend on the hosting service.

## Key Features

- **Home dashboard** with a custom dark healthcare-inspired interface.
- **Patient Assessment** workflow for the main heart-disease model.
- **Model Insights** page with model configuration and recorded evaluation metrics.
- **Lab Report Analysis** page for PDF and image uploads.
- **PDF text extraction** for PDFs containing selectable text.
- **Optional OCR** for images and scanned PDFs, using Tesseract OCR and Poppler system dependencies.
- **Editable extracted values** so users can review and correct extraction results before analysis.
- **Measurement explanations** for selected values such as BMI, blood pressure, total cholesterol, and HDL.
- **Separate model artifacts** for the primary heart-disease model and the lab-data model.
- **Input validation and limitations messaging** to reduce the chance of misinterpreting extracted values or model scores.

> OCR can misread characters, digits, decimal points, labels, and units. Always verify extracted values against the original report.

## Technology Stack

- **Language:** Python
- **App framework:** Streamlit
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, Logistic Regression
- **Model persistence:** joblib
- **Visualizations:** Matplotlib
- **PDF processing:** PyMuPDF
- **Image handling:** Pillow
- **Optional OCR:** pytesseract, Tesseract OCR, pdf2image, Poppler

## Machine-Learning Models

### 1. Main heart-disease model

The primary model uses Logistic Regression and is documented as being trained with the UCI Heart Disease dataset.

| Item | Recorded configuration/result |
|---|---|
| Dataset | UCI Heart Disease |
| Records | 303 |
| Input features | 13 |
| Train/test split | 80% / 20% |
| Cross-validation | 5-fold stratified |
| Algorithm | Logistic Regression |
| Test accuracy | 88.5% |
| Disease-class recall | 92.9% |
| Test ROC-AUC | 0.966 |

These are recorded project evaluation results. They do not guarantee similar performance on new patients or in other populations.

### 2. Lab-report model

The separate lab model uses a Logistic Regression pipeline and eight features: age, sex, BMI, systolic blood pressure, diastolic blood pressure, total cholesterol, HDL, and diabetes.

Its target is based on **historical self-reported cardiovascular disease** in the CDC NHANES 2017–2018 data. It is not trained or validated as a future cardiovascular-risk predictor.

| Metric | Recorded result |
|---|---:|
| Accuracy | 71.09% |
| Precision | 27.25% |
| Recall | 78.99% |
| F1-score | 40.52% |
| ROC-AUC | 0.8129 |

The lab model's recorded precision is low, so positive outputs may be false positives. Its score is **not a calibrated clinical probability**. Use it only as an educational example and do not use it to diagnose, rule out, or manage a medical condition.

## Repository Structure

```text
CodeAlpha_CardioPredict_AI/
├── app.py
├── lab_report.py
├── requirements.txt
├── packages.txt
├── verify_lab_model.py
├── .gitignore
└── artifacts/
    ├── heart_disease_model.joblib
    ├── model_metadata.json
    ├── lab_cvd_model.joblib
    └── lab_cvd_metadata.json
```

The model artifacts and metadata are included in the repository. The `packages.txt` file lists Linux system packages used for OCR support on Streamlit Community Cloud.

## Run Locally

### Prerequisites

- Python 3.12 recommended
- Git
- Optional: Tesseract OCR and Poppler for image/scanned-PDF OCR

### 1. Clone the repository

```bash
git clone https://github.com/Ashishcoder01/CodeAlpha_CardioPredict_AI.git
cd CodeAlpha_CardioPredict_AI
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For scanned-PDF OCR on Windows or macOS, install the system applications Tesseract OCR and Poppler separately and ensure they are available to your environment. The Python packages alone do not install these system tools.

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

### 5. Verify the lab model (optional)

```bash
python verify_lab_model.py
```

This script loads the lab model and runs a software smoke test using synthetic example values. The sample values are not a real patient's data.

## Using the Application

1. Open the Home dashboard.
2. Go to **Patient Assessment**, provide the requested feature values, and review the model output and disclaimer.
3. Go to **Model Insights** to examine the documented model configuration and metrics.
4. Go to **Lab Report Analysis** and upload a supported PDF or image.
5. Carefully compare extracted values with the original report; correct any errors and check all units.
6. Read the measurement explanations and model output as educational information, not as medical advice.

## Data and Reproducibility Notes

- The primary model is documented as using the UCI Heart Disease dataset.
- The lab-model metadata identifies CDC NHANES 2017–2018 as its data source.
- The lab target represents historical self-reported cardiovascular disease, not future risk.
- Feature values and units must match the expectations of the relevant model.
- The tracked model artifacts must be kept consistent with their metadata and preprocessing.
- Review each dataset's terms and citation requirements before redistribution or reuse.

## Limitations and Medical Disclaimer

**CardioPredict AI is an educational software project, not a medical device.**

- It does not diagnose, confirm, or rule out heart disease or another medical condition.
- Model metrics from a particular dataset do not establish clinical validity, calibration, fairness, or generalizability.
- The lab model reflects historical self-reported associations and has low recorded precision; its output should not be used as an individual future-risk estimate.
- OCR results may be incorrect. Verify all values, labels, decimal points, and units before proceeding.
- Measurement categories are general explanations and may not match the reference ranges supplied by a particular laboratory or the guidance used in an individual's care.
- Do not upload private patient reports or personally identifying health information to a public repository.
- For the interpretation of medical results, contact a qualified healthcare professional. If you experience severe or urgent symptoms, seek urgent medical care.

## Future Improvements

- Add automated tests for page flows, input validation, model loading, and report extraction.
- Improve OCR handling for different report layouts and table structures.
- Add more explicit unit checks and configurable laboratory reference ranges.
- Publish a model card with dataset provenance, intended use, evaluation, and limitations.
- Add screenshots of the deployed interface and a reproducible automated deployment check.

## Acknowledgements

Developed as part of the **CodeAlpha Machine Learning Internship**, Task 4: Disease Prediction from Medical Data.

## Author

**Ashish**  
GitHub: [Ashishcoder01](https://github.com/Ashishcoder01)

---

**For educational purposes only. Not for clinical decision-making.**
