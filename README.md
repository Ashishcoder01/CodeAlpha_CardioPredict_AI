# CardioPredict AI

### Heart Disease Prediction & Lab Report Insights

A machine-learning educational application for exploring heart-health data, reviewing selected laboratory measurements, and understanding model outputs through a Streamlit interface.

**CodeAlpha Machine Learning Internship — Task 4: Disease Prediction from Medical Data**

---

## Overview

**CardioPredict AI** is a Streamlit application that demonstrates how machine-learning classification can be applied to structured medical data. It combines a heart-disease prediction workflow with a separate lab-report analysis page.

The project is designed for learning and demonstration. Its outputs are not medical diagnoses, and they must not replace advice from a qualified healthcare professional.

## Features

- **Home dashboard** — overview of the application.
- **Patient assessment** — enter structured patient features and view the heart-disease model output.
- **Model insights** — review the selected model, training configuration, and recorded evaluation metrics.
- **Lab report analysis** — upload a supported PDF or image, review extracted values, and correct them before analysis.
- **Measurement explanations** — provide cautious, general explanations for selected measurements such as blood pressure, BMI, total cholesterol, and HDL.
- **Separate lab-model output** — run the lab-data model independently from the original heart-disease model.
- **Input review and safety notes** — emphasize checking extracted values, units, and limitations before interpreting results.

> Availability of PDF/image OCR depends on local OCR tools and their system dependencies. Text-based PDFs are generally easier to extract than scanned documents.

## Technology Stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- joblib
- PyMuPDF (`pymupdf`) for PDF text extraction
- Pillow for image handling
- pytesseract and Tesseract OCR for OCR-enabled workflows
- pdf2image and Poppler for scanned-PDF OCR workflows, where used

## Machine-Learning Models

### Heart-disease model

The primary model uses **Logistic Regression** trained on the UCI Heart Disease dataset.

| Item | Recorded configuration |
|---|---|
| Dataset | UCI Heart Disease |
| Records | 303 |
| Input features | 13 |
| Train/test split | 80% / 20% |
| Cross-validation | 5-fold stratified |
| Selected algorithm | Logistic Regression |
| Test accuracy | 88.5% |
| Disease-class recall | 92.9% |
| Test ROC-AUC | 0.966 |

These are the project's recorded evaluation results, not a guarantee of performance on new patients. Results depend on the dataset, split, preprocessing, and evaluation setup.

### Lab-report model

The separate lab model uses a Logistic Regression pipeline with imputation and feature scaling. Its target was based on **historical, self-reported cardiovascular disease** in NHANES data; it should not be interpreted as a future-risk predictor or a diagnostic model.

| Metric | Recorded result |
|---|---:|
| Accuracy | 71.09% |
| Precision | 27.25% |
| Recall | 78.99% |
| F1-score | 40.52% |
| ROC-AUC | 0.8129 |

The lab model has low precision in the recorded evaluation. A positive output may therefore be a false positive, and its score is **not a calibrated clinical probability**. Review the model limitations before using or presenting its output.

## Project Structure

The following is the expected core layout. Your local project may contain additional files.

```text
CodeAlpha_CardioPredict_AI/
├── app.py
├── lab_report.py
├── requirements.txt
└── artifacts/
    ├── heart_disease_model.joblib
    ├── model_metadata.json
    ├── lab_cvd_model.joblib
    └── lab_cvd_metadata.json
```

The trained model files and their metadata are required at the paths expected by the application. If they are not included in the repository, arrange a reliable way to obtain them and document that process before deploying.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ashishcoder01/CodeAlpha_CardioPredict_AI.git
cd CodeAlpha_CardioPredict_AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use your usual approved environment-activation method or run the environment's Python executable directly.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If `requirements.txt` is present and up to date:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Otherwise, install the core packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib pymupdf Pillow
```

For OCR workflows, also install the Python packages and system tools used by your version of `lab_report.py`:

```bash
pip install pytesseract pdf2image
```

Tesseract OCR and Poppler are separate system applications and may need to be installed and configured for your operating system. Check that the model artifacts exist before launching the application.

### 4. Run the app

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, which you can open in your browser.

## Using the Application

1. Open the Home dashboard.
2. Select **Patient Assessment** and enter the requested values.
3. Review the prediction and its disclaimer.
4. Open **Model Insights** to inspect the recorded model configuration and metrics.
5. Open **Lab Report Analysis** to upload a supported report.
6. Carefully review extracted values and units; correct any mistakes before continuing.
7. Read measurement explanations and the lab model output as separate pieces of information.

## Data and Model Notes

- The primary model is documented as using the UCI Heart Disease dataset.
- The lab-model training workflow used NHANES 2017–2018 tables, including demographic, blood pressure, body-measurement, cholesterol, diabetes, and medical-condition data.
- The lab-model target was derived from historical self-reported cardiovascular-condition responses.
- Data sources and feature definitions should be cited in any formal report, presentation, or redistribution of this project.
- Do not upload private patient reports, personal health information, API keys, virtual environments, or other sensitive files to a public repository.
- Confirm dataset terms and redistribution requirements before including raw datasets in the repository.

## Limitations and Medical Disclaimer

- This project is an educational machine-learning demonstration, **not a medical device**.
- It cannot diagnose, confirm, or rule out heart disease or another medical condition.
- General measurement flags are not personalized diagnoses and may not match the reference interval on a particular laboratory report.
- Laboratory values can use different units and reference ranges. Verify units and extracted values before interpreting them.
- OCR can misread digits, decimal points, labels, and units. Always check extracted values against the original report.
- The lab model reflects historical self-reported associations and has limited precision in the recorded evaluation. It does not estimate an individual's calibrated probability of future cardiovascular disease.
- Model metrics from a small or specific dataset do not establish clinical safety, fairness, calibration, or generalizability.
- Seek a qualified healthcare professional for interpretation of medical results. For urgent or severe symptoms, seek urgent medical care.

## Future Improvements

- Add automated tests for input validation, model loading, and report extraction.
- Improve OCR handling for varied laboratory-report layouts.
- Support configurable laboratory reference intervals and units.
- Add a model card documenting dataset provenance, intended use, limitations, and evaluation.
- Add deployment checks and a reproducible environment configuration.
- Include application screenshots and a verified live demo link when available.

## Acknowledgements

Developed as part of the **CodeAlpha Machine Learning Internship**, Task 4: Disease Prediction from Medical Data.

## Author

**Ashish**

GitHub: [Ashishcoder01](https://github.com/Ashishcoder01)

---

**For educational purposes only. Not for clinical decision-making.**
