# 📊 Customer Churn Prediction & Analysis

A professional, end-to-end machine learning solution for predicting customer churn. This project features a **real-time Flask API** and an **interactive web dashboard** to visualize risk factors and probability scores.

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-black)
![ML Library](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-orange)

## 🚀 Overview

This repository provides a production-ready system to identify at-risk customers before they churn. It utilizes a gradient-boosted decision tree (XGBoost) to achieve high accuracy and leverages SHAP for model interpretability.

### Key Features
*   **Real-time Scoring**: Instant churn probability for any customer via REST API.
*   **Batch Prediction**: Score thousands of customers at once by uploading a JSON array.
*   **Interpretability**: Identifies the top risk factors driving the model's decision (SHAP).
*   **Interactive Dashboard**: A sleek, modern UI for business users to interact with the model.

---

## 🛠️ Project Structure

```text
├── api/                # Flask API implementation
├── dashboard/          # Frontend assets (HTML, CSS, JS)
├── models/             # Serialized model (.pkl) and artifacts
├── src/                # Modular source code
│   ├── train.py        # Model training pipeline
│   ├── preprocess.py   # Data cleaning & transformation
│   └── predict.py      # Inference logic & feature engineering
├── requirements.txt    # Production & Development dependencies
└── README.md
```

---

## 💻 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Application
The API and Dashboard are served together using Flask.
```bash
python api/app.py
```
Visit **[http://127.0.0.1:5000/dashboard/](http://127.0.0.1:5000/dashboard/)** in your browser to access the interactive UI.

---

## 📡 API Documentation

### Single Prediction
`POST /predict`
```json
{
  "tenure": 5,
  "MonthlyCharges": 72.0,
  "TotalCharges": 360.0,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  ... (other features)
}
```

### Batch Prediction
`POST /predict/batch`
```json
[
  { "tenure": 5, "MonthlyCharges": 72.0, ... },
  { "tenure": 24, "MonthlyCharges": 50.0, ... }
]
```

---

## 🧪 Model Performance
The current model uses **XGBoost** and was evaluated on a standard churn dataset.
*   **Accuracy**: ~84%
*   **F1-Score (Churn Class)**: ~0.62
*   **AUC-ROC**: ~0.87

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
