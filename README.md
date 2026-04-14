# 📊 Customer Churn Prediction API & Dashboard

[![Live Demo](https://img.shields.io/badge/Live_Demo-View_Project-2ea44f?style=for-the-badge&logo=vercel)](https://web-production-c6a31.up.railway.app/)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![ML Library](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)

A professional, end-to-end Machine Learning solution for predicting customer churn. This project provides a **real-time Flask REST API** backed by an **XGBoost classification model**, packaged with a highly interactive, responsive web dashboard for business users.

*🚀 **Live Demo**: [https://web-production-c6a31.up.railway.app/](https://web-production-c6a31.up.railway.app/)*

---

## 🌟 Key Features

*   ⚡ **Real-time Scoring**: Get instant churn predictions along with percentage probabilities.
*   📦 **Batch Prediction processing**: Score thousands of customers at once by uploading a JSON or CSV array.
*   🔍 **Model Interpretability**: Provides the top driving risk factors for every prediction using data from SHAP.
*   🎨 **Interactive Dashboard**: A sleek, modern user interface built using vanilla HTML/CSS/JS with smooth animations and dynamic SVGs.
*   ☁️ **Cloud Native**: Fully containerized and optimized for deployment on Serverless or ephemeral platforms like Railway and Render.

## 🏗️ Project Structure

The repository is modular and structured following software engineering best practices:

```text
customer-churn-prediction/
├── api/                         # Flask Backend
│   ├── app.py                   # Main Flask application and API routes
│   ├── static/                  # Dashboard assets (CSS, JS)
│   └── templates/               # Dashboard UI (HTML)
├── models/                      # Serialized ML Artifacts (Pickle files)
│   ├── xgb_model.pkl            # Final trained XGBoost model
│   └── scaler.pkl               # Data standard scaler
├── notebooks/                   # Jupyter Notebooks for experimentation
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── ...
├── src/                         # Machine Learning Pipeline source code
│   ├── preprocess.py            # Data cleaning logic
│   ├── features.py              # Feature engineering
│   ├── train.py                 # Model training workflow
│   └── predict.py               # Inference and loading logic
├── Procfile                     # Deployment profile for Railway
├── requirements.txt             # Python dependencies
└── runtime.txt                  # Python runtime definition (3.11)
```

## 🛠️ Technology Stack

*   **Languages & Core**: Python, SQL, JavaScript (Vanilla), HTML5, CSS3
*   **Data Science Libs**: Pandas, NumPy, Matplotlib, Seaborn
*   **Machine Learning**: XGBoost, Scikit-Learn (Logistic Regression, Random Forest)
*   **Backend & Framework**: Flask, Gunicorn
*   **Version Control & Deployment**: GitHub, Railway
*   **Analytics & BI**: Power BI

---

## 🚦 Quick Start (Local Development)

Want to run the API and Dashboard on your own machine? Follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/aniket252005/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Create a virtual environment**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the Application**
```bash
python api/app.py
```

*The dashboard will be available at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)*

---

## 📡 API Reference

If you prefer to bypass the UI, you can call the API directly using REST.

### 1. Health Check
`GET /health`
Returns the status of the model and the server.

### 2. Predict Single Customer
`POST /predict`
Score a single customer's data in real-time.

**Request Body (JSON):**
```json
{
  "tenure": 24,
  "MonthlyCharges": 75.50,
  "TotalCharges": 1812.00,
  "Contract_One year": 1,
  "Contract_Two year": 0,
  "InternetService_Fiber optic": 1
}
```

**Response:**
```json
{
  "PredictedChurn": 0,
  "ChurnProbability": 0.28,
  "RiskLevel": "Low",
  "latency_ms": 12.5
}
```

## ☁️ Deployment

This application is actively deployed on **Railway**.
To deploy your own version:
1. Connect your GitHub repository to Railway.
2. The platform will automatically detect the `Procfile` and `requirements.txt`.
3. Ensure the environment uses `PORT` (assigned automatically by Railway).
