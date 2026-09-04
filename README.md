# 💳 Credit Card Fraud Detection

> A machine learning based application that detects potentially fraudulent credit card transactions using feature engineering, classification models, and an interactive Streamlit dashboard.

## 🚀 Live Demo

**Streamlit App:** Add your deployed Streamlit URL here

---

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where the objective is to identify suspicious transactions while minimizing incorrect fraud alerts.

This project uses transaction data from the Kaggle **Fraud Detection** dataset and compares three machine learning algorithms:

* Logistic Regression
* Decision Tree
* Random Forest

The project also includes feature engineering, class-imbalance handling, model evaluation, and an interactive Streamlit interface for transaction analysis.

---

## ✨ Features

* 📊 Exploratory data analysis
* 🧹 Data cleaning and preprocessing
* ⚙️ Feature engineering
* ⚖️ Class imbalance handling
* 🤖 Logistic Regression
* 🌳 Decision Tree
* 🌲 Random Forest
* 📈 ROC-AUC curve
* 🎯 Precision-Recall curve
* 🔲 Confusion matrix
* 📋 Model performance comparison
* 💳 Interactive transaction prediction
* 🌐 Streamlit web application

---

## 🧠 Feature Engineering

Several additional features are created from the raw transaction data.

### Transaction Time Features

The transaction timestamp is used to extract:

* Transaction hour
* Day
* Month
* Weekday
* Weekend indicator

### Customer Age

Customer age is estimated from the transaction date and date of birth.

### Transaction Amount

A logarithmic transformation of the transaction amount is created to reduce the effect of highly skewed transaction values.

### Geographic Distance

The approximate distance between the customer's location and merchant location is calculated using latitude and longitude coordinates.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Used as a simple and interpretable baseline classification model.

### 2. Decision Tree

Captures non-linear relationships between transaction characteristics and fraud behavior.

### 3. Random Forest

Combines multiple decision trees and is used as the primary ensemble model for comparison.

---

## ⚖️ Handling Class Imbalance

Fraudulent transactions represent a small portion of the complete dataset.

Instead of relying only on accuracy, this project evaluates:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

The training data is also balanced by retaining fraudulent transactions and sampling legitimate transactions.

---

## 📊 Evaluation

The models are evaluated using:

| Metric    | Purpose                                            |
| --------- | -------------------------------------------------- |
| Accuracy  | Overall classification correctness                 |
| Precision | How many predicted fraud cases were actually fraud |
| Recall    | How many actual fraud cases were detected          |
| F1 Score  | Balance between precision and recall               |
| ROC-AUC   | Overall class discrimination                       |
| PR-AUC    | Performance on the minority fraud class            |

Precision-Recall analysis is particularly useful for this problem because fraud detection involves a highly imbalanced target variable.

---

## 🖥️ Streamlit Application

The application provides several sections:

### Dashboard

Displays:

* Number of models
* Best-performing model
* ROC-AUC
* Model comparison

### Transaction Prediction

Users can enter transaction information and receive:

* Fraud/legitimate prediction
* Fraud probability

### Model Performance

Displays model metrics and confusion matrices.

### ROC Curve

Compares the ROC curves of the trained models.

### Precision-Recall Curve

Compares precision and recall behavior across different classification thresholds.

---

## 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── fraud_model.pkl
│   ├── model_metadata.pkl
│   ├── model_results.csv
│   └── validation_predictions.csv
│
└── data/
    └── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
```

Move into the project directory:

```bash
cd credit-card-fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Models

Run:

```bash
python train_model.py
```

The script downloads the dataset through KaggleHub, performs feature engineering, trains the models, evaluates their performance, and saves the trained model.

---

## 🌐 Run the Streamlit App

After training:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Dataset

The project uses the Kaggle **Fraud Detection** dataset by `kartik2112`.

The dataset is downloaded programmatically using KaggleHub.

The raw CSV files are intentionally excluded from this repository because of their large size.

---

## 🔐 Data & Model Considerations

This project is intended for educational and portfolio purposes.

A real-world fraud detection system would require additional considerations such as:

* Real-time transaction monitoring
* Model drift detection
* Cost-sensitive learning
* Threshold optimization
* Explainable AI
* Privacy protection
* Secure transaction processing
* Continuous model retraining

---

## 🔮 Future Improvements

Possible extensions include:

* XGBoost / LightGBM comparison
* SMOTE and other imbalance techniques
* Threshold optimization
* SHAP-based explainability
* Real-time API deployment
* Docker deployment
* Database integration
* Model monitoring
* Automated retraining pipeline

---

## 🛠️ Tech Stack

**Language:** Python

**Machine Learning:** Scikit-learn

**Data Processing:** Pandas, NumPy

**Visualization:** Matplotlib

**Web Application:** Streamlit

**Model Persistence:** Joblib

**Dataset:** Kaggle / KaggleHub

---

## 👨‍💻 Author

**Utkarsh Sharma**

