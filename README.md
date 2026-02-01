# 🚬 Tobacco Mortality Risk Prediction System

An AI-powered web application that predicts **tobacco-related mortality risk** using machine learning.  
The system leverages historical mortality and household tobacco expenditure data to classify individuals into **Low Risk** or **High Risk** categories.

---

## 📌 Project Overview

Tobacco consumption is a major contributor to preventable mortality worldwide.  
This project uses a **Random Forest Classifier** trained on an aggregated mortality dataset to estimate mortality risk based on individual lifestyle and socio-economic factors.

The application provides:
- Machine learning–based risk prediction
- Probability-based mortality assessment
- A Flask-powered web interface for real-time predictions

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Classifier  
- **Training Strategy:**  
  - Aggregated mortality data is converted into **synthetic individual-level samples**
  - Multiple samples are generated per dataset row to learn underlying patterns
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

### 🎯 Input Features
| Feature | Description |
|------|-----------|
| Age | Age of the individual |
| Cigarettes per Day | Average cigarettes smoked daily |
| Years of Smoking | Total years of smoking |
| Income | Annual disposable income |
| Disease Indicator | Existing smoking-related disease (0/1) |

### 🧾 Output
- **Low Mortality Risk**
- **High Mortality Risk**
- Optional probability score for high risk

---

## 🌐 Web Application (Flask)

The Flask application exposes:
- A homepage UI (`/`)
- A prediction API (`/predict`) that accepts JSON input and returns risk classification

---
## 📂 Project Structure

├── app.py # Flask web application
├── model.py # Machine learning model logic
├── requirements.txt # Python dependencies
├── combined_mortality_dataset.csv # Dataset used for training
├── templates/
│ └── index.html # Frontend UI
└── README.md # Project documentation


