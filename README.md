# 💳 Fraud Detection in Online Banking using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent transactions in online banking systems using various machine learning and deep learning algorithms. The system analyzes patterns in transaction data and flags suspicious activities, helping banks and financial institutions to mitigate financial losses and protect user assets.

---

## 📊 Problem Statement

With the increase in digital banking, fraudsters have more opportunities to exploit vulnerabilities. The aim is to develop a predictive model that classifies whether a transaction is **fraudulent** or **genuine** based on the transaction features.

---

## 🧠 Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Neural Network (MLPClassifier / Keras-based model)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow/Keras
- **Jupyter Notebook** for development
- **Google Colab** or **VS Code** as IDE

---

## 🧾 Dataset

- Dataset Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains **284,807** transactions
- Features: `Time`, `Amount`, `V1` to `V28` (anonymized PCA components), `Class` (0 = Legit, 1 = Fraud)

---

## 📈 Exploratory Data Analysis (EDA)

- Highly **imbalanced dataset**: Fraudulent cases are ~0.17%
- Distribution plots for `Amount`, `Time`
- Correlation matrix and heatmap
- Class imbalance visualization

---

## ⚙️ Model Building

1. **Preprocessing**:
   - Normalization using `StandardScaler`
   - Handling class imbalance with **SMOTE (Synthetic Minority Oversampling Technique)**

2. **Training**:
   - Split data into training and testing (80/20)
   - Train using various models and tune hyperparameters

3. **Evaluation Metrics**:
   - Accuracy
   - Precision, Recall
   - F1-Score
   - ROC-AUC Curve
   - Confusion Matrix

---

## 🔍 Results

| Model              | Accuracy | Precision | Recall | F1-Score | AUC  |
|-------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 97.8%   | 91.2%     | 87.6%  | 89.4%    | 0.94 |
| Random Forest       | 99.2%   | 96.5%     | 93.7%  | 95.1%    | 0.98 |
| XGBoost             | 99.3%   | 97.1%     | 94.2%  | 95.6%    | 0.99 |
| Neural Network      | 98.9%   | 95.3%     | 91.8%  | 93.5%    | 0.97 |

---

## 📌 Conclusion

- XGBoost outperformed other models in detecting fraud.
- SMOTE helped address the class imbalance issue.
- With proper tuning and feature analysis, fraud detection accuracy can be improved significantly.

---

## 🚀 Future Work

- Deploy the model with a REST API (Flask/FastAPI)
- Real-time fraud detection with streaming data (Kafka + Spark)
- Integration with banking dashboards
- Explainable AI (LIME/SHAP) for model transparency

---

## 🔗 Demo & Resources


- Dataset: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🧑‍💻 Author

**K Thirumalesh**  
Email: thirumaleshk991@gmail.com  

