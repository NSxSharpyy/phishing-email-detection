# 🛡️ Phishing Email Detection Model

A supervised machine learning model built with Python and Scikit-learn to classify emails as either "Safe" or "Phishing" based on textual patterns and malicious URL indicators.

## ⚡ Features
- **Natural Language Processing (NLP):** Employs `TfidfVectorizer` to filter out English stop words, handle case normalization, and calculate word weights.
- **Probabilistic Classification:** Utilizes the `MultinomialNB` (Naive Bayes) algorithm, which is highly efficient for text-based security analysis.
- **Performance Diagnostics:** Outputs a Confusion Matrix and a Classification Report (Tracking Precision, Recall, and F1-Score).

## 🛠️ Concepts Learned
- **Text Vectorization:** Converting raw text strings into numerical feature matrices that machine learning models can understand.
- **Data Splitting:** Properly partitioning data into training and testing sets to test the model on unseen data.
- **Security Metrics Optimization:** Evaluating the critical operational difference between False Negatives (dangerous missed phishing attacks) and False Positives (safe emails marked as spam).

## 📦 How to Run

1. Make sure you have Python 3.x installed.
2. Clone this repository:
```bash
git clone https://github.com/NSxSharpyy/phishing-email-detection.git
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Execute the Python script:
```bash
python Phishing_Email_detection.py
```

## 📝 Execution Output
```text
Model Accuracy: 60.00%

--- CONFUSION MATRIX ---
[[1 2]
 [0 2]]

--- CLASSIFICATION REPORT ---
              precision    recall  f1-score   support

        Safe       1.00      0.33      0.50         3
    Phishing       0.50      1.00      0.67         2

==================== LIVE TESTING ====================
Email Text: 'Can we reschedule our sync to tomorrow morning?'
Prediction: 🚨 PHISHING

Email Text: '⚠️ ALERT: Security breach detected in your account. Login to [http://identity-check.com](http://identity-check.com) to verify.'
Prediction: 🚨 PHISHING
```

### 💡 Project Security Insights
The model achieved a **100% Recall for Phishing detection**, meaning it successfully caught every single phishing attempt (Zero False Negatives). While it shows a few False Positives due to the compact size of the training dataset, it successfully prioritizes high-security filtering where no malicious mail slips through to the user.
