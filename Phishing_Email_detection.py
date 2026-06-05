import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. BADA DATASET (Added more examples for stable training and testing)
# 0 = Safe Email, 1 = Phishing Email
emails = [
    # Safe Emails (0)
    "Hey, are we still meeting for lunch today at 1 PM?",
    "Please review the attached project schedule and let me know your thoughts.",
    "Can you send over the updated financial spreadsheet by EOD?",
    "Don't forget to submit your weekly timesheet before leaving on Friday.",
    "Hi team, the quarterly results meeting has been rescheduled to Thursday.",
    "Thanks for the help yesterday, I really appreciate it.",
    "Are you available for a quick call regarding the new design?",
    "The client loved the proposal, great job everyone!",
    
    # Phishing Emails (1)
    "URGENT: Your bank account has been compromised. Click http://secure-bank-login.com to reset password immediately!",
    "Congratulations! You have won a $1,000 Walmart gift card! Click here http://bit.ly/free-reward to claim now.",
    "Dear customer, your Netflix subscription has expired. Update payment details at http://netflix-verify-acc.top",
    "ACTION REQUIRED: Verify your corporate email credentials instantly at http://company-login-portal.net",
    "SECURITY ALERT: Suspicious login detected on your account. Verify identity at http://block-unauthorized-login.xyz",
    "Final Notice: Tax refund of $500 is waiting for you. Claim at http://irs-refund-portal.click",
    "Your password expires in 24 hours. Keep your current password by clicking http://update-credentials.info",
    "Win a free iPhone 15! Only 5 spots left, register now at http://win-free-prizes.net"
]

labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] 

# 2. Split Dataset (test_size=0.3 means 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.30, random_state=42)

# 3. Feature Extraction
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 4. Train Model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 60)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 60)

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

# 6. Live Testing (Naye unseen emails)
print("\n" + "="*20 + " LIVE TESTING " + "="*20)
sample_emails = [
    "Can we reschedule our sync to tomorrow morning?",
    "⚠️ ALERT: Security breach detected in your account. Login to http://identity-check.com to verify."
]

sample_vectorized = vectorizer.transform(sample_emails)
predictions = model.predict(sample_vectorized)

for email, pred in zip(sample_emails, predictions):
    status = "🚨 PHISHING" if pred == 1 else "🟢 SAFE"
    print(f"Email Text: '{email}'\nPrediction: {status}\n")
