import pandas as pd
import re
import joblib

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Datasets/ytfinal.csv")

# -----------------------------
# Basic Cleaning (IMPORTANT FIX)
# -----------------------------
df = df.dropna(subset=['clean_text'])

# remove empty strings
df = df[df['clean_text'].str.strip() != '']

# ensure all are strings
df['clean_text'] = df['clean_text'].astype(str)

# optional: re-clean (safe)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

# -----------------------------
# Final check
# -----------------------------
print("Null values:\n", df.isnull().sum())
print("Empty rows:", (df['clean_text'] == '').sum())

# -----------------------------
# Features & Labels
# -----------------------------
X = df['clean_text']
y = df['label']

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)   # 🔥 improves accuracy
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

results = {}

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# -----------------------------
# Select Best Model
# -----------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]:.4f}")

# -----------------------------
# Save Model & Vectorizer
# -----------------------------
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model & vectorizer saved successfully!")