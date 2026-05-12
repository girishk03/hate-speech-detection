# Hate Speech Detection

A Flask-based NLP project for detecting toxic/hate speech in text across two use cases:
- YouTube comment classification
- Chatroom moderation

## 📸 Screenshots

### 🎯 YouTube Comment Classification

![Home](docs/screenshots/01-youtube-home.png)
*YouTube comment analysis home — Sentiment Analysis · Toxicity Scoring · Keyword Highlighting*

![Results](docs/screenshots/02-youtube-results.png)
*Analysis results — 50 comments, 24 hate speech detected, 22.6% avg toxicity score*

![Comment List](docs/screenshots/04-comment-list.png)
*Comment list with POSITIVE/NEGATIVE/NEUTRAL labels and confidence scores*

### 💬 AI Polite Chat Room

![Chatroom](docs/screenshots/07-chatroom-home.png)
*Real-time AI moderated chatroom*

![Polite Conversion](docs/screenshots/09-polite-conversion.png)
*Polite version chooser — converts "i hate you" to "I dislike you"*

![Converted Message](docs/screenshots/08-chatroom-result.png)
*Message converted and sent as polite version*

---

## Features
- TF-IDF + LinearSVC text classification
- Web interface with Flask
- SocketIO-based interactive behavior
- Saved model artifacts (`best_model.pkl`, `vectorizer.pkl`)

## Model Performance
Best model: **SVM (LinearSVC)**  
Accuracy: **0.7850** (78.50%)  
Macro F1: **0.78**  
Weighted F1: **0.78**  
Test samples: **6424**

## Run (YouTube Module)
```bash
cd "Youtube comment classification"
python app.py
```
Open: `http://127.0.0.1:5001`

_Last updated: Tue May 12 00:18:13 IST 2026_
