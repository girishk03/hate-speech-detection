from flask import Flask, render_template, send_from_directory, request, jsonify, session
from flask_socketio import SocketIO, emit
import joblib, os, re, nltk, time, uuid, threading, requests
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'combined-secret-key'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Download NLTK data when setup has not already installed it
NLTK_RESOURCES = {
    'vader_lexicon': 'sentiment/vader_lexicon.zip',
}
for package, resource_path in NLTK_RESOURCES.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)

# Load YouTube model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'))
    MODEL_LOADED = True
except Exception as e:
    print(f"Model load failed: {e}")
    MODEL_LOADED = False

sia = SentimentIntensityAnalyzer()

# ─── ROUTES ───────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/youtube')
def youtube():
    return render_template('index.html')

@app.route('/chatroom')
def chatroom():
    return send_from_directory('static', 'chatroom.html')

# ─── YOUTUBE ANALYSIS ─────────────────────────────────────
analysis_jobs = {}
analysis_jobs_lock = threading.Lock()

def classify_text(text):
    if not MODEL_LOADED:
        return 'UNKNOWN', 0.5
    try:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        return str(pred), 0.85
    except:
        return 'UNKNOWN', 0.5

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    url = str(data.get('url') or '').strip()
    if not url:
        return jsonify({'success': False, 'error': 'A YouTube URL is required'}), 400

    job_id = str(uuid.uuid4())
    with analysis_jobs_lock:
        analysis_jobs[job_id] = {'status': 'processing', 'results': [], 'progress': 0}
    threading.Thread(target=run_analysis, args=(job_id, url), daemon=True).start()
    return jsonify({'job_id': job_id})

@app.route('/analyze/start', methods=['POST'])
def analyze_start():
    return analyze()

@app.route('/analyze/status/<job_id>')
def analyze_status(job_id):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id, {'status': 'not_found'})
    return jsonify(job)

def run_analysis(job_id, url):
    try:
        from comment_processor import fetch_clean_comments
        comments = fetch_clean_comments(url, max_comments=50)
    except Exception as e:
        comments = [f"Sample comment {i}" for i in range(10)]

    results = []
    for i, comment in enumerate(comments):
        label, conf = classify_text(comment)
        sentiment = sia.polarity_scores(comment)
        results.append({
            'text': comment, 'label': label,
            'confidence': conf, 'sentiment': sentiment
        })
        with analysis_jobs_lock:
            analysis_jobs[job_id]['progress'] = int((i+1)/len(comments)*100)
            analysis_jobs[job_id]['results'] = results

    with analysis_jobs_lock:
        analysis_jobs[job_id]['status'] = 'done'

# ─── CHATROOM SOCKETIO ────────────────────────────────────
POLITE_ALTERNATIVES = {
    'hate': ['dislike', 'disagree with'],
    'stupid': ['mistaken', 'uninformed'],
    'idiot': ['person who made an error'],
    'kill': ['defeat', 'overcome'],
    'ugly': ['unconventional looking'],
}

def make_polite(text):
    polite = text
    for rude, alts in POLITE_ALTERNATIVES.items():
        polite = re.sub(rude, random.choice(alts), polite, flags=re.IGNORECASE)
    return polite

def is_toxic(text):
    scores = sia.polarity_scores(text)
    return scores['compound'] < -0.3

@socketio.on('send_message')
def handle_message(data):
    text = data.get('message', '')
    username = data.get('username', 'User')
    toxic = is_toxic(text)
    polite = make_polite(text) if toxic else text
    emit('receive_message', {
        'username': username, 'message': text,
        'polite_version': polite, 'is_toxic': toxic,
        'timestamp': datetime.now().strftime('%H:%M')
    }, broadcast=True)

@socketio.on('connect')
def on_connect():
    emit('status', {'msg': 'Connected'})

if __name__ == '__main__':
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        allow_unsafe_werkzeug=True,
    )
