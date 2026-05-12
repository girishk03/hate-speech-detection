from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import joblib
import os
import re
import nltk
import time
import uuid
import threading
import requests
from urllib.parse import urlparse, parse_qs
from urllib.parse import quote_plus
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from comment_processor import fetch_clean_comments, classify_comments, calculate_stats

app = Flask(__name__)
app.config['SECRET_KEY'] = 'integrated-secret-key'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
analysis_jobs = {}
analysis_jobs_lock = threading.Lock()

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'))
    print('✅ Model loaded')
except Exception as e:
    print(f'❌ Model not found: {e}')
    model, vectorizer = None, None

# -----------------------------
# NLTK setup for chatroom
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

HATE_PATTERNS = [
    r'\b(stupid|idiot|dumb|moron|retard)\b',
    r'\b(hate|kill|die|worthless|useless)\b',
    r'\b(ugly|disgusting|gross|nasty)\b',
    r'\b(fuck|shit|bitch|asshole|bastard)\b',
    r'\b(racist|nazi|terrorist)\b',
]

POLITE_ALTERNATIVES = {
    'stupid': 'not very smart',
    'idiot': 'person who made a mistake',
    'dumb': 'not well-informed',
    'moron': 'person who needs to learn more',
    'retard': 'person with different abilities',
    'hate': 'dislike',
    'kill': 'harm',
    'die': 'pass away',
    'worthless': 'not valuable',
    'useless': 'not helpful',
    'ugly': 'not attractive',
    'disgusting': 'unpleasant',
    'gross': 'unpleasant',
    'nasty': 'unkind',
    'fuck': 'have relations with',
    'shit': 'stuff',
    'bitch': 'difficult person',
    'asshole': 'mean person',
    'bastard': 'person born to unmarried parents',
    'racist': 'prejudiced person',
    'nazi': 'extreme nationalist',
    'terrorist': 'extremist',
}

users = {}


def get_current_time():
    return datetime.now().strftime('%H:%M')


def fetch_video_title(url):
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        html = resp.text

        og_match = re.search(r'<meta property="og:title" content="([^"]+)"', html, re.IGNORECASE)
        if og_match:
            return og_match.group(1).strip()

        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            return title.replace('- YouTube', '').strip()

        return 'Unknown Title'
    except Exception:
        return 'Unknown Title'


def extract_video_id(url):
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if 'youtu.be' in host:
            return parsed.path.strip('/').split('/')[0] or None
        if 'youtube.com' in host:
            if parsed.path == '/watch':
                return parse_qs(parsed.query).get('v', [None])[0]
            if parsed.path.startswith('/shorts/'):
                parts = parsed.path.split('/')
                return parts[2] if len(parts) > 2 else None
            if parsed.path.startswith('/embed/'):
                parts = parsed.path.split('/')
                return parts[2] if len(parts) > 2 else None
        return None
    except Exception:
        return None


def make_comment_permalink(video_id, comment_id):
    if not video_id or not comment_id:
        return None
    encoded_cid = quote_plus(str(comment_id))
    return f"https://www.youtube.com/watch?v={video_id}&lc={encoded_cid}&feature=em-comments"


def update_job(job_id, **fields):
    with analysis_jobs_lock:
        if job_id in analysis_jobs:
            analysis_jobs[job_id].update(fields)


def run_analysis_job(job_id, url, max_comments, sort_by):
    start = time.time()
    try:
        video_id = extract_video_id(url)
        update_job(job_id, status='running', stage='fetching', progress=5)

        def on_progress(current, total):
            # fetching represents 5% -> 80%
            pct = 5 + int((current / max(total, 1)) * 75)
            update_job(
                job_id,
                stage='fetching',
                fetched_comments=current,
                progress=min(80, pct),
            )

        comments = fetch_clean_comments(
            url,
            max_comments,
            max_duration_sec=None,
            sort_by=sort_by,
            progress_callback=on_progress,
            return_records=True
        )

        if not comments:
            update_job(job_id, status='failed', error='No comments found', progress=100)
            return

        update_job(job_id, stage='classifying', progress=90, fetched_comments=len(comments))
        predictions = classify_comments(comments, model, vectorizer)
        if video_id:
            for item in predictions:
                cid = item.get('comment_id')
                item['report_url'] = make_comment_permalink(video_id, cid)
        stats = calculate_stats(predictions)

        update_job(
            job_id,
            status='completed',
            stage='done',
            progress=100,
            results=predictions,
            stats=stats,
            processed_comments=len(comments),
            elapsed_sec=round(time.time() - start, 2),
        )
    except Exception as e:
        update_job(job_id, status='failed', error=str(e), progress=100)


def detect_hate_speech(text):
    for pattern in HATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] <= -0.05:
        return 'negative'
    if scores['compound'] >= 0.05:
        return 'positive'
    return 'neutral'


def is_actually_positive(sentence):
    sentence_lower = sentence.lower()
    positive_patterns = [
        r"\b(not\s+stupid|not\s+dumb|not\s+idiot|not\s+hate|not\s+bad|not\s+terrible)\b",
        r"\b(don't\s+hate|don't\s+like|don't\s+want)\b",
        r'\b(no\s+hate|no\s+problem|no\s+issue)\b',
        r'\b(never\s+hate|never\s+bad|never\s+terrible)\b',
    ]
    for pattern in positive_patterns:
        if re.search(pattern, sentence_lower):
            return True
    return analyze_sentiment(sentence) == 'positive'


def generate_polite_alternatives(sentence):
    alt1 = sentence.lower()
    for hate_word, polite_word in POLITE_ALTERNATIVES.items():
        pattern = r'\b' + re.escape(hate_word) + r'\b'
        alt1 = re.sub(pattern, polite_word, alt1, flags=re.IGNORECASE)

    alt1 = re.sub(r"\b(you are|you're)\s+(stupid|dumb|idiot)\b", 'you seem to be having difficulty understanding', alt1)
    alt1 = re.sub(r"\b(i hate|i can't stand)\b", 'i strongly dislike', alt1)
    alt1 = re.sub(r'\b(this is|that is)\s+(terrible|awful|horrible)\b', 'this could be improved', alt1)

    alt2 = sentence.lower()
    alt2 = re.sub(r"\b(you are|you're)\s+(stupid|dumb|idiot)\b", 'you have potential to learn and grow', alt2)
    alt2 = re.sub(r"\b(i hate|i can't stand)\b", 'i prefer to avoid', alt2)
    alt2 = re.sub(r'\b(this is|that is)\s+(terrible|awful|horrible)\b', 'this has room for improvement', alt2)
    alt2 = re.sub(r'\b(kill|die)\b', 'stop', alt2)

    if alt1:
        alt1 = alt1[0].upper() + alt1[1:]
    if alt2:
        alt2 = alt2[0].upper() + alt2[1:]

    return [alt1, alt2]


def process_message(message):
    sentences = sent_tokenize(message)
    was_converted = False
    conversion_options = []

    for sentence in sentences:
        if is_actually_positive(sentence):
            conversion_options.append(None)
        elif detect_hate_speech(sentence) or analyze_sentiment(sentence) == 'negative':
            alternatives = generate_polite_alternatives(sentence)
            if alternatives[0] != sentence:
                conversion_options.append(alternatives)
                was_converted = True
            else:
                conversion_options.append(None)
        else:
            conversion_options.append(None)

    return {
        'original_message': message,
        'was_converted': was_converted,
        'conversion_options': conversion_options,
        'sentences': sentences,
    }


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    video_title = fetch_video_title(url)
    video_id = extract_video_id(url)
    sort_by = data.get('sort_by', 'newest')

    try:
        max_comments = int(data.get('max_comments', 50))
    except (TypeError, ValueError):
        max_comments = 50

    max_comments = max(1, min(100000, max_comments))
    if sort_by not in {'top', 'newest'}:
        sort_by = 'newest'

    if not url:
        return jsonify({'success': False, 'error': 'Please enter a YouTube URL'})

    if model is None or vectorizer is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please train/load model first.'})

    try:
        comments = fetch_clean_comments(url, max_comments, sort_by=sort_by, return_records=True)
        if not comments:
            return jsonify({'success': False, 'error': 'No comments found'})

        predictions = classify_comments(comments, model, vectorizer)
        if video_id:
            for item in predictions:
                cid = item.get('comment_id')
                item['report_url'] = make_comment_permalink(video_id, cid)
        stats = calculate_stats(predictions)
        return jsonify({
            'success': True,
            'results': predictions,
            'stats': stats,
            'requested_comments': max_comments,
            'processed_comments': len(comments),
            'sort_by': sort_by,
            'video_title': video_title,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/analyze/start', methods=['POST'])
def analyze_start():
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    video_title = fetch_video_title(url)
    sort_by = data.get('sort_by', 'newest')

    try:
        max_comments = int(data.get('max_comments', 50))
    except (TypeError, ValueError):
        max_comments = 50
    max_comments = max(1, min(100000, max_comments))
    if sort_by not in {'top', 'newest'}:
        sort_by = 'newest'

    if not url:
        return jsonify({'success': False, 'error': 'Please enter a YouTube URL'})
    if model is None or vectorizer is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please train/load model first.'})

    job_id = str(uuid.uuid4())
    with analysis_jobs_lock:
        analysis_jobs[job_id] = {
            'status': 'queued',
            'stage': 'queued',
            'progress': 0,
            'requested_comments': max_comments,
            'video_title': video_title,
            'sort_by': sort_by,
            'fetched_comments': 0,
            'processed_comments': 0,
            'results': [],
            'stats': None,
            'error': None,
        }

    thread = threading.Thread(target=run_analysis_job, args=(job_id, url, max_comments, sort_by), daemon=True)
    thread.start()
    return jsonify({'success': True, 'job_id': job_id})


@app.route('/analyze/status/<job_id>', methods=['GET'])
def analyze_status(job_id):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if not job:
            return jsonify({'success': False, 'error': 'Invalid job id'})
        return jsonify({'success': True, **job})


# -----------------------------
# Socket.IO events (Chatroom)
# -----------------------------
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')


@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in users:
        del users[request.sid]
        emit('user_list', list(users.values()), broadcast=True)


@socketio.on('join')
def handle_join(username):
    users[request.sid] = username
    emit('user_list', list(users.values()), broadcast=True)


@socketio.on('send_message')
def handle_message(message):
    username = users.get(request.sid, 'Anonymous')
    processed = process_message(message)

    if processed['was_converted']:
        emit('conversion_options', {
            'message_id': f"{request.sid}_{os.urandom(4).hex()}",
            'original_message': processed['original_message'],
            'conversion_options': processed['conversion_options'],
            'sentences': processed['sentences'],
        })
    else:
        emit('receive_message', {
            'user': username,
            'original_message': processed['original_message'],
            'converted_message': processed['original_message'],
            'was_converted': False,
            'time': get_current_time(),
        }, broadcast=True)


@socketio.on('choose_conversion')
def handle_choose_conversion(data):
    username = users.get(request.sid, 'Anonymous')
    chosen_alternatives = data.get('chosen_alternatives', [])
    sentences = data.get('sentences', [])

    final_message_parts = []
    for i, sentence in enumerate(sentences):
        if i < len(chosen_alternatives) and chosen_alternatives[i] is not None:
            final_message_parts.append(chosen_alternatives[i])
        else:
            final_message_parts.append(sentence)

    final_message = ' '.join(final_message_parts)
    emit('receive_message', {
        'user': username,
        'original_message': data.get('original_message', ''),
        'converted_message': final_message,
        'was_converted': True,
        'time': get_current_time(),
    }, broadcast=True)


@socketio.on('typing')
def handle_typing():
    username = users.get(request.sid, 'Anonymous')
    emit('user_typing', {'username': username}, broadcast=True, include_self=False)


@socketio.on('stop_typing')
def handle_stop_typing():
    emit('user_stop_typing', broadcast=True, include_self=False)


if __name__ == '__main__':
    print('🚀 Running Integrated App...')
    print('👉 http://localhost:5001')
    socketio.run(app, debug=True, port=5001)
