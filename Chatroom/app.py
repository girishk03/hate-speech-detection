from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Hate speech patterns (simplified for demonstration)
HATE_PATTERNS = [
    r'\b(stupid|idiot|dumb|moron|retard)\b',
    r'\b(hate|kill|die|worthless|useless)\b',
    r'\b(ugly|disgusting|gross|nasty)\b',
    r'\b(fuck|shit|bitch|asshole|bastard)\b',
    r'\b(racist|nazi|terrorist)\b'
]

# Polite alternatives
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
    'terrorist': 'extremist'
}

def detect_hate_speech(text):
    """Detect if text contains hate speech patterns"""
    for pattern in HATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    scores = sid.polarity_scores(text)
    
    if scores['compound'] <= -0.05:
        return 'negative'
    elif scores['compound'] >= 0.05:
        return 'positive'
    else:
        return 'neutral'

def is_actually_positive(sentence):
    """Check if sentence is actually positive despite containing negative words"""
    sentence_lower = sentence.lower()
    
    # Patterns that indicate positive sentiment despite negative words
    positive_patterns = [
        r'\b(not\s+stupid|not\s+dumb|not\s+idiot|not\s+hate|not\s+bad|not\s+terrible)\b',
        r'\b(don\'t\s+hate|don\'t\s+like|don\'t\s+want)\b',
        r'\b(no\s+hate|no\s+problem|no\s+issue)\b',
        r'\b(never\s+hate|never\s+bad|never\s+terrible)\b'
    ]
    
    # Check if sentence contains positive negation patterns
    for pattern in positive_patterns:
        if re.search(pattern, sentence_lower):
            print(f"✅ Positive pattern detected: {sentence}")
            return True
    
    # Check overall sentiment
    sentiment = analyze_sentiment(sentence)
    print(f"📊 Sentiment for '{sentence}': {sentiment}")
    
    # If sentiment is positive, don't convert
    if sentiment == 'positive':
        return True
    
    return False

def generate_polite_alternatives(sentence):
    """Generate 2 polite alternatives for hate speech"""
    alternatives = []
    
    # Alternative 1: Direct word replacement
    alt1 = sentence.lower()
    for hate_word, polite_word in POLITE_ALTERNATIVES.items():
        pattern = r'\b' + re.escape(hate_word) + r'\b'
        alt1 = re.sub(pattern, polite_word, alt1, flags=re.IGNORECASE)
    
    # Additional transformations for alternative 1
    alt1 = re.sub(r'\b(you are|you\'re)\s+(stupid|dumb|idiot)\b', 
                  'you seem to be having difficulty understanding', alt1)
    alt1 = re.sub(r'\b(i hate|i can\'t stand)\b', 'i strongly dislike', alt1)
    alt1 = re.sub(r'\b(this is|that is)\s+(terrible|awful|horrible)\b', 
                  'this could be improved', alt1)
    
    # Alternative 2: More constructive approach
    alt2 = sentence.lower()
    alt2 = re.sub(r'\b(you are|you\'re)\s+(stupid|dumb|idiot)\b', 
                  'you have potential to learn and grow', alt2)
    alt2 = re.sub(r'\b(i hate|i can\'t stand)\b', 'i prefer to avoid', alt2)
    alt2 = re.sub(r'\b(this is|that is)\s+(terrible|awful|horrible)\b', 
                  'this has room for improvement', alt2)
    alt2 = re.sub(r'\b(kill|die)\b', 'stop', alt2)
    
    # Capitalize first letter
    if alt1:
        alt1 = alt1[0].upper() + alt1[1:]
    if alt2:
        alt2 = alt2[0].upper() + alt2[1:]
    
    alternatives.append(alt1)
    alternatives.append(alt2)
    
    return alternatives

def convert_to_polite(text):
    """Convert hate speech to polite language with options"""
    return generate_polite_alternatives(text)

def process_message(message):
    """Process message to detect and convert hate speech with options"""
    sentences = sent_tokenize(message)
    processed_sentences = []
    was_converted = False
    original_sentences = []
    conversion_options = []
    
    print(f"🔍 Processing message: {message}")
    print(f"📝 Sentences: {sentences}")
    
    for i, sentence in enumerate(sentences):
        original_sentences.append(sentence)
        
        # Check if sentence is actually positive despite negative words
        if is_actually_positive(sentence):
            print(f"✅ Sentence {i} is positive: {sentence}")
            processed_sentences.append(sentence)
            conversion_options.append(None)
        # Check for hate speech patterns
        elif detect_hate_speech(sentence):
            print(f"⚠️ Hate speech detected in sentence {i}: {sentence}")
            # Generate conversion options
            alternatives = generate_polite_alternatives(sentence)
            conversion_options.append(alternatives)
            processed_sentences.append(sentence)  # Keep original for now
            was_converted = True
        else:
            # Check sentiment for negative content
            sentiment = analyze_sentiment(sentence)
            print(f"📊 Sentence {i} sentiment: {sentiment}")
            if sentiment == 'negative':
                print(f"⚠️ Negative sentiment in sentence {i}: {sentence}")
                # Generate conversion options
                alternatives = generate_polite_alternatives(sentence)
                if alternatives[0] != sentence:  # Only if conversion actually changes something
                    conversion_options.append(alternatives)
                    processed_sentences.append(sentence)  # Keep original for now
                    was_converted = True
                else:
                    processed_sentences.append(sentence)
                    conversion_options.append(None)
            else:
                print(f"✅ Sentence {i} is neutral/positive: {sentence}")
                processed_sentences.append(sentence)
                conversion_options.append(None)
    
    result = {
        'original_message': message,
        'was_converted': was_converted,
        'conversion_options': conversion_options,
        'sentences': original_sentences
    }
    
    print(f"📤 Processing result: {result}")
    return result

# Store connected users
users = {}

@app.route('/')
def index():
    return app.send_static_file('chatroom.html')

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in users:
        username = users[request.sid]
        del users[request.sid]
        emit('user_list', list(users.values()), broadcast=True)
        print(f'Client disconnected: {username}')

@socketio.on('join')
def handle_join(username):
    users[request.sid] = username
    emit('user_list', list(users.values()), broadcast=True)
    print(f'User joined: {username}')

@socketio.on('send_message')
def handle_message(message):
    username = users.get(request.sid, 'Anonymous')
    
    # Process the message for hate speech detection and conversion
    processed = process_message(message)
    
    # Send conversion options to the sender only
    if processed['was_converted']:
        emit('conversion_options', {
            'message_id': f"{request.sid}_{hash(message)}",
            'original_message': processed['original_message'],
            'conversion_options': processed['conversion_options'],
            'sentences': processed['sentences']
        })
    else:
        # If no conversion needed, send message directly to all
        emit('receive_message', {
            'user': username,
            'original_message': processed['original_message'],
            'converted_message': processed['original_message'],
            'was_converted': False,
            'time': get_current_time()
        }, broadcast=True)

@socketio.on('choose_conversion')
def handle_choose_conversion(data):
    username = users.get(request.sid, 'Anonymous')
    message_id = data['message_id']
    chosen_alternatives = data['chosen_alternatives']
    
    # Reconstruct the message with chosen alternatives
    sentences = data['sentences']
    final_message_parts = []
    
    for i, sentence in enumerate(sentences):
        if i < len(chosen_alternatives) and chosen_alternatives[i] is not None:
            final_message_parts.append(chosen_alternatives[i])
        else:
            final_message_parts.append(sentence)
    
    final_message = ' '.join(final_message_parts)
    
    # Send the converted message to all users
    emit('receive_message', {
        'user': username,
        'original_message': data['original_message'],
        'converted_message': final_message,
        'was_converted': True,
        'time': get_current_time()
    }, broadcast=True)

@socketio.on('typing')
def handle_typing():
    username = users.get(request.sid, 'Anonymous')
    emit('user_typing', {'username': username}, broadcast=True, include_self=False)

@socketio.on('stop_typing')
def handle_stop_typing():
    emit('user_stop_typing', broadcast=True, include_self=False)

def get_current_time():
    """Get current time in HH:MM format"""
    from datetime import datetime
    return datetime.now().strftime('%H:%M')

if __name__ == '__main__':
    print("🚀 Starting AI Polite Chat Room...")
    print("👉 http://localhost:5002")
    socketio.run(app, debug=False, host='0.0.0.0', port=5002)
