# AI Polite Chat Room

A real-time chat application with AI-powered hate speech detection and automatic conversion to polite language.

## 🚀 Features

- **Real-time Chat**: Multi-user chat with Socket.io
- **Hate Speech Detection**: Automatically detects offensive language
- **Smart Conversion**: Converts hate speech to polite alternatives
- **Mixed Sentence Handling**: Processes only negative sentences, leaves positive ones unchanged
- **Modern UI**: Clean, responsive interface with Tailwind CSS
- **User Management**: Shows online users and typing indicators

## 🛠️ Installation

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python app.py
   ```

3. **Access the Chat Room:**
   Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
hate/
├── app.py                 # Flask backend with Socket.io
├── requirements.txt       # Python dependencies
├── static/
│   └── chatroom.html     # Frontend interface
└── README.md             # This file
```

## 🎯 How It Works

### Hate Speech Detection
- Uses pattern matching for common hate words
- Sentiment analysis with NLTK's VADER
- Sentence-level processing

### Text Conversion Logic
- **Hate Speech**: Replaces offensive words with polite alternatives
- **Negative Sentiment**: Converts to more constructive language
- **Positive Content**: Leaves positive sentences unchanged
- **Mixed Messages**: Processes only the negative parts

### Example Conversions

| Original | Converted |
|----------|-----------|
| "You are stupid" | "You seem to be having difficulty understanding" |
| "I hate this" | "I strongly dislike this" |
| "This is terrible" | "This could be improved" |
| "Great job! You're an idiot" | "Great job! You seem to be having difficulty understanding" |

## 🔧 Technical Details

- **Backend**: Flask with Socket.io for real-time communication
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **AI Processing**: NLTK for sentiment analysis and tokenization
- **Real-time**: WebSocket connections via Socket.io

## 🌐 Usage

1. Open the application in your browser
2. Enter a username to join the chat
3. Type messages and send them
4. Watch as hate speech is automatically detected and converted
5. See converted messages highlighted in green

## 📊 Features Breakdown

### Chat Features
- Real-time messaging
- Online user list
- Typing indicators
- Message timestamps
- Responsive design

### AI Features
- Pattern-based hate speech detection
- Sentiment analysis
- Intelligent text conversion
- Context-aware processing
- Mixed sentence handling

### UI Features
- Modern, clean interface
- Mobile responsive
- Smooth animations
- Color-coded messages
- User-friendly interactions

## 🔮 Future Enhancements

- Machine learning model for better detection
- Multi-language support
- Custom conversion rules
- User preferences
- Chat history
- Admin controls

## 📝 Notes

- The application uses a simplified hate speech detection system
- Conversion rules can be customized in the `POLITE_ALTERNATIVES` dictionary
- The system processes messages in real-time as they are sent
- All processing happens on the server side for consistency
