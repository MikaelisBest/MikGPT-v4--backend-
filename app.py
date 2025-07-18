from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY loaded:", bool(GROQ_API_KEY))  # Debug print

# Init app
app = Flask(__name__)

# ✅ Proper CORS setup — allow frontend domain
CORS(app, resources={r"/*": {"origins": [
    "https://mikgpt-v4-frontend-production.up.railway.app"
]}})

# Groq config
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

# Download VADER once
nltk.download('vader_lexicon')

@app.route('/api', methods=['POST'])  # ✅ Make sure your frontend calls /api not /api/chat
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            raise ValueError("Empty message received.")

        # Sentiment-based system prompt
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(user_message)

        if sentiment['compound'] < -0.5:
            system_prompt = "You are MikGPT-V4, providing supportive responses."
        elif sentiment['compound'] > 0.5:
            system_prompt = "You are MikGPT-V4, engaging enthusiastically."
        else:
            system_prompt = "You are MikGPT-V4, a friendly AI assistant."

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload)

        if response.status_code != 200:
            print("Groq Error:", response.status_code, response.text)
            response.raise_for_status()

        result = response.json()
        ai_reply = result['choices'][0]['message']['content']

        return jsonify({'status': 'success', 'reply': ai_reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
