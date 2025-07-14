from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ ADD THIS

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import traceback
import os
from dotenv import load_dotenv

load_dotenv()
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))  

app = Flask(__name__)

from flask_cors import CORS

CORS(app, resources={r"/*": {"origins": "*"}})


# Groq API Config
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-8b-8192"

nltk.download('vader_lexicon')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            raise ValueError("Empty message received.")

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

        return jsonify({'status': 'success', 'response': ai_reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
# Dummy comment to force redeploy


