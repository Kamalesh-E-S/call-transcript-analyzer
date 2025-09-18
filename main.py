from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import csv
import os
from datetime import datetime
from groq import Groq
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

# Ensure CSV exists
CSV_FILE = "call_analysis.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Timestamp', 'Transcript', 'Summary', 'Sentiment'])
        writer.writeheader()

# def analyze_with_groq(transcript):
#     try:
#         client = Groq(api_key=api_key)

#         # Summarize transcript
#         summary_response = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": "You are a customer service analyst. Summarize the call transcript in 2-3 sentences."},
#                 {"role": "user", "content": transcript}
#             ],
#             model="llama-3.1-8b-instant",
#             temperature=0.2,
#             max_tokens=150
#         )
#         summary = summary_response.choices[0].message.content.strip()

#         # Sentiment analysis
#         sentiment_response = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": "Analyze customer sentiment. Respond with one word: Positive, Negative, or Neutral"},
#                 {"role": "user", "content": summary}
#             ],
#             model="llama-3.1-8b-instant",
#             temperature=0.1,
#             max_tokens=5
#         )
#         sentiment = sentiment_response.choices[0].message.content.strip()

#         return summary, sentiment
#     except Exception as e:
#         return None, str(e)
def analyze_with_groq(transcript):
    try:
        client = Groq(api_key=api_key)

        # Summarize transcript
        summary_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": (
                    "You are an expert customer service analyst. "
                    "Your task is to read the call transcript and produce a concise summary. "
                    "The summary must be 2–3 sentences long, clear, and professional. "
                    "Do not include any extra commentary."
                )},
                {"role": "user", "content": transcript}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=150
        )
        summary = summary_response.choices[0].message.content.strip()

        # Sentiment analysis
        sentiment_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": (
                    "You are an AI sentiment analyzer. "
                    "Classify the customer’s overall sentiment in the transcript. "
                    "Respond with only one word: Positive, Negative, or Neutral. "
                    "Do not include any explanation."
                )},
                {"role": "user", "content": transcript}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=5
        )
        sentiment = sentiment_response.choices[0].message.content.strip()

        return summary, sentiment
    except Exception as e:
        return None, str(e)


def save_to_csv(transcript, summary, sentiment):
    data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Transcript': transcript.replace('\n', ' ').replace('\r', ''),
        'Summary': summary,
        'Sentiment': sentiment
    }
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Timestamp', 'Transcript', 'Summary', 'Sentiment'])
        writer.writerow(data)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        transcript = request.form.get('transcript', '').strip()
        if not transcript:
            error = "Please enter a transcript."
        elif len(transcript) < 20:
            error = "Transcript too short. Minimum 20 characters required."
        else:
            summary, sentiment = analyze_with_groq(transcript)
            if summary is None:
                error = f"Analysis failed: {sentiment}"
            else:
                save_to_csv(transcript, summary, sentiment)
                result = {
                    'summary': summary,
                    'sentiment': sentiment,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }   
    # Load CSV data for display
    df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    return render_template('index.html', result=result, error=error, table=df.to_dict(orient='records'))

@app.route('/download')
def download_csv():
    return send_file(CSV_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
