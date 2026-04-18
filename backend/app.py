from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@app.route("/")
def home():
    return "API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print("Received:", data)

    text = data["text"]

    clean = clean_text(text)
    vector = vectorizer.transform([clean])

    result = model.predict(vector)[0]

    return jsonify({
        "result": "Real News" if result == 1 else "Fake News"
    })

app.run()