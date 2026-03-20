import os, string, warnings, json, pickle
import numpy as np
import pandas as pd
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")
nltk.download("stopwords",     quiet=True)
nltk.download("wordnet",       quiet=True)
nltk.download("vader_lexicon", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sid        = SentimentIntensityAnalyzer()

# ── load pre-trained models from models/ folder ──────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "models")

def load(name):
    with open(os.path.join(BASE, name), "rb") as f:
        return pickle.load(f)

print("Loading models...")
lr     = load("lr_sentiment.pkl")
tfidf  = load("tfidf.pkl")
rf     = load("rf_adoption.pkl")
le_field = load("le_field.pkl")
le_funds = load("le_funds.pkl")
scaler = load("scaler.pkl")
kmeans = load("kmeans.pkl")
gb     = load("gb_sentiment.pkl")
cv     = load("count_vectorizer.pkl")

with open(os.path.join(BASE, "meta.json")) as f:
    meta = json.load(f)

stats  = meta["stats"]
fields = meta["fields"]
funds  = meta["funds"]
print(f"✓ Models loaded | LR={stats['models']['lr_accuracy']} RF={stats['models']['rf_accuracy']} GB={stats['models']['gb_accuracy']}")

# ── helpers ──────────────────────────────────────────────────────────────────
def preprocess(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"): return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

def vader_label(score):
    if score >=  0.05: return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_ready": True})

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)

@app.route("/api/fields")
def get_fields():
    return jsonify({"fields": fields, "funds": funds})

@app.route("/api/predict", methods=["POST"])
def predict():
    data  = request.get_json(force=True)
    text  = data.get("text", "").strip()
    field = data.get("field", "")
    fund  = data.get("funds", "")
    if not text:
        return jsonify({"error": "text required"}), 400

    cleaned  = preprocess(text)
    compound = sid.polarity_scores(str(cleaned))["compound"]

    # Model 1 — LR sentiment
    v1         = tfidf.transform([cleaned])
    lr_sent    = lr.predict(v1)[0]
    lr_proba   = lr.predict_proba(v1)[0].tolist()
    lr_classes = lr.classes_.tolist()

    # Model 2 — RF adoption
    try:    fe = int(le_field.transform([field])[0])
    except: fe = 0
    try:    fu = int(le_funds.transform([fund])[0])
    except: fu = 0

    feat_row   = np.array([[compound, fe, fu]])
    adopt_prob = float(rf.predict_proba(feat_row)[0][1])
    adopt_lbl  = "User" if adopt_prob >= 0.5 else "Non-User"

    # Model 3 — cluster
    cluster = int(kmeans.predict(scaler.transform(feat_row))[0]) + 1

    # Model 4 — GB sentiment
    gb_sent = gb.predict(cv.transform([cleaned]).toarray())[0]

    return jsonify({
        "vader_score":          round(compound, 4),
        "vader_label":          vader_label(compound),
        "lr_sentiment":         lr_sent,
        "lr_probabilities":     dict(zip(lr_classes, [round(p, 3) for p in lr_proba])),
        "adoption_probability": round(adopt_prob * 100, 1),
        "adoption_label":       adopt_lbl,
        "cluster":              cluster,
        "gb_sentiment":         gb_sent,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
