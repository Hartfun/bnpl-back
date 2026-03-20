import os, string, json, pickle, warnings
import numpy as np
import pandas as pd
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("vader_lexicon", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sid        = SentimentIntensityAnalyzer()

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def preprocess_text(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def vader_label(score):
    if score >= 0.05:  return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"

def load_pkl(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as f:
        return pickle.load(f)

print("Loading pre-trained models from pkl files...")
lr       = load_pkl("lr_sentiment.pkl")
tfidf    = load_pkl("tfidf.pkl")
rf       = load_pkl("rf_adoption.pkl")
le_field = load_pkl("le_field.pkl")
le_funds = load_pkl("le_funds.pkl")
scaler   = load_pkl("scaler.pkl")
kmeans   = load_pkl("kmeans.pkl")
gb       = load_pkl("gb_sentiment.pkl")
cv       = load_pkl("count_vectorizer.pkl")

with open(os.path.join(MODELS_DIR, "meta.json")) as f:
    meta = json.load(f)

stats_cache = meta["stats"]
fields_list = meta["fields"]
funds_list  = meta["funds"]
print("All models loaded instantly from pkl!")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_ready": True, "source": "pkl"})

@app.route("/api/stats")
def get_stats():
    return jsonify(stats_cache)

@app.route("/api/fields")
def get_fields():
    return jsonify({"fields": fields_list, "funds": funds_list})

@app.route("/api/predict", methods=["POST"])
def predict():
    data  = request.get_json(force=True)
    text  = data.get("text", "").strip()
    field = data.get("field", "")
    funds = data.get("funds", "")

    if not text:
        return jsonify({"error": "text is required"}), 400

    cleaned  = preprocess_text(text)
    compound = sid.polarity_scores(str(cleaned))["compound"]

    vec1         = tfidf.transform([cleaned])
    lr_sentiment = lr.predict(vec1)[0]
    lr_proba     = lr.predict_proba(vec1)[0].tolist()
    lr_classes   = lr.classes_.tolist()

    try: field_enc = int(le_field.transform([field])[0])
    except: field_enc = 0
    try: funds_enc = int(le_funds.transform([funds])[0])
    except: funds_enc = 0

    feat_row    = np.array([[compound, field_enc, funds_enc]])
    adopt_prob  = float(rf.predict_proba(feat_row)[0][1])
    feat_scaled = scaler.transform(feat_row)
    cluster     = int(kmeans.predict(feat_scaled)[0]) + 1
    vec2        = cv.transform([cleaned]).toarray()
    gb_sentiment = gb.predict(vec2)[0]

    return jsonify({
        "vader_score":          round(compound, 4),
        "vader_label":          vader_label(compound),
        "lr_sentiment":         lr_sentiment,
        "lr_probabilities":     dict(zip(lr_classes, [round(p, 3) for p in lr_proba])),
        "adoption_probability": round(adopt_prob * 100, 1),
        "adoption_label":       "User" if adopt_prob >= 0.5 else "Non-User",
        "cluster":              cluster,
        "gb_sentiment":         gb_sentiment,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
