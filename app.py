import os
import string
import warnings
import numpy as np
import pandas as pd
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

warnings.filterwarnings("ignore")

# ── NLTK downloads ──────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)
CORS(app)

# ── Globals ──────────────────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

models = {}   # populated in train_models()
stats  = {}   # summary stats for /api/stats

# ── Text helpers ─────────────────────────────────────────────────────────────
def preprocess_text(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def vader_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

def likert_sentiment(val, reverse=False):
    try:
        val = int(val)
        if reverse:
            return "Positive" if val <= 2 else ("Neutral" if val == 3 else "Negative")
        return "Positive" if val >= 4 else ("Neutral" if val == 3 else "Negative")
    except Exception:
        return "Unknown"

# ── Training ─────────────────────────────────────────────────────────────────
def train_models():
    global models, stats

    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Timestamp"], errors="ignore").drop_duplicates().reset_index(drop=True)

    text_cols = [
        "What is your overall opinion of 'Buy Now, Pay Later' (BNPL) services like Slice or LazyPay?",
        "Describe a positive experience you have had using an EMI or BNPL service. What made it good?",
        "Describe a negative experience you have had, if any. What went wrong?",
        "What is your single biggest fear or concern about using these services regularly?",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col + "_clean"] = df[col].astype(str).apply(preprocess_text)

    clean_cols = [c + "_clean" for c in text_cols if c in df.columns]
    df["combined_text"] = df[clean_cols].agg(" ".join, axis=1)
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)

    # VADER
    df["compound_score"] = df["combined_text"].apply(
        lambda x: sid.polarity_scores(str(x))["compound"]
    )
    df["vader_sentiment"] = df["compound_score"].apply(vader_label)

    users_col = "Have you ever used a 'Buy Now, Pay Later' (BNPL) or an EMI service for a purchase?"
    users_mask = df[users_col].str.strip().str.lower() == "yes"
    df["user_type"] = np.where(users_mask, "User", "Non-User")

    # ── Model 1: Logistic Regression (Sentiment) ────────────────────────────
    X = df["combined_text"].values
    y = df["vader_sentiment"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf = TfidfVectorizer(max_features=100, min_df=5, max_df=0.8)
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr_vec, y_tr)
    lr_acc = accuracy_score(y_te, lr.predict(X_te_vec))

    # ── Model 2: Random Forest (Adoption) ──────────────────────────────────
    le_field = LabelEncoder()
    le_funds = LabelEncoder()
    df["field_encoded"] = le_field.fit_transform(df["What is your field of study?"])
    df["funds_encoded"] = le_funds.fit_transform(
        df["What is your primary source of funds for personal expenses?"]
    )

    feat = ["compound_score", "field_encoded", "funds_encoded"]
    X_ad = df[feat].fillna(0)
    y_ad = (df["user_type"] == "User").astype(int)

    X_ad_tr, X_ad_te, y_ad_tr, y_ad_te = train_test_split(
        X_ad, y_ad, test_size=0.2, random_state=42, stratify=y_ad
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight="balanced")
    rf.fit(X_ad_tr, y_ad_tr)
    rf_acc = accuracy_score(y_ad_te, rf.predict(X_ad_te))

    # ── Model 3: K-Means (Segmentation) ────────────────────────────────────
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(X_ad)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(feat_scaled)

    # ── Model 4: Gradient Boosting (Sentiment) ──────────────────────────────
    cv = CountVectorizer(max_features=100, min_df=5, max_df=0.8)
    X_gb_tr = cv.fit_transform(X_tr).toarray()
    X_gb_te = cv.transform(X_te).toarray()
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    gb.fit(X_gb_tr, y_tr)
    gb_acc = accuracy_score(y_te, gb.predict(X_gb_te))

    # Store everything
    models.update({
        "lr": lr, "tfidf": tfidf,
        "rf": rf, "le_field": le_field, "le_funds": le_funds, "scaler": scaler,
        "kmeans": kmeans,
        "gb": gb, "cv": cv,
        "df": df,
        "fields": list(le_field.classes_),
        "funds": list(le_funds.classes_),
    })

    # Summary stats
    vc = df["vader_sentiment"].value_counts()
    cluster_info = []
    for i in range(3):
        c = df[df["cluster"] == i]
        cluster_info.append({
            "id": i + 1,
            "size": int(len(c)),
            "adoption_rate": round(float((c["user_type"] == "User").mean() * 100), 1),
            "avg_sentiment": round(float(c["compound_score"].mean()), 3),
        })

    stats.update({
        "total": int(len(df)),
        "users": int(users_mask.sum()),
        "non_users": int((~users_mask).sum()),
        "sentiment": {
            "positive": int(vc.get("Positive", 0)),
            "neutral": int(vc.get("Neutral", 0)),
            "negative": int(vc.get("Negative", 0)),
        },
        "models": {
            "lr_accuracy": round(lr_acc, 3),
            "rf_accuracy": round(rf_acc, 3),
            "gb_accuracy": round(gb_acc, 3),
        },
        "clusters": cluster_info,
        "field_distribution": df["What is your field of study?"].value_counts().to_dict(),
    })

    print(f"✓ Models trained | LR={lr_acc:.3f} RF={rf_acc:.3f} GB={gb_acc:.3f}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_ready": bool(models)})

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    field = data.get("field", "")
    funds = data.get("funds", "")

    if not text:
        return jsonify({"error": "text is required"}), 400

    cleaned = preprocess_text(text)
    compound = sid.polarity_scores(str(cleaned))["compound"]
    sentiment_label = vader_label(compound)

    # Model 1 – Logistic Regression sentiment
    vec1 = models["tfidf"].transform([cleaned])
    lr_sentiment = models["lr"].predict(vec1)[0]
    lr_proba = models["lr"].predict_proba(vec1)[0].tolist()
    lr_classes = models["lr"].classes_.tolist()

    # Model 2 – Random Forest adoption
    try:
        field_enc = models["le_field"].transform([field])[0]
    except Exception:
        field_enc = 0
    try:
        funds_enc = models["le_funds"].transform([funds])[0]
    except Exception:
        funds_enc = 0

    feat_row = np.array([[compound, field_enc, funds_enc]])
    adopt_prob = float(models["rf"].predict_proba(feat_row)[0][1])
    adopt_label = "User" if adopt_prob >= 0.5 else "Non-User"

    # Model 3 – Cluster
    feat_scaled = models["scaler"].transform(feat_row)
    cluster = int(models["kmeans"].predict(feat_scaled)[0]) + 1

    # Model 4 – Gradient Boosting sentiment
    vec2 = models["cv"].transform([cleaned]).toarray()
    gb_sentiment = models["gb"].predict(vec2)[0]

    return jsonify({
        "vader_score": round(compound, 4),
        "vader_label": sentiment_label,
        "lr_sentiment": lr_sentiment,
        "lr_probabilities": dict(zip(lr_classes, [round(p, 3) for p in lr_proba])),
        "adoption_probability": round(adopt_prob * 100, 1),
        "adoption_label": adopt_label,
        "cluster": cluster,
        "gb_sentiment": gb_sentiment,
    })

@app.route("/api/fields")
def get_fields():
    return jsonify({"fields": models.get("fields", []), "funds": models.get("funds", [])})

# ── Start ─────────────────────────────────────────────────────────────────────
train_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
