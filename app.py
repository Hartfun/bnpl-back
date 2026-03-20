import os, string, warnings
import numpy as np
import pandas as pd
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("vader_lexicon",quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster        import KMeans
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics        import accuracy_score

app  = Flask(__name__)
CORS(app)

COL_FIELD = "What is your field of study?"
COL_FUNDS = "What is your primary source of funds for personal expenses?"
COL_USAGE = "Have you ever used a 'Buy Now, Pay Later' (BNPL) or an EMI service for a purchase?"
TEXT_COLS = [
    "What is your overall opinion of 'Buy Now, Pay Later' (BNPL) services like Slice or LazyPay?",
    "Describe a positive experience you have had using an EMI or BNPL service. What made it good?",
    "Describe a negative experience you have had, if any. What went wrong?",
    "What is your single biggest fear or concern about using these services regularly?",
]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sid        = SentimentIntensityAnalyzer()
models = {}
stats  = {}

def preprocess(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"): return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

def vader_label(score):
    if score >=  0.05: return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"

def train():
    global models, stats
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Timestamp"], errors="ignore").drop_duplicates().reset_index(drop=True)

    for col in TEXT_COLS:
        if col in df.columns:
            df[col + "_clean"] = df[col].astype(str).apply(preprocess)
    clean_cols = [c + "_clean" for c in TEXT_COLS if c in df.columns]
    df["combined_text"] = df[clean_cols].agg(" ".join, axis=1)
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)

    df["compound_score"]  = df["combined_text"].apply(lambda x: sid.polarity_scores(str(x))["compound"])
    df["vader_sentiment"] = df["compound_score"].apply(vader_label)
    df["user_type"]       = np.where(df[COL_USAGE].str.strip().str.lower() == "yes", "User", "Non-User")

    le_field = LabelEncoder()
    le_funds = LabelEncoder()
    df["field_enc"] = le_field.fit_transform(df[COL_FIELD].fillna("Other"))
    df["funds_enc"] = le_funds.fit_transform(df[COL_FUNDS].fillna("Other"))

    feat = ["compound_score", "field_enc", "funds_enc"]
    X_ad = df[feat].fillna(0)
    y_ad = (df["user_type"] == "User").astype(int)

    X = df["combined_text"].values
    y = df["vader_sentiment"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf  = TfidfVectorizer(max_features=200, min_df=3, max_df=0.85)
    lr     = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(tfidf.fit_transform(X_tr), y_tr)
    lr_acc = accuracy_score(y_te, lr.predict(tfidf.transform(X_te)))

    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_ad, y_ad, test_size=0.2, random_state=42, stratify=y_ad)
    rf     = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight="balanced")
    rf.fit(Xa_tr, ya_tr)
    rf_acc = accuracy_score(ya_te, rf.predict(Xa_te))

    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaler.fit_transform(X_ad))

    cv     = CountVectorizer(max_features=200, min_df=3, max_df=0.85)
    gb     = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    gb.fit(cv.fit_transform(X_tr).toarray(), y_tr)
    gb_acc = accuracy_score(y_te, gb.predict(cv.transform(X_te).toarray()))

    models.update({
        "lr": lr, "tfidf": tfidf,
        "rf": rf, "scaler": scaler, "kmeans": kmeans,
        "gb": gb, "cv": cv,
        "le_field": le_field, "le_funds": le_funds,
        "fields": sorted(le_field.classes_.tolist()),
        "funds":  sorted(le_funds.classes_.tolist()),
    })

    vc = df["vader_sentiment"].value_counts()
    stats.update({
        "total":    int(len(df)),
        "users":    int((df["user_type"] == "User").sum()),
        "non_users":int((df["user_type"] == "Non-User").sum()),
        "sentiment": {"positive": int(vc.get("Positive",0)), "neutral": int(vc.get("Neutral",0)), "negative": int(vc.get("Negative",0))},
        "models":   {"lr_accuracy": round(lr_acc,3), "rf_accuracy": round(rf_acc,3), "gb_accuracy": round(gb_acc,3)},
        "clusters": [{"id":i+1,"size":int(len(df[df["cluster"]==i])),"adoption_rate":round(float((df[df["cluster"]==i]["user_type"]=="User").mean()*100),1),"avg_sentiment":round(float(df[df["cluster"]==i]["compound_score"].mean()),3)} for i in range(3)],
        "field_distribution": df[COL_FIELD].value_counts().to_dict(),
    })
    print(f"Trained {len(df)} rows | LR={lr_acc:.3f} RF={rf_acc:.3f} GB={gb_acc:.3f}")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_ready": bool(models)})

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)

@app.route("/api/fields")
def get_fields():
    return jsonify({"fields": models.get("fields",[]), "funds": models.get("funds",[])})

@app.route("/api/predict", methods=["POST"])
def predict():
    data  = request.get_json(force=True)
    text  = data.get("text","").strip()
    field = data.get("field","")
    funds = data.get("funds","")
    if not text: return jsonify({"error":"text required"}), 400

    cleaned  = preprocess(text)
    compound = sid.polarity_scores(str(cleaned))["compound"]

    v1 = models["tfidf"].transform([cleaned])
    lr_sent    = models["lr"].predict(v1)[0]
    lr_proba   = models["lr"].predict_proba(v1)[0].tolist()
    lr_classes = models["lr"].classes_.tolist()

    try:    fe = int(models["le_field"].transform([field])[0])
    except: fe = 0
    try:    fu = int(models["le_funds"].transform([funds])[0])
    except: fu = 0

    feat_row   = np.array([[compound, fe, fu]])
    adopt_prob = float(models["rf"].predict_proba(feat_row)[0][1])
    cluster    = int(models["kmeans"].predict(models["scaler"].transform(feat_row))[0]) + 1
    gb_sent    = models["gb"].predict(models["cv"].transform([cleaned]).toarray())[0]

    return jsonify({
        "vader_score":          round(compound,4),
        "vader_label":          vader_label(compound),
        "lr_sentiment":         lr_sent,
        "lr_probabilities":     dict(zip(lr_classes,[round(p,3) for p in lr_proba])),
        "adoption_probability": round(adopt_prob*100,1),
        "adoption_label":       "User" if adopt_prob>=0.5 else "Non-User",
        "cluster":              cluster,
        "gb_sentiment":         gb_sent,
    })

train()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
