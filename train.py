"""
Run this ONCE locally to generate all .pkl files.
Then commit them to your repo — the API loads them instead of retraining.

Usage:
    cd bnpl-back
    pip install -r requirements.txt
    python save_models.py
"""

import os, string, warnings
import numpy as np
import pandas as pd
import pickle
import nltk

warnings.filterwarnings("ignore")

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
from sklearn.metrics import accuracy_score

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

def preprocess_text(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def vader_label(score):
    if score >= 0.05: return "Positive"
    elif score <= -0.05: return "Negative"
    return "Neutral"

print("📂 Loading data...")
df = pd.read_csv("data.csv")
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

df["compound_score"] = df["combined_text"].apply(
    lambda x: sid.polarity_scores(str(x))["compound"]
)
df["vader_sentiment"] = df["compound_score"].apply(vader_label)

users_col = "Have you ever used a 'Buy Now, Pay Later' (BNPL) or an EMI service for a purchase?"
users_mask = df[users_col].str.strip().str.lower() == "yes"
df["user_type"] = np.where(users_mask, "User", "Non-User")

# ── Model 1: Logistic Regression ────────────────────────────────────────────
print("🤖 Training Logistic Regression...")
X = df["combined_text"].values
y = df["vader_sentiment"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=100, min_df=5, max_df=0.8)
X_tr_vec = tfidf.fit_transform(X_tr)
X_te_vec = tfidf.transform(X_te)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_tr_vec, y_tr)
print(f"   ✓ LR accuracy: {accuracy_score(y_te, lr.predict(X_te_vec)):.3f}")

# ── Model 2: Random Forest ───────────────────────────────────────────────────
print("🌲 Training Random Forest...")
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
print(f"   ✓ RF accuracy: {accuracy_score(y_ad_te, rf.predict(X_ad_te)):.3f}")

# ── Model 3: KMeans ──────────────────────────────────────────────────────────
print("📍 Training KMeans...")
scaler = StandardScaler()
feat_scaled = scaler.fit_transform(X_ad)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(feat_scaled)
print(f"   ✓ KMeans inertia: {kmeans.inertia_:.2f}")

# ── Model 4: Gradient Boosting ───────────────────────────────────────────────
print("🚀 Training Gradient Boosting...")
cv = CountVectorizer(max_features=100, min_df=5, max_df=0.8)
X_gb_tr = cv.fit_transform(X_tr).toarray()
X_gb_te = cv.transform(X_te).toarray()
gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb.fit(X_gb_tr, y_tr)
print(f"   ✓ GB accuracy: {accuracy_score(y_te, gb.predict(X_gb_te)):.3f}")

# ── Save all pkl files ───────────────────────────────────────────────────────
print("\n💾 Saving models...")
os.makedirs("models", exist_ok=True)

artifacts = {
    "models/lr_sentiment.pkl":   lr,
    "models/tfidf.pkl":          tfidf,
    "models/rf_adoption.pkl":    rf,
    "models/le_field.pkl":       le_field,
    "models/le_funds.pkl":       le_funds,
    "models/scaler.pkl":         scaler,
    "models/kmeans.pkl":         kmeans,
    "models/gb_sentiment.pkl":   gb,
    "models/count_vectorizer.pkl": cv,
}

for path, obj in artifacts.items():
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size = os.path.getsize(path) / 1024
    print(f"   ✓ {path} ({size:.0f} KB)")

# Save metadata (field/fund class lists + stats)
import json
vc = df["vader_sentiment"].value_counts()
cluster_info = []
df["cluster"] = kmeans.predict(feat_scaled)
for i in range(3):
    c = df[df["cluster"] == i]
    cluster_info.append({
        "id": i + 1,
        "size": int(len(c)),
        "adoption_rate": round(float((c["user_type"] == "User").mean() * 100), 1),
        "avg_sentiment": round(float(c["compound_score"].mean()), 3),
    })

meta = {
    "fields": list(le_field.classes_),
    "funds":  list(le_funds.classes_),
    "stats": {
        "total": int(len(df)),
        "users": int(users_mask.sum()),
        "non_users": int((~users_mask).sum()),
        "sentiment": {
            "positive": int(vc.get("Positive", 0)),
            "neutral":  int(vc.get("Neutral", 0)),
            "negative": int(vc.get("Negative", 0)),
        },
        "models": {
            "lr_accuracy": round(accuracy_score(y_te, lr.predict(X_te_vec)), 3),
            "rf_accuracy": round(accuracy_score(y_ad_te, rf.predict(X_ad_te)), 3),
            "gb_accuracy": round(accuracy_score(y_te, gb.predict(X_gb_te)), 3),
        },
        "clusters": cluster_info,
        "field_distribution": df["What is your field of study?"].value_counts().to_dict(),
    }
}

with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("   ✓ models/meta.json")

print("\n✅ All done! Commit the models/ folder to your repo.")
print("   git add models/")
print("   git commit -m 'feat: add pre-trained model pkl files'")
print("   git push")
