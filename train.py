"""
Run this ONCE locally to generate all .pkl files.
Then commit the models/ folder — the API loads them instead of retraining.

Usage:
    cd bnpl-back
    pip install -r requirements.txt
    python train.py
"""

import os, string, warnings, json, pickle
import numpy as np
import pandas as pd
import nltk

warnings.filterwarnings("ignore")
nltk.download("stopwords",     quiet=True)
nltk.download("wordnet",       quiet=True)
nltk.download("vader_lexicon", quiet=True)

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

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sid        = SentimentIntensityAnalyzer()

# ── fix 6: always find data.csv relative to this script ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_text(text):
    if pd.isna(text) or str(text).strip() in ("", "nan"): return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

def vader_label(score):
    if score >= 0.05:  return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"

# ── load ──────────────────────────────────────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv(os.path.join(BASE_DIR, "data.csv"))
df.columns = df.columns.str.strip()
df = df.drop(columns=["Timestamp"], errors="ignore").drop_duplicates().reset_index(drop=True)
print(f"   ✓ {len(df)} rows loaded")

# ── text preprocessing ────────────────────────────────────────────────────────
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
print(f"   ✓ {len(df)} rows after text filter")

# ── VADER sentiment ───────────────────────────────────────────────────────────
df["compound_score"]  = df["combined_text"].apply(lambda x: sid.polarity_scores(str(x))["compound"])
df["vader_sentiment"] = df["compound_score"].apply(vader_label)

users_col  = "Have you ever used a 'Buy Now, Pay Later' (BNPL) or an EMI service for a purchase?"
df["user_type"] = np.where(df[users_col].str.strip().str.lower() == "yes", "User", "Non-User")

# ── Model 1: Logistic Regression (sentiment) ──────────────────────────────────
print("🤖 Training Logistic Regression...")
X = df["combined_text"].values
y = df["vader_sentiment"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf    = TfidfVectorizer(max_features=100, min_df=5, max_df=0.8)
X_tr_vec = tfidf.fit_transform(X_tr)
X_te_vec = tfidf.transform(X_te)
lr       = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_tr_vec, y_tr)
lr_acc   = accuracy_score(y_te, lr.predict(X_te_vec))
print(f"   ✓ LR accuracy: {lr_acc:.3f}")

# ── Model 2: Random Forest (adoption) ─────────────────────────────────────────
print("🌲 Training Random Forest...")
le_field = LabelEncoder()
le_funds = LabelEncoder()
le_year  = LabelEncoder()
df["field_encoded"] = le_field.fit_transform(df["What is your field of study?"])
df["funds_encoded"] = le_funds.fit_transform(df["What is your primary source of funds for personal expenses?"])
df["year_encoded"]  = le_year.fit_transform(df["What is your current year of study?"].fillna("Other"))

# Sentiment polarity flags — strong signal for user vs non-user
df["is_positive"]   = (df["compound_score"] >= 0.05).astype(int)
df["is_negative"]   = (df["compound_score"] <= -0.05).astype(int)
df["abs_sentiment"] = df["compound_score"].abs()

feat     = ["compound_score", "is_positive", "is_negative", "abs_sentiment",
            "field_encoded", "funds_encoded", "year_encoded"]
X_ad     = df[feat].fillna(0)
y_ad     = (df["user_type"] == "User").astype(int)
X_ad_tr, X_ad_te, y_ad_tr, y_ad_te = train_test_split(X_ad, y_ad, test_size=0.2, random_state=42, stratify=y_ad)
rf       = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, class_weight="balanced")
rf.fit(X_ad_tr, y_ad_tr)
rf_acc   = accuracy_score(y_ad_te, rf.predict(X_ad_te))
print(f"   ✓ RF accuracy: {rf_acc:.3f}")
print(f"   Features: {feat}")

# ── Model 3: KMeans (segmentation) ────────────────────────────────────────────
print("📍 Training KMeans...")
scaler      = StandardScaler()
feat_scaled = scaler.fit_transform(X_ad)
kmeans      = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(feat_scaled)
df["cluster"] = kmeans.predict(feat_scaled)
print(f"   ✓ KMeans inertia: {kmeans.inertia_:.2f}")

# ── Model 4: Gradient Boosting (sentiment) ────────────────────────────────────
print("🚀 Training Gradient Boosting...")
cv       = CountVectorizer(max_features=100, min_df=5, max_df=0.8)
X_gb_tr  = cv.fit_transform(X_tr).toarray()
X_gb_te  = cv.transform(X_te).toarray()
gb       = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb.fit(X_gb_tr, y_tr)
gb_acc   = accuracy_score(y_te, gb.predict(X_gb_te))
print(f"   ✓ GB accuracy: {gb_acc:.3f}")

# ── save pkl files ─────────────────────────────────────────────────────────────
print("\n💾 Saving models...")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

artifacts = {
    "models/lr_sentiment.pkl":     lr,
    "models/tfidf.pkl":            tfidf,
    "models/rf_adoption.pkl":      rf,
    "models/le_field.pkl":         le_field,
    "models/le_funds.pkl":         le_funds,
    "models/le_year.pkl":          le_year,
    "models/scaler.pkl":           scaler,
    "models/kmeans.pkl":           kmeans,
    "models/gb_sentiment.pkl":     gb,
    "models/count_vectorizer.pkl": cv,
}
for path, obj in artifacts.items():
    full = os.path.join(BASE_DIR, path)
    with open(full, "wb") as f:
        pickle.dump(obj, f)
    print(f"   ✓ {path} ({os.path.getsize(full)/1024:.0f} KB)")

# ── save meta.json ─────────────────────────────────────────────────────────────
# fix 2: use filtered df for user counts  |  fix 5: cast int64 → int
vc = df["vader_sentiment"].value_counts()
cluster_info = []
for i in range(3):
    c = df[df["cluster"] == i]
    cluster_info.append({
        "id": i + 1,
        "size":           int(len(c)),
        "adoption_rate":  round(float((c["user_type"] == "User").mean() * 100), 1),
        "avg_sentiment":  round(float(c["compound_score"].mean()), 3),
    })

meta = {
    "fields": list(le_field.classes_),
    "funds":  list(le_funds.classes_),
    "stats": {
        "total":     int(len(df)),
        "users":     int((df["user_type"] == "User").sum()),      # fix 2
        "non_users": int((df["user_type"] == "Non-User").sum()),  # fix 2
        "sentiment": {
            "positive": int(vc.get("Positive", 0)),
            "neutral":  int(vc.get("Neutral",  0)),
            "negative": int(vc.get("Negative", 0)),
        },
        "models": {
            "lr_accuracy": round(lr_acc, 3),
            "rf_accuracy": round(rf_acc, 3),
            "gb_accuracy": round(gb_acc, 3),
        },
        "clusters": cluster_info,
        "field_distribution": {                                   # fix 5
            k: int(v) for k, v in
            df["What is your field of study?"].value_counts().items()
        },
    }
}

meta_path = os.path.join(BASE_DIR, "models", "meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("   ✓ models/meta.json")

print(f"\n✅ All done! {len(df)} rows | LR={lr_acc:.3f} RF={rf_acc:.3f} GB={gb_acc:.3f}")
print("   git add models/")
print("   git commit -m 'feat: retrained models'")
print("   git push")
