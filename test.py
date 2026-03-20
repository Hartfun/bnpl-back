
import pandas as pd, string, numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

def preprocess(text):
    if pd.isna(text) or str(text).strip() in ('','nan'): return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('','', string.punctuation + string.digits))
    return ' '.join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

def vader_label(s):
    if s >= 0.05: return 'Positive'
    if s <= -0.05: return 'Negative'
    return 'Neutral'

df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()
df = df.drop(columns=['Timestamp'], errors='ignore').drop_duplicates().reset_index(drop=True)

text_cols = [
    "What is your overall opinion of 'Buy Now, Pay Later' (BNPL) services like Slice or LazyPay?",
    'Describe a positive experience you have had using an EMI or BNPL service. What made it good?',
    'Describe a negative experience you have had, if any. What went wrong?',
    'What is your single biggest fear or concern about using these services regularly?',
]
for col in text_cols:
    if col in df.columns:
        df[col+'_clean'] = df[col].astype(str).apply(preprocess)
df['combined_text'] = df[[c+'_clean' for c in text_cols if c in df.columns]].agg(' '.join, axis=1)
df = df[df['combined_text'].str.len() > 0].reset_index(drop=True)
df['compound'] = df['combined_text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df['sentiment'] = df['compound'].apply(vader_label)
C_USED = "Have you ever used a 'Buy Now, Pay Later' (BNPL) or an EMI service for a purchase?"
df['user_type'] = np.where(df[C_USED].str.strip().str.lower()=='yes','User','Non-User')

print('=== Overall ===')
print(df['sentiment'].value_counts())
print()
print('=== Users only ===')
print(df[df['user_type']=='User']['sentiment'].value_counts())
print()
print('=== Non-users only ===')
print(df[df['user_type']=='Non-User']['sentiment'].value_counts())
print()
print('=== Avg compound score ===')
print('Users:', df[df['user_type']=='User']['compound'].mean().round(3))
print('Non-users:', df[df['user_type']=='Non-User']['compound'].mean().round(3))
