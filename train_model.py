"""
train_model.py
---------------
Loads the Sentiment140 dataset, cleans the tweets, runs basic statistics,
trains a TF-IDF + Logistic Regression sentiment classifier, and saves:
    - models/sentiment_model.pkl   (trained classifier)
    - models/vectorizer.pkl        (fitted TF-IDF vectorizer)
    - data/clean_sample.csv        (cleaned sample used by the dashboard)

HOW TO GET THE DATA (do this once, on your own machine):
1. Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
2. Unzip it, rename the CSV to: sentiment140.csv
3. Place it inside the data/ folder of this project.

Run:
    python train_model.py
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
import joblib
import os

DATA_PATH = "data/sentiment140.csv"
SAMPLE_SIZE = 50000  # keep training fast; raise this later if you have time
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path):
    """Load the raw Sentiment140 CSV (no header, latin-1 encoding)."""
    cols = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(path, encoding="latin-1", header=None, names=cols)
    # Original labels: 0 = negative, 4 = positive
    df["sentiment"] = df["target"].map({0: "negative", 4: "positive"})
    df = df.dropna(subset=["sentiment"])
    return df[["text", "sentiment"]]


def clean_text(text):
    """Basic NLP preprocessing for tweets."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                          # remove mentions
    text = re.sub(r"#", "", text)                              # keep hashtag word, drop symbol
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)                            # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_hashtags(text):
    """Pull #hashtags out of the RAW (uncleaned) text, before # is stripped."""
    return re.findall(r"#(\w+)", str(text).lower())


def simulate_engagement_score(row):
    """
    NOTE: Sentiment140 contains only tweet text + sentiment label — it has no real
    likes/comments/shares. To demonstrate an 'Engagement Prediction' module, we
    generate a SIMULATED engagement score from plausible text-based features
    (this is a formula, not real social data — disclose this clearly in your report).
    """
    base = 10
    base += row["word_count"] * 1.5
    base += row["num_hashtags"] * 8
    base += row["num_exclaim"] * 5
    base += 15 if row["sentiment"] == "positive" else 0
    noise = np.random.normal(0, 5)
    return max(0, base + noise)


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Total rows available: {len(df)}")

    # Balanced sample for speed (equal positive/negative)
    n_pos_avail = (df.sentiment == "positive").sum()
    n_neg_avail = (df.sentiment == "negative").sum()
    n_each = min(SAMPLE_SIZE // 2, n_pos_avail, n_neg_avail)
    df_pos = df[df.sentiment == "positive"].sample(n_each, random_state=42)
    df_neg = df[df.sentiment == "negative"].sample(n_each, random_state=42)
    df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Module 3: Trend/Hashtag Detection (from RAW text, before cleaning) ---
    df["hashtags"] = df["text"].apply(extract_hashtags)
    df["num_hashtags"] = df["hashtags"].apply(len)
    df["num_exclaim"] = df["text"].apply(lambda t: str(t).count("!"))

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # --- Basic statistics (covers your Statistics subject) ---
    print("\n--- Dataset Statistics ---")
    print(df["sentiment"].value_counts())
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    print(f"Average words per tweet: {df['word_count'].mean():.2f}")
    print(f"Std dev of word count: {df['word_count'].std():.2f}")

    # Top hashtags across the whole sample -> for the "Trend Detection" dashboard page
    all_hashtags = [h for tags in df["hashtags"] for h in tags]
    top_hashtags = pd.Series(all_hashtags).value_counts().head(20)
    top_hashtags.to_csv("data/top_hashtags.csv", header=["count"])
    print(f"\nTop hashtags found: {len(top_hashtags)} unique (see data/top_hashtags.csv)")

    # --- ML: TF-IDF + Logistic Regression (Sentiment module) ---
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    print("\nVectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model (Sentiment)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    print(f"Saved sentiment model + vectorizer to {MODEL_DIR}/")

    # --- Module 1: Audience/Content Segmentation (K-Means on TF-IDF vectors) ---
    print("\nRunning K-Means clustering (Audience Segmentation)...")
    full_vec = vectorizer.transform(df["clean_text"])
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(full_vec)

    terms = vectorizer.get_feature_names_out()
    cluster_summary = []
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_idx = center.argsort()[-8:][::-1]
        top_terms = ", ".join(terms[idx] for idx in top_idx)
        cluster_summary.append({
            "cluster": i,
            "size": int((df["cluster"] == i).sum()),
            "top_terms": top_terms,
        })
        print(f"  Cluster {i} ({(df['cluster'] == i).sum()} posts): {top_terms}")

    pd.DataFrame(cluster_summary).to_csv("data/cluster_summary.csv", index=False)
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    print(f"Saved clustering model to {MODEL_DIR}/kmeans_model.pkl")

    # --- Module 2: Engagement Prediction (Regression on simulated engagement score) ---
    print("\nTraining Engagement Prediction model (simulated target — see docstring)...")
    df["engagement_score"] = df.apply(simulate_engagement_score, axis=1)

    feature_cols = ["word_count", "num_hashtags", "num_exclaim"]
    df["sentiment_positive"] = (df["sentiment"] == "positive").astype(int)
    feature_cols.append("sentiment_positive")

    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        df[feature_cols], df["engagement_score"], test_size=0.2, random_state=42
    )
    reg_model = LinearRegression()
    reg_model.fit(Xe_train, ye_train)
    ye_pred = reg_model.predict(Xe_test)
    mae = mean_absolute_error(ye_test, ye_pred)
    r2 = r2_score(ye_test, ye_pred)
    print(f"Engagement model -> MAE: {mae:.2f}, R2: {r2:.2f}")

    joblib.dump(reg_model, os.path.join(MODEL_DIR, "engagement_model.pkl"))
    print(f"Saved engagement model to {MODEL_DIR}/engagement_model.pkl")

    # Save a cleaned sample (now including cluster + engagement columns) for the dashboard
    df.sample(min(5000, len(df)), random_state=1).to_csv(
        "data/clean_sample.csv", index=False
    )


if __name__ == "__main__":
    main()
