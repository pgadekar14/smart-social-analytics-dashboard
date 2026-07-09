"""
app.py
------
Streamlit dashboard: Social Media Sentiment Analysis & Business Insight Tool.

Run locally:
    streamlit run app.py

Deploy:
    Push this project to a GitHub repo, then deploy free on
    https://share.streamlit.io (Streamlit Community Cloud).
"""

import os
import re
import string
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from watson_helper import get_watson_sentiment
import db_helper

db_helper.init_db()

st.set_page_config(page_title="Social Media Sentiment Analytics", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VEC_PATH = BASE_DIR / "models" / "vectorizer.pkl"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "clean_sample.csv"
KMEANS_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
ENGAGEMENT_MODEL_PATH = BASE_DIR / "models" / "engagement_model.pkl"
CLUSTER_SUMMARY_PATH = BASE_DIR / "data" / "cluster_summary.csv"
TOP_HASHTAGS_PATH = BASE_DIR / "data" / "top_hashtags.csv"
TEXT_COLUMNS = ("text", "tweet_text", "full_text", "content", "body")
XQUIK_METADATA_COLUMNS = ("created_at", "query", "username", "author_username", "url")


# ---------- Helpers ----------
@st.cache_resource
def load_model():
    if not (MODEL_PATH.exists() and VEC_PATH.exists()):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return pred, proba


@st.cache_data
def load_sample_data():
    if SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH)
    return None


@st.cache_resource
def load_kmeans():
    return joblib.load(KMEANS_PATH) if KMEANS_PATH.exists() else None


@st.cache_resource
def load_engagement_model():
    return joblib.load(ENGAGEMENT_MODEL_PATH) if ENGAGEMENT_MODEL_PATH.exists() else None


@st.cache_data
def load_cluster_summary():
    return pd.read_csv(CLUSTER_SUMMARY_PATH) if CLUSTER_SUMMARY_PATH.exists() else None


@st.cache_data
def load_top_hashtags():
    return pd.read_csv(TOP_HASHTAGS_PATH) if TOP_HASHTAGS_PATH.exists() else None


def normalize_batch_input(data):
    text_column = next((column for column in TEXT_COLUMNS if column in data.columns), None)
    if text_column is None:
        return None, "CSV must include one text-like column: " + ", ".join(TEXT_COLUMNS)

    normalized = pd.DataFrame({"text": data[text_column].astype(str).str.strip()})
    for column in XQUIK_METADATA_COLUMNS:
        if column in data.columns:
            normalized[column] = data[column]

    normalized = normalized[normalized["text"] != ""]
    if normalized.empty:
        return None, "CSV text rows are empty after trimming whitespace."
    return normalized, None


# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview & Business Insights",
        "Live Prediction",
        "Batch Analysis (Upload CSV)",
        "Watson Comparison",
        "Audience Segmentation",
        "Trend Detection",
        "Engagement Prediction",
        "Prediction History",
    ],
)

model, vectorizer = load_model()
if model is None:
    st.sidebar.error("Model not found. Run `python train_model.py` first.")

# ---------- Page 1: Overview ----------
if page == "Overview & Business Insights":
    st.title("📊 Social Media Sentiment Analysis Dashboard")
    st.markdown(
        """
        This dashboard analyzes public social media / review text to classify sentiment
        as **positive** or **negative** using a Machine Learning model (TF-IDF + Logistic
        Regression), and presents statistical and business insights that a company could
        use to track brand perception, product feedback, or campaign reception.
        """
    )

    df = load_sample_data()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records Analyzed", len(df))
        col2.metric("Positive %", f"{(df.sentiment == 'positive').mean() * 100:.1f}%")
        col3.metric("Negative %", f"{(df.sentiment == 'negative').mean() * 100:.1f}%")

        st.subheader("Sentiment Distribution")
        fig = px.pie(df, names="sentiment", title="Overall Sentiment Split", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Word Count Statistics (per post)")
        fig2 = px.histogram(df, x="word_count", color="sentiment", barmode="overlay",
                             nbins=30, title="Distribution of Post Length by Sentiment")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Most Frequent Words")
        sentiment_choice = st.selectbox("Choose sentiment", ["positive", "negative"])
        text_blob = " ".join(df[df.sentiment == sentiment_choice]["clean_text"].astype(str))
        word_counts = Counter(text_blob.split())
        common = pd.DataFrame(word_counts.most_common(15), columns=["word", "count"])
        fig3 = px.bar(common, x="word", y="count", title=f"Top words in {sentiment_choice} posts")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run `python train_model.py` first to generate the sample dataset for charts.")

# ---------- Page 2: Live Prediction ----------
elif page == "Live Prediction":
    st.title("🔮 Try It: Predict Sentiment of Your Own Text")
    user_text = st.text_area("Enter a tweet / review / comment:", height=120,
                              placeholder="e.g. This product completely exceeded my expectations!")
    if st.button("Analyze Sentiment"):
        if model is None:
            st.error("Model not loaded. Run train_model.py first.")
        elif not user_text.strip():
            st.warning("Please enter some text.")
        else:
            pred, proba = predict_sentiment(user_text, model, vectorizer)
            emoji = "😊" if pred == "positive" else "😞"
            st.success(f"Predicted Sentiment: **{pred.upper()}** {emoji} (confidence: {proba*100:.1f}%)")
            db_helper.log_prediction(user_text, pred, proba, source="Live Prediction")

# ---------- Page 3: Batch Analysis ----------
elif page == "Batch Analysis (Upload CSV)":
    st.title("📁 Batch Sentiment Analysis")
    st.markdown(
        "Upload a CSV with social posts or reviews to analyze in bulk. Xquik tweet/search "
        "exports are supported when they include a text-like column."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        normalized_data, error = normalize_batch_input(data)
        if model is None:
            st.error("Model not loaded. Run `python train_model.py` first.")
        elif error:
            st.error(error)
        else:
            with st.spinner("Analyzing..."):
                preds, confs = [], []
                for t in normalized_data["text"].astype(str):
                    p, c = predict_sentiment(t, model, vectorizer)
                    preds.append(p)
                    confs.append(c)
                normalized_data["predicted_sentiment"] = preds
                normalized_data["confidence"] = confs
            st.dataframe(
                normalized_data.head(20),
                column_config={
                    "confidence": st.column_config.NumberColumn(
                        "confidence", format="percent"
                    )
                },
            )
            fig = px.pie(normalized_data, names="predicted_sentiment", title="Batch Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Results as CSV",
                                normalized_data.to_csv(index=False).encode("utf-8"),
                                "sentiment_results.csv", "text/csv")

# ---------- Page 4: Watson Comparison ----------
elif page == "Watson Comparison":
    st.title("🧠 Compare with IBM Watson NLU")
    st.markdown(
        "This page sends your text to **your own ML model** and to **IBM Watson Natural "
        "Language Understanding**, and shows both results side by side. "
        "Requires `WATSON_APIKEY` and `WATSON_URL` to be set (see watson_helper.py)."
    )
    user_text = st.text_area("Enter text to compare:", height=120)
    if st.button("Compare"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Your ML Model")
                if model is not None:
                    pred, proba = predict_sentiment(user_text, model, vectorizer)
                    st.info(f"{pred.upper()} ({proba*100:.1f}% confidence)")
                else:
                    st.error("Model not loaded.")
            with col2:
                st.subheader("IBM Watson NLU")
                result = get_watson_sentiment(user_text)
                if "error" in result:
                    st.warning(f"Watson not available: {result['error']}")
                else:
                    st.info(f"{result['label'].upper()} (score: {result['score']:.2f})")

# ---------- Page 5: Audience Segmentation (K-Means) ----------
elif page == "Audience Segmentation":
    st.title("👥 Audience / Content Segmentation")
    st.markdown(
        "Posts are grouped into clusters of similar content using **K-Means clustering** "
        "on the TF-IDF text vectors — an unsupervised ML technique. This helps a business "
        "identify distinct themes or audience segments in their social media mentions "
        "without needing pre-labeled categories."
    )
    summary = load_cluster_summary()
    df = load_sample_data()
    if summary is not None:
        st.subheader("Cluster Summary")
        st.dataframe(summary)

        fig = px.bar(summary, x="cluster", y="size", title="Posts per Cluster",
                      labels={"cluster": "Cluster ID", "size": "Number of Posts"})
        st.plotly_chart(fig, use_container_width=True)

        for _, row in summary.iterrows():
            st.markdown(f"**Cluster {row['cluster']}** ({row['size']} posts) — top terms: `{row['top_terms']}`")

        if df is not None and "cluster" in df.columns:
            st.subheader("Sentiment Mix Within Each Cluster")
            fig2 = px.histogram(df, x="cluster", color="sentiment", barmode="group",
                                 title="Sentiment Distribution by Cluster")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run `python train_model.py` first to generate clustering results.")

# ---------- Page 6: Trend Detection (Hashtags) ----------
elif page == "Trend Detection":
    st.title("📈 Trend Detection — Top Hashtags & Keywords")
    st.markdown(
        "Extracts and ranks hashtags found in the raw posts, giving a quick view of "
        "what topics/campaigns are trending in the dataset."
    )
    hashtags = load_top_hashtags()
    if hashtags is not None and len(hashtags) > 0:
        hashtags.columns = ["hashtag", "count"]
        fig = px.bar(hashtags, x="hashtag", y="count", title="Top Trending Hashtags")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hashtags)
    else:
        st.info(
            "No hashtags found in this dataset sample (common with short/synthetic demo data). "
            "Once you run this on the real Sentiment140 data, hashtag-bearing tweets will show up here."
        )

# ---------- Page 7: Engagement Prediction ----------
elif page == "Engagement Prediction":
    st.title("🚀 Engagement Prediction")
    st.warning(
        "**Note on data:** Sentiment140 doesn't include real likes/comments/shares. "
        "This module predicts a *simulated* engagement score derived from post features "
        "(length, hashtags, exclamation marks, sentiment) — built to demonstrate a regression "
        "ML module, not to reflect real social media engagement."
    )
    eng_model = load_engagement_model()
    user_text = st.text_area("Enter a post to estimate engagement for:", height=120)
    if st.button("Predict Engagement"):
        if eng_model is None:
            st.error("Engagement model not found. Run train_model.py first.")
        elif not user_text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_text)
            word_count = len(cleaned.split())
            num_hashtags = len(re.findall(r"#(\w+)", user_text.lower()))
            num_exclaim = user_text.count("!")
            sentiment_positive = 0
            if model is not None:
                pred, _ = predict_sentiment(user_text, model, vectorizer)
                sentiment_positive = 1 if pred == "positive" else 0
            features = pd.DataFrame([{
                "word_count": word_count,
                "num_hashtags": num_hashtags,
                "num_exclaim": num_exclaim,
                "sentiment_positive": sentiment_positive,
            }])
            score = eng_model.predict(features)[0]
            st.success(f"Predicted (simulated) Engagement Score: **{score:.1f}**")
            st.caption("Higher score = more words, more hashtags, more excitement, positive tone.")

# ---------- Page 8: Prediction History (SQLite backend) ----------
elif page == "Prediction History":
    st.title("🗄️ Prediction History (SQLite Database)")
    st.markdown(
        "Every prediction made on the **Live Prediction** page is logged to a local "
        "SQLite database (`data/predictions.db`), demonstrating a real RDBMS backend "
        "with Create, Read, and Delete operations."
    )

    total = db_helper.count_records()
    st.metric("Total Logged Predictions", total)

    rows = db_helper.get_history(limit=200)
    if rows:
        hist_df = pd.DataFrame(
            rows, columns=["id", "text", "sentiment", "confidence", "source", "created_at"]
        )
        st.dataframe(hist_df, use_container_width=True)

        fig = px.pie(hist_df, names="sentiment", title="Sentiment Split of Logged Predictions")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.warning("This will permanently delete all logged prediction history.")
        if st.button("🗑️ Clear History"):
            db_helper.clear_history()
            st.success("History cleared.")
            st.rerun()
    else:
        st.info("No predictions logged yet. Go to the 'Live Prediction' page and analyze some text first.")
