"""
watson_helper.py
-----------------
Thin wrapper around IBM Watson Natural Language Understanding (NLU)
sentiment analysis, used to compare against our own trained ML model.

SETUP (do this once):
1. Go to https://cloud.ibm.com/registration and create a free "Lite" account.
2. In the IBM Cloud catalog, search for "Natural Language Understanding"
   and create a Lite (free) instance.
3. Go to the service's "Manage" tab -> copy the API Key and URL.
4. Set them as environment variables before running the app, e.g. (Linux/Mac):
       export WATSON_APIKEY="your_api_key_here"
       export WATSON_URL="your_service_url_here"
   On Windows (PowerShell):
       $env:WATSON_APIKEY="your_api_key_here"
       $env:WATSON_URL="your_service_url_here"

   Or, if deploying to Streamlit Community Cloud, add them under
   Settings -> Secrets as:
       WATSON_APIKEY = "your_api_key_here"
       WATSON_URL = "your_service_url_here"
"""

import os
import requests


def get_watson_sentiment(text, api_key=None, url=None):
    """
    Calls IBM Watson NLU sentiment endpoint.
    Returns a dict: {"label": "positive"/"negative"/"neutral", "score": float}
    or {"error": "..."} if it fails / isn't configured.
    """
    api_key = api_key or os.environ.get("WATSON_APIKEY")
    url = url or os.environ.get("WATSON_URL")

    if not api_key or not url:
        return {"error": "Watson API key/URL not configured."}

    endpoint = f"{url}/v1/analyze?version=2022-04-07"
    payload = {
        "text": text,
        "features": {"sentiment": {}},
        "language": "en",
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            auth=("apikey", api_key),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        doc_sentiment = data.get("sentiment", {}).get("document", {})
        return {
            "label": doc_sentiment.get("label", "unknown"),
            "score": doc_sentiment.get("score", 0.0),
        }
    except Exception as e:
        return {"error": str(e)}
