import os

import streamlit as st
from huggingface_hub import InferenceClient

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


def safe_secret_get(key: str, default: str = "") -> str:
    """Read a Streamlit secret without crashing when no secrets file exists."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def get_spotify_client():
    if spotipy is None or SpotifyClientCredentials is None:
        return None

    client_id = safe_secret_get("SPOTIPY_CLIENT_ID", os.getenv("SPOTIPY_CLIENT_ID", ""))
    client_secret = safe_secret_get("SPOTIPY_CLIENT_SECRET", os.getenv("SPOTIPY_CLIENT_SECRET", ""))

    if not client_id or not client_secret:
        return None

    try:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        return spotipy.Spotify(auth_manager=auth)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_lyrics_generation_pipeline():
    if pipeline is None:
        return None
    try:
        return pipeline("text-generation", model="gpt2")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_hf_client():
    token = safe_secret_get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    if not token:
        return None
    try:
        return InferenceClient(token=token)
    except Exception:
        return None
