"""
AI Music Hub - Streamlit full-stack music intelligence app.
Deploy:
1) pip install -r requirements.txt
2) streamlit run app.py
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from huggingface_hub import InferenceClient
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    import librosa
except Exception:
    librosa = None

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

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Music Hub",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling (Spotify-like dark UI)
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --spotify-black: #000000;
            --spotify-green: #1DB954;
            --spotify-dark: #101010;
            --spotify-mid: #181818;
            --spotify-light: #b3b3b3;
            --spotify-white: #ffffff;
        }

        html, body, [class*="css"] {
            background: radial-gradient(circle at top left, #1a1a1a 0%, #000 40%, #000 100%);
            color: var(--spotify-white);
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.4px;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            color: var(--spotify-light);
            margin-bottom: 1rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #121212 0%, #0a0a0a 100%);
            border: 1px solid #222;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .feature-card {
            background: #121212;
            border: 1px solid #212121;
            border-radius: 14px;
            padding: 0.8rem;
            transition: transform 0.25s ease;
            height: 100%;
        }

        .feature-card:hover {
            transform: translateY(-2px);
            border-color: #2d2d2d;
        }

        .track-scroll {
            display: flex;
            overflow-x: auto;
            gap: 12px;
            padding-bottom: 8px;
            margin-bottom: 0.4rem;
        }

        .track-pill {
            min-width: 220px;
            background: #141414;
            border: 1px solid #262626;
            border-radius: 12px;
            padding: 10px;
        }

        .spotify-badge {
            background: rgba(29, 185, 84, 0.15);
            border: 1px solid rgba(29, 185, 84, 0.5);
            color: #79ffad;
            border-radius: 999px;
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }

        .stButton > button {
            background: #1DB954;
            color: #000;
            border: none;
            border-radius: 999px;
            font-weight: 700;
        }

        .stButton > button:hover {
            background: #22d462;
            color: #000;
        }

        .sidebar-header {
            color: #1DB954;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .main-title {
                font-size: 1.7rem;
            }
            .track-pill {
                min-width: 180px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Data Structures and Constants
# -----------------------------
@dataclass
class SpotifyTrack:
    id: str
    name: str
    artist: str
    album: str
    popularity: int
    preview_url: Optional[str]
    external_url: Optional[str]


FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
]

NAV_ITEMS = {
    "Album Analyzer": "🎵 Album Analyzer",
    "Lyrics Generator": "✍️ Lyrics Generator",
    "Similarity Finder": "🔎 Similarity Finder",
    "Genre Predictor": "🎚️ Genre Predictor",
    "Playlist Curator": "📻 Playlist Curator",
}


# -----------------------------
# Caching and Clients
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_spotify_client():
    if spotipy is None or SpotifyClientCredentials is None:
        return None

    client_id = st.secrets.get("SPOTIPY_CLIENT_ID", os.getenv("SPOTIPY_CLIENT_ID", ""))
    client_secret = st.secrets.get("SPOTIPY_CLIENT_SECRET", os.getenv("SPOTIPY_CLIENT_SECRET", ""))

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
    token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    if not token:
        return None
    try:
        return InferenceClient(token=token)
    except Exception:
        return None


# -----------------------------
# Utility Helpers
# -----------------------------
def info_banner(sp_available: bool) -> None:
    badge = "Connected" if sp_available else "Demo mode"
    st.markdown(f"<span class='spotify-badge'>Spotify API: {badge}</span>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def search_spotify_albums(query: str, limit: int = 9) -> List[dict]:
    sp = get_spotify_client()
    if sp is None:
        return []
    try:
        data = sp.search(q=query, type="album", limit=limit)
        return data.get("albums", {}).get("items", [])
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_album_tracks_and_features(album_id: str) -> pd.DataFrame:
    sp = get_spotify_client()
    if sp is None:
        return pd.DataFrame()

    try:
        album_tracks = sp.album_tracks(album_id).get("items", [])
        if not album_tracks:
            return pd.DataFrame()

        track_ids = [t["id"] for t in album_tracks if t.get("id")]
        if not track_ids:
            return pd.DataFrame()

        features = sp.audio_features(track_ids)
        rows = []

        for i, t in enumerate(album_tracks):
            feat = features[i] if i < len(features) else None
            row = {
                "track_id": t.get("id"),
                "track_name": t.get("name"),
                "duration_ms": t.get("duration_ms", 0),
                "track_number": t.get("track_number", i + 1),
            }
            for col in FEATURE_COLUMNS:
                row[col] = (feat or {}).get(col, np.nan)
            rows.append(row)

        return pd.DataFrame(rows).sort_values("track_number")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_lyrics(artist: str, title: str) -> str:
    # Free public lyrics endpoint used for educational/demo purposes.
    endpoint = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    try:
        response = requests.get(endpoint, timeout=8)
        if response.status_code == 200:
            data = response.json()
            return data.get("lyrics", "")[:4000]
    except Exception:
        pass
    return ""


def score_sentiment(text: str) -> Tuple[str, float]:
    if not text.strip():
        return "NEUTRAL", 0.5

    sentiment_fn = get_sentiment_pipeline()
    if sentiment_fn is not None:
        try:
            result = sentiment_fn(text[:500])[0]
            label = result.get("label", "NEUTRAL")
            score = float(result.get("score", 0.5))
            return label, score
        except Exception:
            pass

    # Lightweight fallback if model is unavailable.
    pos_words = ["love", "light", "happy", "dance", "dream", "alive", "sun"]
    neg_words = ["dark", "pain", "cry", "alone", "broken", "fear", "lost"]
    lower = text.lower()
    pos = sum(word in lower for word in pos_words)
    neg = sum(word in lower for word in neg_words)
    if pos >= neg:
        return "POSITIVE", min(1.0, 0.55 + 0.05 * (pos - neg))
    return "NEGATIVE", min(1.0, 0.55 + 0.05 * (neg - pos))


def render_album_grid(albums: List[dict]) -> None:
    cols = st.columns(3)
    for i, album in enumerate(albums):
        with cols[i % 3]:
            image = album.get("images", [{}])
            image_url = image[0].get("url") if image else None
            name = album.get("name", "Unknown")
            artists = ", ".join([a.get("name", "") for a in album.get("artists", [])])
            year = (album.get("release_date") or "")[:4]

            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            if image_url:
                st.image(image_url, use_container_width=True)
            st.markdown(f"**{name}**")
            st.caption(f"{artists} • {year}")
            st.markdown("</div>", unsafe_allow_html=True)


def render_track_scroll(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No tracks available to display.")
        return

    pills = []
    for _, row in df.head(20).iterrows():
        pills.append(
            f"<div class='track-pill'><strong>{row['track_name']}</strong><br/>"
            f"Energy: {row.get('energy', 0):.2f} | Dance: {row.get('danceability', 0):.2f}<br/>"
            f"Valence: {row.get('valence', 0):.2f}</div>"
        )
    html = "<div class='track-scroll'>" + "".join(pills) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def spotify_embed(url: Optional[str], width: str = "100%", height: int = 152) -> None:
    if not url:
        return
    embed_url = url.replace("open.spotify.com", "open.spotify.com/embed")
    st.components.v1.html(
        f'<iframe src="{embed_url}" width="{width}" height="{height}" '
        'frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
        height=height + 8,
    )


def extract_audio_features_from_upload(uploaded_file) -> Optional[np.ndarray]:
    if librosa is None:
        return None
    try:
        y, sr = librosa.load(uploaded_file, sr=22050, mono=True)
        if len(y) == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        chroma = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        mfcc = float(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)))

        # Map extracted values into a Spotify-like feature vector scale.
        mapped = np.array(
            [
                np.clip(chroma, 0, 1),
                np.clip(rms * 20, 0, 1),
                np.clip(1 - zcr * 5, 0, 1),
                np.clip(float(tempo) / 200, 0, 1),
                np.clip(spec_rolloff / 10000, 0, 1),
                np.clip((spec_centroid / 6000), 0, 1),
                np.clip((mfcc + 200) / 400, 0, 1),
            ]
        )
        return mapped
    except Exception:
        return None


def parse_keywords(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) > 2]
    return list(dict.fromkeys(tokens))[:8]


# -----------------------------
# Feature: Album Analyzer
# -----------------------------
def album_analyzer(global_query: str) -> None:
    st.subheader("Album Analyzer")
    st.caption("Search Spotify albums, inspect mood trends, and run lyric sentiment analysis.")

    q = st.text_input("Search artist or album", value=global_query, key="album_search")
    if not q:
        st.info("Type an artist or album to begin.")
        return

    with st.spinner("Searching albums..."):
        albums = search_spotify_albums(q, limit=9)

    if not albums:
        st.warning("No albums found. Check API credentials or try another query.")
        return

    render_album_grid(albums)

    album_choices = {
        f"{a.get('name', 'Unknown')} - {', '.join(ar['name'] for ar in a.get('artists', []))}": a
        for a in albums
    }
    selected_label = st.selectbox("Choose an album to analyze", list(album_choices.keys()))
    selected_album = album_choices[selected_label]

    with st.spinner("Fetching track features..."):
        df = get_album_tracks_and_features(selected_album["id"])

    if df.empty:
        st.warning("Audio feature data not available for this album.")
        return

    st.markdown("### Track Mood Strip")
    render_track_scroll(df)

    metric_df = df[["track_name", "danceability", "energy", "valence"]].melt(
        id_vars=["track_name"], var_name="feature", value_name="value"
    )
    fig = px.line(
        metric_df,
        x="track_name",
        y="value",
        color="feature",
        markers=True,
        title="Mood Trend Across Album Tracks",
        color_discrete_sequence=["#1DB954", "#00C2FF", "#F4D03F"],
    )
    fig.update_layout(template="plotly_dark", xaxis_title="Track", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Lyrics Sentiment")
    artist_name = selected_album.get("artists", [{}])[0].get("name", "")
    sample_tracks = df.head(5)["track_name"].tolist()

    sentiment_rows = []
    with st.spinner("Analyzing lyric sentiment on sample tracks..."):
        for track_name in sample_tracks:
            lyric_text = fetch_lyrics(artist_name, track_name)
            label, score = score_sentiment(lyric_text)
            sentiment_rows.append(
                {
                    "track": track_name,
                    "sentiment": label,
                    "confidence": round(score, 3),
                    "lyrics_found": bool(lyric_text.strip()),
                }
            )

    sentiment_df = pd.DataFrame(sentiment_rows)
    st.dataframe(sentiment_df, use_container_width=True)

    if not sentiment_df.empty:
        sent_fig = px.bar(
            sentiment_df,
            x="track",
            y="confidence",
            color="sentiment",
            title="Lyric Sentiment Confidence",
            color_discrete_map={"POSITIVE": "#1DB954", "NEGATIVE": "#E74C3C", "NEUTRAL": "#95A5A6"},
        )
        sent_fig.update_layout(template="plotly_dark")
        st.plotly_chart(sent_fig, use_container_width=True)


# -----------------------------
# Feature: Lyrics Generator
# -----------------------------
def generate_lyrics_from_prompt(prompt: str, max_new_tokens: int = 220) -> str:
    hf_client = get_hf_client()
    if hf_client is not None:
        for model_name in ["nevoit/Song-Lyrics-Generator", "gpt2"]:
            try:
                response = hf_client.text_generation(
                    prompt,
                    model=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=0.9,
                    top_p=0.95,
                )
                if isinstance(response, str) and response.strip():
                    return response
            except Exception:
                continue

    local_pipe = get_lyrics_generation_pipeline()
    if local_pipe is not None:
        try:
            out = local_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.92)
            text = out[0].get("generated_text", "")
            if text.strip():
                return text
        except Exception:
            pass

    # Deterministic fallback template if model APIs are unavailable.
    theme = prompt.split("Theme:")[-1].strip()[:80] if "Theme:" in prompt else "midnight stories"
    return (
        "[Verse 1]\n"
        f"Streetlights whisper in {theme}, we chase the echo tonight,\n"
        "Broken records spin our names in neon light.\n\n"
        "[Chorus]\n"
        "Hold on, hold on, to the sound of who we are,\n"
        "Every heartbeat turns a shadow into stars.\n\n"
        "[Verse 2]\n"
        "We write our future on the windows of the train,\n"
        "Sing it louder till the sky forgets the rain.\n"
    )


def lyrics_generator(global_query: str) -> None:
    st.subheader("Lyrics Generator")
    st.caption("Generate full song drafts by genre, theme, and artist style cues.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        genre = st.text_input("Genre", value="indie pop")
    with col2:
        theme = st.text_input("Theme", value=global_query or "late-night city drive")
    with col3:
        style_artist = st.text_input("Band/Artist style", value="Coldplay")

    style_hint = ""
    sp = get_spotify_client()
    if sp is not None and style_artist.strip():
        try:
            artist_result = sp.search(q=style_artist, type="artist", limit=1)
            items = artist_result.get("artists", {}).get("items", [])
            if items:
                artist_id = items[0]["id"]
                top_tracks = sp.artist_top_tracks(artist_id).get("tracks", [])[:5]
                style_hint = ", ".join([t.get("name", "") for t in top_tracks])
        except Exception:
            style_hint = ""

    st.markdown("Sample style hints from Spotify top tracks:")
    st.write(style_hint or "No style hints found. A generic style prompt will be used.")

    structure = st.selectbox("Song structure", ["Verse-Chorus-Verse-Chorus", "Verse-Pre-Chorus-Chorus", "Ballad"])
    length = st.slider("Output length (tokens)", 120, 320, 220, 20)

    if st.button("Generate Lyrics"):
        prompt = (
            f"Write complete song lyrics in the style of {style_artist}.\n"
            f"Genre: {genre}\n"
            f"Theme: {theme}\n"
            f"Structure: {structure}\n"
            f"Style cues from song titles: {style_hint or 'none'}\n"
            "Output with [Verse], [Chorus], and [Bridge] labels when appropriate."
        )

        with st.spinner("Generating lyrics with Hugging Face model..."):
            lyrics = generate_lyrics_from_prompt(prompt, max_new_tokens=length)

        st.text_area("Generated Lyrics", value=lyrics, height=380)

        label, score = score_sentiment(lyrics)
        st.caption(f"Generated lyrics sentiment: {label} ({score:.2f})")


# -----------------------------
# Feature: Similarity Finder
# -----------------------------
def fetch_track_candidates(query: str, limit: int = 25) -> pd.DataFrame:
    sp = get_spotify_client()
    if sp is None:
        return pd.DataFrame()

    try:
        result = sp.search(q=query, type="track", limit=limit)
        tracks = result.get("tracks", {}).get("items", [])
        if not tracks:
            return pd.DataFrame()

        ids = [t["id"] for t in tracks if t.get("id")]
        feats = sp.audio_features(ids)

        rows = []
        for idx, t in enumerate(tracks):
            f = feats[idx] if idx < len(feats) else None
            if not f:
                continue
            rows.append(
                {
                    "id": t.get("id"),
                    "name": t.get("name"),
                    "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                    "album": t.get("album", {}).get("name"),
                    "external_url": t.get("external_urls", {}).get("spotify"),
                    **{c: f.get(c, np.nan) for c in FEATURE_COLUMNS},
                }
            )

        return pd.DataFrame(rows).dropna()
    except Exception:
        return pd.DataFrame()


def nearest_neighbors(df: pd.DataFrame, query_vector: np.ndarray, top_k: int = 8) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].values.astype("float32")
    q = query_vector.astype("float32").reshape(1, -1)

    if faiss is not None:
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        distances, indices = index.search(q, min(top_k, len(df)))
        result = df.iloc[indices[0]].copy()
        result["distance"] = distances[0]
        return result

    sim = cosine_similarity(q, X)[0]
    out = df.copy()
    out["similarity"] = sim
    return out.sort_values("similarity", ascending=False).head(top_k)


def similarity_finder(global_query: str) -> None:
    st.subheader("Similarity Finder")
    st.caption("Find similar songs and bands using Spotify audio embeddings + cosine similarity/FAISS.")

    col1, col2 = st.columns([2, 1])
    with col1:
        q = st.text_input("Search tracks", value=global_query or "dream pop")
    with col2:
        top_k = st.slider("Recommendations", 5, 15, 8)

    uploaded_audio = st.file_uploader("Or upload audio (.wav/.mp3)", type=["wav", "mp3", "ogg"])

    with st.spinner("Building candidate pool..."):
        candidate_df = fetch_track_candidates(q, limit=35) if q else pd.DataFrame()

    if candidate_df.empty:
        st.warning("No candidate tracks found.")
        return

    query_vector = None
    query_track_name = "Uploaded audio"

    if uploaded_audio is not None:
        query_vector = extract_audio_features_from_upload(uploaded_audio)
        if query_vector is None:
            st.error("Audio analysis failed. Try another file format.")
            return
    else:
        picked = st.selectbox("Reference track", candidate_df["name"].tolist())
        row = candidate_df[candidate_df["name"] == picked].head(1)
        query_vector = row[FEATURE_COLUMNS].values[0]
        query_track_name = picked

    neighbors = nearest_neighbors(candidate_df, query_vector, top_k=top_k)
    st.markdown(f"### Fans also like: tracks similar to **{query_track_name}**")

    cols = st.columns(4)
    for i, (_, rec) in enumerate(neighbors.iterrows()):
        with cols[i % 4]:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.markdown(f"**{rec['name']}**")
            st.caption(f"{rec['artist']}")
            st.caption(f"Album: {rec['album']}")
            if rec.get("external_url"):
                spotify_embed(rec["external_url"], height=120)
            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Feature: Genre Predictor
# -----------------------------
@st.cache_data(show_spinner=False)
def build_genre_dataset() -> pd.DataFrame:
    sp = get_spotify_client()
    genres = ["rock", "pop", "hip hop", "jazz", "electronic", "classical"]
    rows = []

    if sp is None:
        # Synthetic fallback dataset with approximate feature centers per genre.
        centers = {
            "rock": [0.55, 0.8, 0.45, 0.65, 0.22, 0.05, 0.09],
            "pop": [0.72, 0.68, 0.63, 0.62, 0.18, 0.01, 0.08],
            "hip hop": [0.79, 0.7, 0.52, 0.58, 0.15, 0.0, 0.33],
            "jazz": [0.52, 0.46, 0.58, 0.48, 0.45, 0.2, 0.07],
            "electronic": [0.74, 0.83, 0.55, 0.7, 0.12, 0.42, 0.09],
            "classical": [0.28, 0.25, 0.44, 0.37, 0.76, 0.62, 0.03],
        }
        for g, center in centers.items():
            for _ in range(40):
                jitter = np.random.normal(0, 0.06, size=7)
                vals = np.clip(np.array(center) + jitter, 0, 1)
                rows.append({**{FEATURE_COLUMNS[i]: vals[i] for i in range(7)}, "genre": g})
        return pd.DataFrame(rows)

    for genre in genres:
        try:
            result = sp.search(q=genre, type="track", limit=25)
            tracks = result.get("tracks", {}).get("items", [])
            ids = [t.get("id") for t in tracks if t.get("id")]
            feats = sp.audio_features(ids)
            for f in feats:
                if not f:
                    continue
                row = {col: f.get(col, np.nan) for col in FEATURE_COLUMNS}
                row["genre"] = genre
                rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows).dropna()
    return df


def train_genre_model(df: pd.DataFrame):
    if df.empty or len(df["genre"].unique()) < 2:
        return None, None, None, None

    X = df[FEATURE_COLUMNS]
    y = df["genre"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))

    return model, acc, cm, sorted(y.unique())


def genre_predictor(global_query: str) -> None:
    st.subheader("Genre Predictor")
    st.caption("Train a lightweight classifier and predict genre from Spotify/audio features.")

    with st.spinner("Preparing dataset and training model..."):
        df = build_genre_dataset()
        model, acc, cm, labels = train_genre_model(df)

    if model is None:
        st.error("Insufficient data to train genre model.")
        return

    st.metric("Validation Accuracy", f"{acc*100:.1f}%")

    cm_fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale=[[0, "#0f0f0f"], [1, "#1DB954"]],
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Genre Classifier Confusion Matrix",
        text_auto=True,
    )
    cm_fig.update_layout(template="plotly_dark")
    st.plotly_chart(cm_fig, use_container_width=True)

    option = st.radio("Prediction input", ["Spotify track search", "Audio upload"], horizontal=True)

    sample_vec = None
    if option == "Spotify track search":
        query = st.text_input("Track name", value=global_query or "blinding lights", key="genre_track_search")
        tracks_df = fetch_track_candidates(query, limit=10) if query else pd.DataFrame()
        if tracks_df.empty:
            st.warning("No tracks found for prediction.")
            return

        pick = st.selectbox("Choose track", tracks_df["name"].tolist(), key="genre_track_pick")
        sample_vec = tracks_df[tracks_df["name"] == pick].iloc[0][FEATURE_COLUMNS].values
    else:
        uploaded = st.file_uploader("Upload audio file for genre prediction", type=["wav", "mp3", "ogg"], key="genre_upload")
        if uploaded is not None:
            sample_vec = extract_audio_features_from_upload(uploaded)

    if sample_vec is not None:
        pred = model.predict(np.array(sample_vec).reshape(1, -1))[0]
        probs = model.predict_proba(np.array(sample_vec).reshape(1, -1))[0]
        prob_df = pd.DataFrame({"genre": model.classes_, "probability": probs}).sort_values("probability", ascending=False)

        st.success(f"Predicted Genre: {pred}")

        pfig = px.bar(
            prob_df,
            x="genre",
            y="probability",
            color="probability",
            color_continuous_scale=[[0, "#222"], [1, "#1DB954"]],
            title="Genre Probability Distribution",
        )
        pfig.update_layout(template="plotly_dark")
        st.plotly_chart(pfig, use_container_width=True)


# -----------------------------
# Feature: Playlist Curator
# -----------------------------
def generate_playlist_queries(mood_prompt: str, target_energy: float) -> List[str]:
    hf_client = get_hf_client()
    prompt = (
        "Create a compact playlist concept and return 12 comma-separated search phrases. "
        f"Mood: {mood_prompt}. Energy score: {target_energy:.2f}."
    )

    if hf_client is not None:
        for model_name in ["microsoft/DialoGPT-small", "gpt2"]:
            try:
                text = hf_client.text_generation(
                    prompt,
                    model=model_name,
                    max_new_tokens=120,
                    temperature=0.9,
                )
                if isinstance(text, str) and text.strip():
                    candidates = [s.strip() for s in re.split(r",|\n", text) if len(s.strip()) > 2]
                    if candidates:
                        return candidates[:12]
            except Exception:
                continue

    keywords = parse_keywords(mood_prompt)
    seed_terms = keywords[:4] if keywords else ["nostalgic", "night", "vibes"]
    base = [
        f"{seed_terms[0]} indie",
        f"{seed_terms[0]} rock",
        f"{seed_terms[0]} pop",
        f"{seed_terms[-1]} electronic",
        f"{seed_terms[0]} classics",
        "feel good anthems",
        "late night drive",
        "sunset chill",
        "emotional guitar",
        "retro hits",
        "dreamy synth",
        "weekend energy",
    ]
    return base[:12]


def search_tracks_for_playlist(queries: List[str], cap: int = 15) -> pd.DataFrame:
    sp = get_spotify_client()
    if sp is None:
        rows = []
        for i, q in enumerate(queries[:cap], 1):
            rows.append(
                {
                    "order": i,
                    "track": f"{q.title()} Track",
                    "artist": "Demo Artist",
                    "reason": f"Matches mood phrase: {q}",
                    "url": "",
                }
            )
        return pd.DataFrame(rows)

    seen = set()
    rows = []
    for q in queries:
        try:
            result = sp.search(q=q, type="track", limit=3)
            items = result.get("tracks", {}).get("items", [])
            for t in items:
                key = t.get("id")
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "order": len(rows) + 1,
                        "track": t.get("name"),
                        "artist": ", ".join([a["name"] for a in t.get("artists", [])]),
                        "reason": f"Selected from query: {q}",
                        "url": t.get("external_urls", {}).get("spotify", ""),
                    }
                )
                if len(rows) >= cap:
                    return pd.DataFrame(rows)
        except Exception:
            continue

    return pd.DataFrame(rows)


def playlist_curator(global_query: str) -> None:
    st.subheader("Playlist Curator")
    st.caption("Create mood-driven playlists with explainable track picks and Spotify embeds.")

    mood = st.text_input("Mood prompt", value=global_query or "nostalgic 90s rock with hopeful lyrics")
    energy_slider = st.slider("Target energy", 0.0, 1.0, 0.6, 0.05)

    if st.button("Curate Playlist"):
        with st.spinner("Generating playlist concept..."):
            queries = generate_playlist_queries(mood, energy_slider)
            playlist_df = search_tracks_for_playlist(queries, cap=15)

        if playlist_df.empty:
            st.warning("Could not build playlist from the current mood prompt.")
            return

        st.markdown("### Playlist Draft")
        st.caption("Edit the order column to simulate drag/reorder behavior.")

        edited = st.data_editor(
            playlist_df,
            column_config={
                "order": st.column_config.NumberColumn("Order", min_value=1, step=1),
                "track": st.column_config.TextColumn("Track"),
                "artist": st.column_config.TextColumn("Artist"),
                "reason": st.column_config.TextColumn("Why it fits"),
                "url": st.column_config.LinkColumn("Spotify Link"),
            },
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            key="playlist_editor",
        )

        ordered = edited.sort_values("order").reset_index(drop=True)

        st.markdown("### Prominent Play")
        first_url = ordered.iloc[0].get("url", "") if not ordered.empty else ""
        if first_url:
            spotify_embed(first_url, height=152)

        st.markdown("### Spotify-style Queue")
        for _, row in ordered.head(8).iterrows():
            st.markdown(
                f"- **{row['order']}. {row['track']}** by {row['artist']}  \\n  _{row['reason']}_"
            )


# -----------------------------
# Main App Shell
# -----------------------------
def render_header() -> str:
    st.markdown("<div class='main-title'>AI Music Hub</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Analyze albums, generate lyrics, find musical twins, predict genres, and curate playlists.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    global_query = st.text_input("Search bar", placeholder="Try: Arctic Monkeys, nostalgic synthwave, jazz fusion...")
    st.markdown("</div>", unsafe_allow_html=True)
    return global_query


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>AI Music Hub</div>", unsafe_allow_html=True)
        st.caption("Spotify-like AI workstation")

        # Icon + label navigation.
        display_options = list(NAV_ITEMS.values())
        selected_display = st.radio("Menu", display_options)

        feature_name = [k for k, v in NAV_ITEMS.items() if v == selected_display][0]

        st.markdown("---")
        st.caption("Environment")
        sp_ok = get_spotify_client() is not None
        info_banner(sp_ok)

        if not sp_ok:
            st.info("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in secrets/env for full functionality.")

        return feature_name


def main() -> None:
    inject_css()
    selected = sidebar_nav()
    global_query = render_header()

    if selected == "Album Analyzer":
        album_analyzer(global_query)
    elif selected == "Lyrics Generator":
        lyrics_generator(global_query)
    elif selected == "Similarity Finder":
        similarity_finder(global_query)
    elif selected == "Genre Predictor":
        genre_predictor(global_query)
    elif selected == "Playlist Curator":
        playlist_curator(global_query)


if __name__ == "__main__":
    main()
