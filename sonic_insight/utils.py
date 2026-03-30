import hashlib
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from sonic_insight.clients import get_sentiment_pipeline, get_spotify_client
from sonic_insight.constants import FEATURE_COLUMNS

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    import librosa
except Exception:
    librosa = None


logger = logging.getLogger(__name__)


def report_error(action: str, error: Exception, **context) -> None:
    context_str = ", ".join(f"{k}={v!r}" for k, v in context.items()) or "no_context"
    logger.warning(
        "Sonic Insight failure | action=%s | context=%s | error=%s",
        action,
        context_str,
        repr(error),
        exc_info=True,
    )


def _retry_wait_seconds(response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return float(2**attempt)


def get_with_retry(url: str, *, params: Optional[dict], timeout: int, action: str, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 429 and attempt < max_attempts - 1:
                wait_s = _retry_wait_seconds(response, attempt)
                logger.warning(
                    "Rate limited | action=%s | attempt=%s/%s | wait=%.2fs",
                    action,
                    attempt + 1,
                    max_attempts,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

            if 500 <= response.status_code < 600 and attempt < max_attempts - 1:
                wait_s = _retry_wait_seconds(response, attempt)
                logger.warning(
                    "Transient upstream error | action=%s | status=%s | attempt=%s/%s | wait=%.2fs",
                    action,
                    response.status_code,
                    attempt + 1,
                    max_attempts,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

            return response
        except requests.RequestException as exc:
            report_error(action, exc, attempt=attempt + 1, max_attempts=max_attempts, url=url)
            if attempt < max_attempts - 1:
                time.sleep(float(2**attempt))
                continue
            return None

    return None


def info_banner(sp_available: bool) -> None:
    badge = "Connected" if sp_available else "Public data mode"
    st.markdown(f"<span class='spotify-badge'>Music Data Source: {badge}</span>", unsafe_allow_html=True)


def pseudo_feature_vector(seed_text: str) -> Dict[str, float]:
    """Create deterministic, Spotify-like numeric features for fallback datasets."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return {
        "danceability": float(np.clip(rng.normal(0.62, 0.18), 0, 1)),
        "energy": float(np.clip(rng.normal(0.63, 0.2), 0, 1)),
        "valence": float(np.clip(rng.normal(0.55, 0.2), 0, 1)),
        "tempo": float(np.clip(rng.normal(118, 28), 60, 190)),
        "acousticness": float(np.clip(rng.normal(0.35, 0.23), 0, 1)),
        "instrumentalness": float(np.clip(rng.normal(0.16, 0.2), 0, 1)),
        "speechiness": float(np.clip(rng.normal(0.11, 0.09), 0, 1)),
    }


@st.cache_data(show_spinner=False)
def search_itunes_albums(query: str, limit: int = 9) -> List[dict]:
    endpoint = "https://itunes.apple.com/search"
    try:
        response = get_with_retry(
            endpoint,
            params={"term": query, "entity": "album", "limit": limit},
            timeout=10,
            action="search_itunes_albums",
        )
        if response is None or response.status_code != 200:
            return []
        results = response.json().get("results", [])
    except Exception as exc:
        report_error("search_itunes_albums", exc, query=query, limit=limit)
        return []

    albums = []
    for item in results:
        collection_id = item.get("collectionId")
        if not collection_id:
            continue
        art = item.get("artworkUrl100", "")
        albums.append(
            {
                "id": f"itunes:{collection_id}",
                "name": item.get("collectionName", "Unknown"),
                "artists": [{"name": item.get("artistName", "Unknown Artist")}],
                "release_date": item.get("releaseDate", ""),
                "images": [{"url": art.replace("100x100", "600x600") if art else ""}],
                "source": "itunes",
            }
        )
    return albums


@st.cache_data(show_spinner=False)
def search_spotify_albums(query: str, limit: int = 9) -> List[dict]:
    sp = get_spotify_client()
    if sp is not None:
        try:
            data = sp.search(q=query, type="album", limit=limit)
            items = data.get("albums", {}).get("items", [])
            if items:
                for item in items:
                    item["source"] = "spotify"
                return items
        except Exception:
            pass

    return search_itunes_albums(query, limit=limit)


@st.cache_data(show_spinner=False)
def get_itunes_album_tracks_and_features(collection_id: str) -> pd.DataFrame:
    endpoint = "https://itunes.apple.com/lookup"
    try:
        response = get_with_retry(
            endpoint,
            params={"id": collection_id, "entity": "song"},
            timeout=10,
            action="get_itunes_album_tracks_and_features",
        )
        if response is None or response.status_code != 200:
            return pd.DataFrame()
        results = response.json().get("results", [])
    except Exception as exc:
        report_error("get_itunes_album_tracks_and_features", exc, collection_id=collection_id)
        return pd.DataFrame()

    rows = []
    for item in results:
        if item.get("wrapperType") != "track":
            continue
        track_name = item.get("trackName", "Unknown Track")
        artist_name = item.get("artistName", "Unknown Artist")
        features = pseudo_feature_vector(f"{artist_name}-{track_name}")
        rows.append(
            {
                "track_id": f"itunes-track-{item.get('trackId', track_name)}",
                "track_name": track_name,
                "duration_ms": item.get("trackTimeMillis", 0),
                "track_number": item.get("trackNumber", len(rows) + 1),
                **features,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("track_number")


@st.cache_data(show_spinner=False)
def get_album_tracks_and_features(album_id: str) -> pd.DataFrame:
    if str(album_id).startswith("itunes:"):
        return get_itunes_album_tracks_and_features(str(album_id).split("itunes:", 1)[1])

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
    except Exception as exc:
        report_error("get_album_tracks_and_features", exc, album_id=album_id)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_lyrics(artist: str, title: str) -> str:
    endpoint = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    try:
        response = get_with_retry(
            endpoint,
            params=None,
            timeout=8,
            action="fetch_lyrics",
        )
        if response is not None and response.status_code == 200:
            data = response.json()
            return data.get("lyrics", "")[:4000]
    except Exception as exc:
        report_error("fetch_lyrics", exc, artist=artist, title=title)
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


def fetch_track_candidates(query: str, limit: int = 25) -> pd.DataFrame:
    sp = get_spotify_client()
    if sp is None:
        endpoint = "https://itunes.apple.com/search"
        try:
            response = get_with_retry(
                endpoint,
                params={"term": query, "entity": "song", "limit": limit},
                timeout=10,
                action="fetch_track_candidates",
            )
            if response is None or response.status_code != 200:
                return pd.DataFrame()
            tracks = response.json().get("results", [])
        except Exception as exc:
            report_error("fetch_track_candidates_itunes", exc, query=query, limit=limit)
            return pd.DataFrame()

        rows = []
        for t in tracks:
            track_name = t.get("trackName")
            artist_name = t.get("artistName")
            if not track_name or not artist_name:
                continue
            features = pseudo_feature_vector(f"{artist_name}-{track_name}")
            rows.append(
                {
                    "id": f"itunes-{t.get('trackId', track_name)}",
                    "name": track_name,
                    "artist": artist_name,
                    "album": t.get("collectionName", "Unknown Album"),
                    "external_url": t.get("trackViewUrl", ""),
                    "preview_url": t.get("previewUrl", ""),
                    **features,
                }
            )

        return pd.DataFrame(rows)

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
                    "preview_url": t.get("preview_url"),
                    **{c: f.get(c, np.nan) for c in FEATURE_COLUMNS},
                }
            )

        return pd.DataFrame(rows).dropna()
    except Exception as exc:
        report_error("fetch_track_candidates", exc, query=query, limit=limit)
        return pd.DataFrame()


def nearest_neighbors(df: pd.DataFrame, query_vector: np.ndarray, top_k: int = 8) -> pd.DataFrame:
    X_raw = df[FEATURE_COLUMNS].values.astype("float32")
    q_raw = query_vector.astype("float32").reshape(1, -1)

    # Normalize each feature to [0, 1] using candidate-pool bounds so tempo and
    # other larger-scale features do not dominate nearest-neighbor distance.
    mins = X_raw.min(axis=0, keepdims=True)
    maxs = X_raw.max(axis=0, keepdims=True)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))

    X = (X_raw - mins) / denom
    q = np.clip((q_raw - mins) / denom, 0.0, 1.0)

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


@st.cache_data(show_spinner=False)
def build_genre_dataset() -> pd.DataFrame:
    sp = get_spotify_client()
    genres = ["rock", "pop", "hip hop", "jazz", "electronic", "classical"]
    rows = []

    if sp is None:
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


def generate_playlist_queries(mood_prompt: str, target_energy: float, hf_client) -> List[str]:
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
        for q in queries:
            try:
                response = get_with_retry(
                    "https://itunes.apple.com/search",
                    params={"term": q, "entity": "song", "limit": 2},
                    timeout=10,
                    action="search_tracks_for_playlist",
                )
                if response is None or response.status_code != 200:
                    continue
                items = response.json().get("results", [])
                for t in items:
                    track = t.get("trackName")
                    artist = t.get("artistName")
                    if not track or not artist:
                        continue
                    rows.append(
                        {
                            "order": len(rows) + 1,
                            "track": track,
                            "artist": artist,
                            "reason": f"Selected from mood phrase: {q}",
                            "url": t.get("trackViewUrl", ""),
                        }
                    )
                    if len(rows) >= cap:
                        return pd.DataFrame(rows)
            except Exception as exc:
                report_error("search_tracks_for_playlist_itunes", exc, query=q)
                continue

        if rows:
            return pd.DataFrame(rows)

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
