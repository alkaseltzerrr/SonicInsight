# AI Music Hub

Spotify-inspired full-stack music intelligence app built with Python + Streamlit.

## Features

- Album Analyzer: Spotify album search, audio feature trends, and lyric sentiment.
- Lyrics Generator: Prompt-based song generation with Hugging Face model fallback.
- Similarity Finder: Track/audio similarity using Spotify features + cosine/FAISS.
- Genre Predictor: Lightweight Decision Tree classifier with Plotly confusion matrix.
- Playlist Curator: Mood-based playlist generation with Spotify embeds and editable queue.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add secrets:

- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
- Fill in:
	- `SPOTIPY_CLIENT_ID`
	- `SPOTIPY_CLIENT_SECRET`
	- optional `HF_TOKEN`

3. Run app:

```bash
streamlit run app.py
```

## Deployment Notes

- Streamlit Cloud and Hugging Face Spaces are supported.
- Add the same secrets in your deployment platform secret manager.
- If Spotify/HF secrets are missing, the app runs in graceful demo/fallback mode.