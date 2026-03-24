# Sonic Insight

> [!WARNING]
> **UNDER DEVELOPMENT**: This project is actively being built and may change frequently.

Your cozy little AI music lab with big main-stage energy.

Sonic Insight is a Spotify-style Streamlit app where you can explore albums, generate lyrics, discover similar tracks, predict genre vibes, and auto-curate playlists from mood prompts.

## Why It Is Fun

- It looks like a dark, clean music dashboard.
- It feels playful without being chaotic.
- It gives you charts, recommendations, and lyric experiments in one place.

## What You Can Do

- Album Analyzer:
Search artists/albums, pull Spotify audio features, and watch mood trends across tracks.
- Lyrics Generator:
Create verse/chorus style lyrics from a theme + genre + artist mood.
- Similarity Finder:
Find fans-also-like track matches from Spotify features or uploaded audio.
- Genre Predictor:
Train a lightweight model and predict genre with an interactive confusion matrix.
- Playlist Curator:
Turn mood prompts into a playlist draft with explainable track picks.

## Quick Start (2-Minute Version)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Add API secrets

- Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml
- Fill in values:
  - SPOTIPY_CLIENT_ID
  - SPOTIPY_CLIENT_SECRET
  - HF_TOKEN (optional)

3. Launch

```bash
streamlit run app.py
```

Then open: http://localhost:8501

## Tiny Setup Notes

- Spotify keys unlock the full experience.
- Hugging Face token improves text generation and sentiment calls.
- No keys? The app still runs with graceful fallback/demo behavior.

## Deploy Anywhere (Student-Friendly)

- Streamlit Community Cloud
- Hugging Face Spaces

Use the same dependencies and secrets from this repo.

## Tech Stack

- Python + Streamlit
- Spotipy + Hugging Face
- scikit-learn + torch + transformers
- Plotly + pandas + librosa + FAISS

## One-Line Mission

Sonic Insight helps you explore music like a fan, analyze it like a nerd, and remix ideas like an artist.