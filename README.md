# Sonic Insight

> [!WARNING]
> **UNDER DEVELOPMENT**: This project is actively being built and may change frequently.

Sonic Insight is a Streamlit app for music analysis and playlist experimentation.

## Features

- Analyze albums and track-level audio features.
- Generate lyrics from theme, genre, and artist mood.
- Find similar tracks from Spotify features or uploaded audio.
- Predict genre with a lightweight model and confusion matrix.
- Create playlist drafts from mood prompts.

## Quick Start

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

3. Run the app

```bash
streamlit run app.py
```

Then open: http://localhost:8501

## Notes

- Spotify keys unlock the full experience.
- Hugging Face token improves text generation and sentiment calls.
- No keys? The app still runs with graceful fallback/demo behavior.

## Deploy

- Streamlit Community Cloud
- Hugging Face Spaces

Use the same dependencies and secrets from this repo.

## Tech Stack

- Python + Streamlit
- Spotipy + Hugging Face
- scikit-learn + torch + transformers
- Plotly + pandas + librosa + FAISS
