import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sonic_insight.clients import get_hf_client, get_lyrics_generation_pipeline, get_spotify_client
from sonic_insight.constants import FEATURE_COLUMNS
from sonic_insight.utils import (
    build_genre_dataset,
    extract_audio_features_from_upload,
    fetch_lyrics,
    fetch_track_candidates,
    generate_playlist_queries,
    get_album_tracks_and_features,
    nearest_neighbors,
    render_album_grid,
    render_track_scroll,
    score_sentiment,
    search_spotify_albums,
    search_tracks_for_playlist,
    spotify_embed,
)


def current_plotly_template() -> str:
    theme = str(st.session_state.get("theme_mode", "Dark")).lower()
    return "plotly_white" if theme == "light" else "plotly_dark"


def album_analyzer(global_query: str) -> None:
    st.subheader("Album Analyzer")
    st.caption("Search albums, inspect mood trends, and run lyric sentiment analysis.")

    q = st.text_input("Search artist or album", value=global_query, key="album_search")
    if not q:
        st.info("Type an artist or album to begin.")
        return

    with st.spinner("Searching albums..."):
        albums = search_spotify_albums(q, limit=9)

    if not albums:
        st.warning("No albums found. Try another query.")
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
    fig.update_layout(template=current_plotly_template(), xaxis_title="Track", yaxis_title="Score")
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
        sent_fig.update_layout(template=current_plotly_template())
        st.plotly_chart(sent_fig, use_container_width=True)


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

    st.markdown("Sample style hints from top tracks:")
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


def similarity_finder(global_query: str) -> None:
    st.subheader("Similarity Finder")
    st.caption("Find similar songs and bands using audio-feature embeddings + cosine similarity/FAISS.")

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
            rec_url = str(rec.get("external_url", ""))
            if rec_url and "open.spotify.com" in rec_url:
                spotify_embed(rec_url, height=120)
            elif rec.get("preview_url"):
                st.audio(rec.get("preview_url"))
            elif rec_url:
                st.markdown(f"[Open Track]({rec_url})")
            st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
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
    cm_fig.update_layout(template=current_plotly_template())
    st.plotly_chart(cm_fig, use_container_width=True)

    option = st.radio("Prediction input", ["Track search", "Audio upload"], horizontal=True)

    sample_vec = None
    if option == "Track search":
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
        pfig.update_layout(template=current_plotly_template())
        st.plotly_chart(pfig, use_container_width=True)


def playlist_curator(global_query: str) -> None:
    st.subheader("Playlist Curator")
    st.caption("Create mood-driven playlists with explainable track picks and music links.")

    prefill = st.session_state.pop("quick_playlist_prompt", "") if "quick_playlist_prompt" in st.session_state else ""
    mood_default = prefill or global_query or "nostalgic 90s rock with hopeful lyrics"
    mood = st.text_input("Mood prompt", value=mood_default)
    energy_slider = st.slider("Target energy", 0.0, 1.0, 0.6, 0.05)

    if st.button("Curate Playlist"):
        with st.spinner("Generating playlist concept..."):
            queries = generate_playlist_queries(mood, energy_slider, get_hf_client())
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
                "url": st.column_config.LinkColumn("Music Link"),
            },
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            key="playlist_editor",
        )

        ordered = edited.sort_values("order").reset_index(drop=True)

        st.markdown("### Queue")
        for _, row in ordered.head(8).iterrows():
            st.markdown(
                f"- **{row['order']}. {row['track']}** by {row['artist']}  \\n  _{row['reason']}_"
            )
