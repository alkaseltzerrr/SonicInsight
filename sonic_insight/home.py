import streamlit as st


def render_home() -> None:
    st.subheader("Welcome to Sonic Insight")
    st.caption("Start from any feature card below or use the top navigation bar.")

    cards = [
        (
            "Album Analyzer",
            "Search albums and inspect mood trends across tracks with interactive charts.",
            "🎵",
        ),
        (
            "Lyrics Generator",
            "Generate lyrics by style, genre, and theme with sentiment scoring.",
            "✍️",
        ),
        (
            "Similarity Finder",
            "Discover fans-also-like recommendations from tracks or uploaded audio.",
            "🔎",
        ),
        (
            "Genre Predictor",
            "Predict genre from audio features and inspect model confidence.",
            "🎚️",
        ),
        (
            "Playlist Curator",
            "Turn mood prompts into ready-to-edit playlists with track links.",
            "📻",
        ),
    ]

    for row_idx in range(0, len(cards), 3):
        row_cards = cards[row_idx : row_idx + 3]
        cols = st.columns(3, gap="large")
        for col_idx in range(3):
            with cols[col_idx]:
                if col_idx < len(row_cards):
                    title, desc, icon = row_cards[col_idx]
                    idx = row_idx + col_idx
                    st.markdown(
                        f"<div class='home-widget'><h4>{icon} {title}</h4><p>{desc}</p></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Open {title}", key=f"home_open_{idx}", use_container_width=True):
                        st.session_state["selected_feature"] = title
                        st.session_state["top_nav_feature"] = title
                        st.rerun()
                else:
                    st.empty()
        st.markdown("<div class='home-row-gap'></div>", unsafe_allow_html=True)

    st.markdown("### Quick Widgets")
    q1, q2, q3 = st.columns([1, 1, 1])
    with q1:
        st.markdown("<div class='quick-widget-title'>Active Modes</div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.metric("", "5", "All online/offline-ready")
    with q2:
        st.markdown("<div class='quick-widget-title'>Today\'s vibe</div>", unsafe_allow_html=True)
        with st.container(border=True):
            mood = st.slider("Today\'s vibe", 0, 100, 67, label_visibility="collapsed")
            st.caption(f"Mood intensity: {mood}%")
    with q3:
        st.markdown("<div class='quick-widget-title'>Quick playlist prompt</div>", unsafe_allow_html=True)
        with st.container(border=True):
            quick_prompt = st.text_input("Quick playlist prompt", placeholder="rainy chill indie", label_visibility="collapsed")
            if st.button("Use In Playlist Curator", key="quick_playlist_jump", use_container_width=True):
                st.session_state["selected_feature"] = "Playlist Curator"
                st.session_state["top_nav_feature"] = "Playlist Curator"
                st.session_state["quick_playlist_prompt"] = quick_prompt
                st.rerun()
