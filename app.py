"""
Sonic Insight - modular Streamlit music intelligence app.
Deploy:
1) pip install -r requirements.txt
2) streamlit run app.py
"""

import streamlit as st

from sonic_insight.clients import get_spotify_client
from sonic_insight.constants import NAV_ITEMS
from sonic_insight.features import (
    album_analyzer,
    genre_predictor,
    lyrics_generator,
    playlist_curator,
    similarity_finder,
)
from sonic_insight.home import render_home
from sonic_insight.style import inject_css
from sonic_insight.utils import info_banner

st.set_page_config(
    page_title="Sonic Insight",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_header() -> str:
    st.markdown("<div class='main-title'>Sonic Insight</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Analyze albums, generate lyrics, find musical twins, predict genres, and curate playlists.</div>",
        unsafe_allow_html=True,
    )

    env_col, mode_col = st.columns([1, 2])
    with env_col:
        sp_ok = get_spotify_client() is not None
        info_banner(sp_ok)
    with mode_col:
        if not sp_ok:
            st.caption("No-key mode active with public music data. Add Spotify keys later for full Spotify-native results.")

    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    global_query = st.text_input("Search bar", placeholder="Try: Arctic Monkeys, nostalgic synthwave, jazz fusion...")
    st.markdown("</div>", unsafe_allow_html=True)
    return global_query


def top_nav() -> str:
    if "selected_feature" not in st.session_state:
        st.session_state["selected_feature"] = "Home"

    keys = list(NAV_ITEMS.keys())
    current = st.session_state.get("selected_feature", "Home")
    if current not in keys:
        current = "Home"

    st.markdown("<div class='nav-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='nav-title'>Navigation</div>", unsafe_allow_html=True)

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Navigation",
            options=keys,
            default=current,
            format_func=lambda key: NAV_ITEMS[key],
            label_visibility="collapsed",
        )
    else:
        display_options = [NAV_ITEMS[k] for k in keys]
        selected_display = st.radio(
            "Navigation",
            display_options,
            horizontal=True,
            label_visibility="collapsed",
            index=keys.index(current),
        )
        selected = [k for k, v in NAV_ITEMS.items() if v == selected_display][0]

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["selected_feature"] = selected
    return selected


def main() -> None:
    inject_css()
    selected = top_nav()
    global_query = render_header()

    if selected == "Home":
        render_home()
    elif selected == "Album Analyzer":
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
