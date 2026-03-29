"""
Sonic Insight - modular Streamlit music intelligence app.
Deploy:
1) pip install -r requirements.txt
2) streamlit run app.py
"""

import os

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

NAV_STATE_KEY = "top_nav_feature"
THEME_STATE_KEY = "theme_mode"


def resolve_default_theme() -> str:
    raw = os.getenv("SONIC_INSIGHT_DEFAULT_THEME", "dark").strip().lower()
    return "Light" if raw == "light" else "Dark"


def init_theme_state() -> None:
    if THEME_STATE_KEY not in st.session_state:
        st.session_state[THEME_STATE_KEY] = resolve_default_theme()


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
        status_col, theme_col = st.columns([3, 1])
        with status_col:
            if not sp_ok:
                st.caption("No-key mode active with public music data. Add Spotify keys later for full Spotify-native results.")
        with theme_col:
            st.caption("Theme")
            current_theme = st.session_state.get(THEME_STATE_KEY, "Dark")
            selected_theme = st.selectbox(
                "Theme",
                ["Dark", "Light"],
                index=0 if current_theme == "Dark" else 1,
                key="theme_selectbox",
                label_visibility="collapsed",
            )
            if selected_theme != current_theme:
                st.session_state[THEME_STATE_KEY] = selected_theme
                st.rerun()

    st.caption("Search")
    global_query = st.text_input(
        "Search bar",
        placeholder="Try: Arctic Monkeys, nostalgic synthwave, jazz fusion...",
        label_visibility="collapsed",
    )
    return global_query


def top_nav() -> str:
    keys = list(NAV_ITEMS.keys())
    if NAV_STATE_KEY not in st.session_state or st.session_state.get(NAV_STATE_KEY) not in keys:
        st.session_state[NAV_STATE_KEY] = "Home"

    st.markdown("<div class='nav-title'>Navigation</div>", unsafe_allow_html=True)

    if hasattr(st, "segmented_control"):
        st.segmented_control(
            "Navigation",
            options=keys,
            key=NAV_STATE_KEY,
            format_func=lambda key: NAV_ITEMS[key],
            label_visibility="collapsed",
        )
    else:
        st.radio(
            "Navigation",
            options=keys,
            horizontal=True,
            key=NAV_STATE_KEY,
            format_func=lambda key: NAV_ITEMS[key],
            label_visibility="collapsed",
        )

    selected = st.session_state.get(NAV_STATE_KEY, "Home")
    st.session_state["selected_feature"] = selected
    return selected


def main() -> None:
    init_theme_state()
    inject_css(st.session_state.get(THEME_STATE_KEY, "Dark").lower())
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
