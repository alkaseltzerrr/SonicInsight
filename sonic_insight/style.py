import streamlit as st


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

        .nav-shell {
            background: linear-gradient(180deg, #111 0%, #0b0b0b 100%);
            border: 1px solid #252525;
            border-radius: 16px;
            padding: 0.6rem 0.9rem 0.2rem 0.9rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
        }

        .nav-title {
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9aa0a6;
            margin-bottom: 0.35rem;
        }

        .home-widget {
            background: linear-gradient(150deg, #141414 0%, #101010 100%);
            border: 1px solid #252525;
            border-radius: 14px;
            padding: 1.05rem;
            min-height: 158px;
            margin-bottom: 0.55rem;
        }

        .home-widget h4 {
            margin: 0 0 0.55rem 0;
            font-size: 1.02rem;
        }

        .home-widget p {
            color: #b5b5b5;
            margin: 0;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .home-row-gap {
            height: 0.4rem;
        }

        .quick-widget {
            background: linear-gradient(160deg, #111 0%, #0d0d0d 100%);
            border: 1px solid #252525;
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
            min-height: 175px;
        }

        .quick-widget-title {
            color: #c7c7c7;
            font-size: 0.92rem;
            margin-bottom: 0.3rem;
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
            min-height: 2.55rem;
            padding: 0.45rem 1rem;
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
