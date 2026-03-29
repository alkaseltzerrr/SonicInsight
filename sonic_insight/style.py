import streamlit as st


def inject_css(theme: str = "dark") -> None:
    palettes = {
        "dark": {
            "bg_main": "#050505",
            "bg_accent": "#1a1a1a",
            "text_primary": "#ffffff",
            "text_muted": "#b3b3b3",
            "card_bg": "#121212",
            "card_bg_alt": "#0f0f0f",
            "border": "#252525",
            "border_strong": "#2d2d2d",
            "shadow": "rgba(0, 0, 0, 0.25)",
            "button_bg": "#1DB954",
            "button_text": "#000000",
            "button_hover_bg": "#22d462",
            "button_hover_text": "#000000",
            "badge_bg": "rgba(29, 185, 84, 0.15)",
            "badge_border": "rgba(29, 185, 84, 0.5)",
            "badge_text": "#79ffad",
            "input_bg": "#111111",
        },
        "light": {
            "bg_main": "#f4f7fb",
            "bg_accent": "#e8effa",
            "text_primary": "#111827",
            "text_muted": "#4b5563",
            "card_bg": "#ffffff",
            "card_bg_alt": "#f8fbff",
            "border": "#d7e0ec",
            "border_strong": "#c9d5e5",
            "shadow": "rgba(17, 24, 39, 0.08)",
            "button_bg": "#1DB954",
            "button_text": "#062013",
            "button_hover_bg": "#17a348",
            "button_hover_text": "#ffffff",
            "badge_bg": "rgba(29, 185, 84, 0.12)",
            "badge_border": "rgba(29, 185, 84, 0.35)",
            "badge_text": "#0f7a36",
            "input_bg": "#ffffff",
        },
    }

    selected = palettes["light"] if theme == "light" else palettes["dark"]

    st.markdown(
        """
        <style>
        :root {
            --spotify-green: %(button_bg)s;
            --text-primary: %(text_primary)s;
            --text-muted: %(text_muted)s;
            --bg-main: %(bg_main)s;
            --bg-accent: %(bg_accent)s;
            --card-bg: %(card_bg)s;
            --card-bg-alt: %(card_bg_alt)s;
            --border-color: %(border)s;
            --border-strong: %(border_strong)s;
            --shadow-color: %(shadow)s;
            --button-bg: %(button_bg)s;
            --button-text: %(button_text)s;
            --button-hover-bg: %(button_hover_bg)s;
            --button-hover-text: %(button_hover_text)s;
            --badge-bg: %(badge_bg)s;
            --badge-border: %(badge_border)s;
            --badge-text: %(badge_text)s;
            --input-bg: %(input_bg)s;
        }

        html, body, [class*="css"] {
            background: radial-gradient(circle at top left, var(--bg-accent) 0%, var(--bg-main) 50%, var(--bg-main) 100%);
            color: var(--text-primary);
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, var(--bg-accent) 0%, var(--bg-main) 50%, var(--bg-main) 100%);
            color: var(--text-primary);
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.4px;
            margin-bottom: 0.2rem;
            color: var(--text-primary);
        }

        .sub-title {
            color: var(--text-muted);
            margin-bottom: 1rem;
        }

        .hero-card {
            background: linear-gradient(135deg, var(--card-bg) 0%, var(--card-bg-alt) 100%);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .nav-shell {
            background: linear-gradient(180deg, var(--card-bg) 0%, var(--card-bg-alt) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 0.6rem 0.9rem 0.2rem 0.9rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 24px var(--shadow-color);
        }

        .nav-title {
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.35rem;
        }

        .home-widget {
            background: linear-gradient(150deg, var(--card-bg) 0%, var(--card-bg-alt) 100%);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.05rem;
            min-height: 158px;
            margin-bottom: 0.55rem;
        }

        .home-widget h4 {
            margin: 0 0 0.55rem 0;
            font-size: 1.02rem;
            color: var(--text-primary);
        }

        .home-widget p {
            color: var(--text-muted);
            margin: 0;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .home-row-gap {
            height: 0.4rem;
        }

        .quick-widget {
            background: linear-gradient(160deg, var(--card-bg) 0%, var(--card-bg-alt) 100%);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
            min-height: 175px;
        }

        .quick-widget-title {
            color: var(--text-muted);
            font-size: 0.92rem;
            margin-bottom: 0.3rem;
        }

        .feature-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 0.8rem;
            transition: transform 0.25s ease;
            height: 100%;
            color: var(--text-primary);
        }

        .feature-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-strong);
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
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 10px;
            color: var(--text-primary);
        }

        .spotify-badge {
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--badge-text);
            border-radius: 999px;
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }

        .stButton > button {
            background: var(--button-bg);
            color: var(--button-text);
            border: none;
            border-radius: 999px;
            font-weight: 700;
            min-height: 2.55rem;
            padding: 0.45rem 1rem;
        }

        .stButton > button:hover {
            background: var(--button-hover-bg);
            color: var(--button-hover-text);
        }

        .sidebar-header {
            color: var(--spotify-green);
            font-weight: 700;
        }

        .stTextInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {
            background: var(--input-bg);
            color: var(--text-primary);
            border-color: var(--border-color);
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
        """
        % selected,
        unsafe_allow_html=True,
    )
