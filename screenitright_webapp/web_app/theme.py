"""
theme.py
--------
ScreenITright's visual identity — a paper-toned, human-designed product
feel (not a dark hacker console): off-white background, one confident
indigo brand accent, and risk-tier colors (teal/amber/orange/red) that are
the ONLY place color carries meaning, kept consistent everywhere.

Import once from app.py (`inject_css()` + `apply_plotly_theme()` +
`masthead()`), then every chart/table in dashboard.py, monitoring.py and
analytics.py just references COLORWAY / TIER_COLORS and it all matches.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ---- Color tokens -------------------------------------------------------
BG = "#F6F5F0"          # page — warm paper, not stark white
SURFACE = "#FFFFFF"     # cards
BORDER = "#E7E4DA"
INK = "#14213D"         # primary text
MUTED = "#6B7280"

BRAND = "#4640DE"        # the one accent used for interaction/CTAs
BRAND_SOFT = "#EDECFB"
SIGNATURE = "#FF7A29"    # the hand-drawn squiggle + "High" tier

ACCENT_TEAL = "#0E8388"   # cleared / low-risk / healthy
ACCENT_AMBER = "#F0A93B"  # medium
ACCENT_RED = "#E63946"    # critical / blocked

TIER_COLORS = {
    "Critical": {"bg": "#FDEAEA", "fg": "#C0392B"},
    "High": {"bg": "#FFF0E3", "fg": "#D2661A"},
    "Medium": {"bg": "#FFF8E3", "fg": "#B7871B"},
    "Low": {"bg": "#E6F7F5", "fg": "#0B6E71"},
}

COLORWAY = [BRAND, ACCENT_TEAL, SIGNATURE, ACCENT_RED,
            "#2A6F97", "#8A63D2", "#118AB2", "#D64550"]

FONT_LOGO = "'Silkscreen', monospace"     # wordmark ONLY
FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'Inter', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"


def apply_plotly_theme():
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, family=FONT_BODY, size=13),
        title=dict(font=dict(family=FONT_DISPLAY, size=18, color=INK), x=0.02),
        colorway=COLORWAY,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, showline=True),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, showline=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, family=FONT_BODY)),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#FFFFFF", family=FONT_MONO, size=12),
                         bordercolor=INK),
        margin=dict(t=60, l=48, r=24, b=44),
    )
    pio.templates["screenit"] = template
    pio.templates.default = "screenit"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: {FONT_BODY}; }}

    .stApp {{ background: {BG}; }}

    section[data-testid="stSidebar"] {{
        background: {SURFACE}; border-right: 1px solid {BORDER};
    }}

    /* ---------------------------------------------------------------- */
    /* Masthead                                                          */
    /* ---------------------------------------------------------------- */
    .sir-masthead {{
        display: flex; flex-direction: column; align-items: center;
        text-align: center; padding: 38px 0 22px 0;
        animation: sir-fade-in 0.7s ease both;
    }}
    .sir-mark {{
        width: 56px; height: 56px; margin-bottom: 14px; position: relative;
    }}
    .sir-mark svg {{ width: 100%; height: 100%; }}
    .sir-mark .sir-dot {{
        animation: sir-blink 1.8s ease-in-out infinite;
        transform-origin: center;
    }}
    .sir-wordmark {{
        font-family: {FONT_LOGO}; font-weight: 700; font-size: 34px;
        color: {INK}; letter-spacing: 1px; line-height: 1;
    }}
    .sir-wordmark span {{
        display: inline-block;
        opacity: 0;
        animation: sir-letter-in 0.45s cubic-bezier(.22,1,.36,1) forwards;
    }}
    .sir-squiggle {{ height: 14px; margin-top: 6px; }}
    .sir-squiggle path {{
        stroke-dasharray: 220; stroke-dashoffset: 220;
        animation: sir-draw 1s 0.5s cubic-bezier(.65,0,.35,1) forwards;
    }}
    .sir-tagline {{
        font-family: {FONT_MONO}; font-size: 12.5px; color: {MUTED};
        letter-spacing: 0.5px; margin-top: 14px; text-transform: uppercase;
    }}
    .sir-status {{
        display: inline-flex; align-items: center; gap: 8px;
        font-family: {FONT_MONO}; font-size: 11.5px; color: {ACCENT_TEAL};
        border: 1px solid #C9E9E7; background: #EFFAF9;
        padding: 6px 14px; border-radius: 999px; margin-top: 16px;
        letter-spacing: 0.4px;
    }}
    .sir-pulse {{
        width: 7px; height: 7px; border-radius: 50%; background: {ACCENT_TEAL};
        animation: sir-pulse 1.6s ease-in-out infinite;
    }}

    @keyframes sir-fade-in {{
        from {{ opacity: 0; transform: translateY(-8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes sir-draw {{ to {{ stroke-dashoffset: 0; }} }}
    @keyframes sir-pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.3; transform: scale(0.7); }}
    }}
    @keyframes sir-blink {{
        0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.25; }}
    }}
    @keyframes sir-letter-in {{
        from {{ opacity: 0; transform: translateY(10px) rotate(-4deg); }}
        to {{ opacity: 1; transform: translateY(0) rotate(0deg); }}
    }}
    @keyframes sir-chart-in {{
        from {{ opacity: 0; transform: translateY(16px) scale(0.97); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    @keyframes sir-slide-up {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes sir-slide-in-left {{
        from {{ opacity: 0; transform: translateX(-18px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes sir-pop-in {{
        0% {{ opacity: 0; transform: scale(0.85); }}
        70% {{ opacity: 1; transform: scale(1.03); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes sir-shimmer {{
        0% {{ background-position: -200px 0; }}
        100% {{ background-position: 200px 0; }}
    }}

    /* ---------------------------------------------------------------- */
    /* Section headers                                                   */
    /* ---------------------------------------------------------------- */
    h2, h3 {{
        font-family: {FONT_DISPLAY} !important; color: {INK} !important;
        letter-spacing: -0.3px;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        font-family: {FONT_MONO} !important; color: {MUTED} !important;
        letter-spacing: 0.3px; text-transform: uppercase; font-size: 11.5px !important;
    }}

    /* ---------------------------------------------------------------- */
    /* Tabs — centered, animated underline                               */
    /* ---------------------------------------------------------------- */
    div[data-baseweb="tab-list"] {{ justify-content: center; gap: 6px; }}
    button[data-baseweb="tab"] {{
        font-family: {FONT_MONO}; font-size: 12.5px; letter-spacing: 0.5px;
        color: {MUTED} !important; padding: 10px 20px !important;
        transition: color 0.2s ease;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {BRAND} !important; }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {BRAND} !important; height: 3px !important;
        border-radius: 3px 3px 0 0; transition: all 0.35s cubic-bezier(.65,0,.35,1) !important;
    }}
    div[data-baseweb="tab-border"] {{ background-color: {BORDER} !important; }}

    /* ---------------------------------------------------------------- */
    /* Metric cards — lift on hover                                      */
    /* ---------------------------------------------------------------- */
    div[data-testid="stMetric"] {{
        background: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 16px 18px 12px 18px;
        box-shadow: 0 1px 2px rgba(20,33,61,0.04);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        animation: sir-fade-in 0.5s ease both;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(20,33,61,0.09);
    }}
    div[data-testid="stMetricValue"] {{
        font-family: {FONT_MONO} !important; color: {INK} !important; font-weight: 600 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-family: {FONT_BODY} !important; color: {MUTED} !important;
        font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.4px;
    }}

    /* ---------------------------------------------------------------- */
    /* Buttons                                                           */
    /* ---------------------------------------------------------------- */
    .stButton > button, .stDownloadButton > button {{
        background: {BRAND} !important; color: #FFFFFF !important;
        border: none !important; font-weight: 600 !important;
        border-radius: 999px !important; padding: 8px 20px !important;
        box-shadow: 0 2px 8px rgba(70,64,222,0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: #3b36c9 !important; transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(70,64,222,0.32);
    }}
    .stButton > button:active, .stDownloadButton > button:active {{ transform: translateY(0) scale(0.98); }}

    /* ---------------------------------------------------------------- */
    /* Expanders, dataframes, misc containers                            */
    /* ---------------------------------------------------------------- */
    div[data-testid="stExpander"] {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
        box-shadow: 0 1px 2px rgba(20,33,61,0.04);
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden;
    }}
    div[data-testid="stAlert"] {{ border-radius: 10px; }}

    /* Sliders / toggle accent */
    div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {BRAND} !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div[data-baseweb="slider"] div[role="slider"]:hover {{
        transform: scale(1.25);
        box-shadow: 0 0 0 6px {BRAND_SOFT};
    }}

    /* ---------------------------------------------------------------- */
    /* Sidebar — slides in from the left on load                        */
    /* ---------------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        animation: sir-slide-in-left 0.55s cubic-bezier(.22,1,.36,1) both;
    }}
    section[data-testid="stSidebar"] * {{
        transition: color 0.15s ease;
    }}

    /* ---------------------------------------------------------------- */
    /* Tab panels — fade + rise every time a tab is switched             */
    /* ---------------------------------------------------------------- */
    div[data-baseweb="tab-panel"] {{
        animation: sir-slide-up 0.4s cubic-bezier(.22,1,.36,1) both;
    }}
    button[data-baseweb="tab"] {{
        transform-origin: center bottom;
    }}
    button[data-baseweb="tab"]:hover {{
        transform: translateY(-2px);
    }}

    /* ---------------------------------------------------------------- */
    /* Plotly charts — pop/scale in, lift slightly on hover              */
    /* Each chart is sized independently in its own render call, so     */
    /* this only governs the *entrance*, not the box size.               */
    /* ---------------------------------------------------------------- */
    div[data-testid="stPlotlyChart"] {{
        animation: sir-chart-in 0.65s cubic-bezier(.22,1,.36,1) both;
        border-radius: 14px;
        background: {SURFACE};
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }}
    div[data-testid="stPlotlyChart"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 14px 30px rgba(20,33,61,0.10);
    }}

    /* Stagger successive charts on the same page slightly */
    div[data-testid="stVerticalBlock"] > div:nth-of-type(2) div[data-testid="stPlotlyChart"] {{ animation-delay: 0.06s; }}
    div[data-testid="stVerticalBlock"] > div:nth-of-type(3) div[data-testid="stPlotlyChart"] {{ animation-delay: 0.12s; }}
    div[data-testid="stVerticalBlock"] > div:nth-of-type(4) div[data-testid="stPlotlyChart"] {{ animation-delay: 0.18s; }}

    /* ---------------------------------------------------------------- */
    /* Metric cards — staggered pop-in per column position               */
    /* ---------------------------------------------------------------- */
    div[data-testid="stMetric"] {{ animation: sir-pop-in 0.5s cubic-bezier(.22,1,.36,1) both; }}
    div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] {{ animation-delay: 0.02s; }}
    div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {{ animation-delay: 0.09s; }}
    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {{ animation-delay: 0.16s; }}
    div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"] {{ animation-delay: 0.23s; }}

    /* ---------------------------------------------------------------- */
    /* DataFrames — fade in, gentle row highlight on hover                */
    /* ---------------------------------------------------------------- */
    div[data-testid="stDataFrame"] {{
        animation: sir-fade-in 0.55s ease both;
        transition: box-shadow 0.2s ease;
    }}
    div[data-testid="stDataFrame"]:hover {{
        box-shadow: 0 6px 18px rgba(20,33,61,0.08);
    }}
    div[data-testid="stDataFrame"] table tbody tr {{
        transition: background-color 0.15s ease, transform 0.15s ease;
    }}
    div[data-testid="stDataFrame"] table tbody tr:hover {{
        background-color: {BRAND_SOFT} !important;
    }}

    /* ---------------------------------------------------------------- */
    /* Expanders — lift on hover, content settles in on open             */
    /* ---------------------------------------------------------------- */
    div[data-testid="stExpander"] {{
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: sir-fade-in 0.5s ease both;
    }}
    div[data-testid="stExpander"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(20,33,61,0.08);
    }}
    div[data-testid="stExpanderDetails"] {{
        animation: sir-slide-up 0.35s ease both;
    }}

    /* ---------------------------------------------------------------- */
    /* Alerts / success / info banners — slide up into view              */
    /* ---------------------------------------------------------------- */
    div[data-testid="stAlert"] {{
        animation: sir-slide-up 0.4s cubic-bezier(.22,1,.36,1) both;
    }}

    /* ---------------------------------------------------------------- */
    /* Selects / multiselects — soft focus glow, animated chip pop-in    */
    /* ---------------------------------------------------------------- */
    div[data-baseweb="select"] {{
        transition: box-shadow 0.2s ease, transform 0.15s ease;
        border-radius: 8px;
    }}
    div[data-baseweb="select"]:focus-within {{
        box-shadow: 0 0 0 3px {BRAND_SOFT};
    }}
    div[data-baseweb="tag"] {{
        animation: sir-pop-in 0.25s ease both;
    }}

    /* ---------------------------------------------------------------- */
    /* Toggle switch */
    /* ---------------------------------------------------------------- */
    div[data-testid="stToggle"] label div:first-child {{
        transition: background-color 0.25s ease;
    }}

    /* ---------------------------------------------------------------- */
    /* Headings — quick rise-in whenever a screen mounts                 */
    /* ---------------------------------------------------------------- */
    h2, h3 {{
        animation: sir-slide-up 0.45s cubic-bezier(.22,1,.36,1) both;
    }}
    [data-testid="stCaptionContainer"] {{
        animation: sir-fade-in 0.5s ease 0.05s both;
    }}
    </style>
    """, unsafe_allow_html=True)


def masthead(status_text: str = "CLOSED-LOOP ACTIVE"):
    """Centered brand masthead: viewfinder mark, pixel wordmark, hand-drawn
    squiggle underline that draws itself in, tagline, and a live status pill."""

    # Wordmark casing: only "IT" is uppercase — "screenITright".
    # Built as individual <span>s so each letter can stagger-fade in on load.
    wordmark_text = "screen-IT-right"
    letters_html = "".join(
        f'<span style="animation-delay:{0.45 + i * 0.045:.3f}s">{ch}</span>'
        for i, ch in enumerate(wordmark_text)
    )

    st.markdown(f"""
    <div class="sir-masthead">
        <div class="sir-mark">
            <svg viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="28" cy="28" r="27" fill="{BRAND_SOFT}" stroke="{BRAND}" stroke-width="1"/>
                <path d="M18 20V16.5C18 15.1193 19.1193 14 20.5 14H24" stroke="{BRAND}" stroke-width="2.4" stroke-linecap="round"/>
                <path d="M38 20V16.5C38 15.1193 36.8807 14 35.5 14H32" stroke="{BRAND}" stroke-width="2.4" stroke-linecap="round"/>
                <path d="M18 36V39.5C18 40.8807 19.1193 42 20.5 42H24" stroke="{BRAND}" stroke-width="2.4" stroke-linecap="round"/>
                <path d="M38 36V39.5C38 40.8807 36.8807 42 35.5 42H32" stroke="{BRAND}" stroke-width="2.4" stroke-linecap="round"/>
                <circle class="sir-dot" cx="28" cy="28" r="4.5" fill="{SIGNATURE}"/>
            </svg>
        </div>
        <div class="sir-wordmark">{letters_html}</div>
        <svg class="sir-squiggle" viewBox="0 0 180 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M2 8C22 2 30 14 48 8C66 2 74 14 92 8C110 2 118 14 136 8C154 2 162 14 178 8"
                  stroke="{SIGNATURE}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
        <div class="sir-tagline">AI fraud screening · attack → learn → defend → discover</div>
        <div class="sir-status"><span class="sir-pulse"></span>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)


def tier_style(val: str) -> str:
    """Row/cell style for a risk_tier or severity column via st.dataframe(...).style.map()."""
    c = TIER_COLORS.get(val)
    if not c:
        return ""
    return f"background-color:{c['bg']}; color:{c['fg']}; font-weight:600; font-family:{FONT_MONO};"