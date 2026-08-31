"""
app.py
------
Entry point for the ScreenITright web prototype. Run with: streamlit run app.py

This file wires together the three modules Member 5 owns:
    web_app/dashboard.py   -> Attack Discovery Dashboard + Generation Studio
    web_app/monitoring.py  -> Defense Monitor
    web_app/analytics.py   -> Feedback Loop Visualization
plus the shared visual identity in web_app/theme.py.
"""

import streamlit as st
from web_app import theme
from web_app.dashboard import render_attack_discovery, render_generation_studio
from web_app.monitoring import render_defense_monitor
from web_app.analytics import render_feedback_loop

st.set_page_config(page_title="ScreenITright", page_icon="🛂", layout="wide")

theme.apply_plotly_theme()
theme.inject_css()
theme.masthead()

tab1, tab2, tab3, tab4 = st.tabs([
    "01 · IDENTIFY", "02 · GENERATE", "03 · DEFEND", "04 · FEEDBACK",
])

with tab1:
    filtered_attacks = render_attack_discovery()

with tab2:
    render_generation_studio(filtered_attacks)

with tab3:
    render_defense_monitor()

with tab4:
    render_feedback_loop()

st.sidebar.markdown("#### System Map")
st.sidebar.markdown(
    "`01` **Identify** — attack discovery\n\n"
    "`02` **Generate** — synthetic fraud data\n\n"
    "`03` **Defend** — real-time detection\n\n"
    "`04` **Feedback** — closed-loop retraining\n\n"
    "---\n"
    "Running on **mock data** wherever a teammate's module isn't wired in "
    "yet — see the `try/except` imports at the top of each file in `web_app/`."
)
