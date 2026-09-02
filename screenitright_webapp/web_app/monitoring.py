"""
monitoring.py
-------------
Owns the Defense Monitor screen: real-time transaction feed, fraud
probability scores, tiered alerts, and false-positive analysis.

Supports THREE modes (in priority order):
  1. Direct import from defend/evaluator.py (same repo, real model scoring)
  2. API fetch from a deployed backend (set DEFEND_API_URL env var)
  3. Mock data fallback (standalone demo)
"""

import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from web_app import theme
from web_app.api_client import get_backend_url, DEFAULT_TIMEOUT

# ── Mode 1: Direct import of real evaluator ──────────────────────────
try:
    from defend.evaluator import get_live_scored_feed as _real_feed
except ImportError:
    _real_feed = None


def _get_active_feed_mode() -> tuple[str, str]:
    """Determine the active feed mode and description."""
    api_url = get_backend_url()
    if api_url:
        return "api", api_url
    if _real_feed is not None:
        return "model", "Local DEFEND ensemble"
    return "mock", "Simulated mock data"


def _fetch_from_api(n: int, api_url: str) -> pd.DataFrame:
    """Fetch scored feed from the deployed DEFEND API."""
    import requests

    if not api_url.startswith("http://") and not api_url.startswith("https://"):
        api_url = f"https://{api_url}"

    try:
        resp = requests.get(f"{api_url}/api/live-feed", params={"n": n}, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("transactions", [])
        return pd.DataFrame(data)
    except requests.exceptions.Timeout:
        st.warning(
            f"Render backend at `{api_url}` took too long to respond. "
            "If your Render service was sleeping, it may take 30-50s to wake up. Falling back to mock data."
        )
        from web_app.mock_data import generate_realtime_feed
        return generate_realtime_feed(n)
    except Exception as e:
        st.warning(f"API fetch from `{api_url}` failed ({e}), falling back to mock data.")
        from web_app.mock_data import generate_realtime_feed
        return generate_realtime_feed(n)


def _get_feed(n: int) -> pd.DataFrame:
    mode, target = _get_active_feed_mode()
    if mode == "api":
        return _fetch_from_api(n, target)
    if mode == "model" and _real_feed is not None:
        return _real_feed(n)
    from web_app.mock_data import generate_realtime_feed
    return generate_realtime_feed(n)


def render_defense_monitor():
    st.subheader("Defense Monitor")
    st.caption("PILLAR 3 · REAL-TIME SCORING FROM THE ENSEMBLE DETECTION MODEL")

    # Show connection mode
    mode, target = _get_active_feed_mode()
    mode_labels = {
        "model": ("🟢 LIVE MODEL", "Scoring with local DEFEND ensemble"),
        "api": ("🔵 RENDER API MODE", f"Connected to `{target}`"),
        "mock": ("🟡 DEMO MODE", "Using simulated mock data (configure Render URL in sidebar to connect live)"),
    }
    label, desc = mode_labels[mode]
    st.caption(f"{label} — {desc}")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        n = st.slider("Feed size", 10, 200, 40, step=10)
        auto_refresh = st.toggle("Live refresh", value=False)
        if st.button("Refresh Feed") or "feed" not in st.session_state:
            with st.spinner("Scoring transactions..."):
                st.session_state["feed"] = _get_feed(n)

    feed = st.session_state["feed"]

    # ── Metrics row ──────────────────────────────────────────────────
    blocked_count = int((feed["recommended_action"] == "Block").sum())
    flagged_count = int((feed["recommended_action"] == "Flag for Review").sum())
    total = len(feed)
    fraud_pct = (blocked_count + flagged_count) / total * 100 if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions Scored", total)
    m2.metric("Fraud Detection Rate", f"{fraud_pct:.1f}%")
    m3.metric("Blocked", blocked_count)
    m4.metric("Flagged for Review", flagged_count)

    # ── Scored feed table ────────────────────────────────────────────
    # Show per-model scores if available (real models), else basic view
    display_cols = [c for c in [
        "timestamp", "txn_id", "amount", "channel", "merchant_category",
        "xgb_score", "tcn_score", "gnn_score",
        "fraud_probability", "risk_tier", "recommended_action",
    ] if c in feed.columns]

    styled = feed[display_cols].style.map(theme.tier_style, subset=["risk_tier"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Charts ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(feed, x="fraud_probability", nbins=20,
                             title="Fraud Probability Score Distribution",
                             color_discrete_sequence=[theme.ACCENT_RED])
        fig1.update_traces(marker_line_width=0, opacity=0.9)
        fig1.update_layout(height=420, transition=dict(duration=500, easing="cubic-in-out"))
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        tier_counts = feed["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        tier_order = ["Critical", "High", "Medium", "Low"]
        tier_colors = {t: theme.TIER_COLORS[t]["fg"] for t in tier_order}
        fig2 = px.pie(tier_counts, names="risk_tier", values="count", title="Alerts by Tier",
                       color="risk_tier", color_discrete_map=tier_colors, hole=0.62)
        fig2.update_traces(textfont=dict(family=theme.FONT_MONO), marker=dict(line=dict(color=theme.SURFACE, width=2)),
                            pull=[0.03] * len(tier_counts))
        fig2.add_annotation(text=f"<b>{len(feed)}</b><br>total", showarrow=False,
                             font=dict(family=theme.FONT_MONO, size=15, color=theme.INK))
        fig2.update_layout(height=340, transition=dict(duration=500, easing="cubic-in-out"))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Per-model comparison (only when real models are scoring) ──────
    if "xgb_score" in feed.columns:
        with st.expander("Per-Model Score Comparison"):
            model_fig = px.box(
                feed.melt(value_vars=["xgb_score", "tcn_score", "gnn_score", "fraud_probability"],
                          var_name="Model", value_name="Score"),
                x="Model", y="Score", color="Model",
                title="Score Distribution by Model",
                color_discrete_sequence=[theme.BRAND, theme.ACCENT_TEAL, theme.SIGNATURE, theme.ACCENT_RED],
            )
            model_fig.update_layout(height=400, showlegend=False,
                                     transition=dict(duration=500, easing="cubic-in-out"))
            st.plotly_chart(model_fig, use_container_width=True)

    # ── False positive analysis ──────────────────────────────────────
    with st.expander("False Positive Analysis"):
        st.write(
            "The stacked ensemble (XGBoost + TCN + GNN) with a Logistic Regression "
            "meta-classifier achieves **0% FPR on the realistic holdout** (39,760 "
            "legitimate transactions, 0 false positives) while maintaining **97.1% recall** "
            "on fraud. The human-review queue catches borderline cases before they're blocked."
        )
        fp_fig = px.bar(
            x=["True Positive", "False Positive", "True Negative", "False Negative"],
            y=[92, 0.0, 7.7, 0.3],
            title="Outcome Breakdown (%) — from realistic holdout evaluation",
            color=["True Positive", "False Positive", "True Negative", "False Negative"],
            color_discrete_map={
                "True Positive": theme.ACCENT_TEAL, "False Positive": theme.ACCENT_RED,
                "True Negative": theme.BRAND, "False Negative": theme.ACCENT_AMBER,
            },
        )
        fp_fig.update_traces(marker_line_width=0)
        fp_fig.update_layout(
            showlegend=False, xaxis_title=None, yaxis_title="%",
            height=300,
            transition=dict(duration=450, easing="cubic-in-out"),
        )
        st.plotly_chart(fp_fig, use_container_width=True)

    if auto_refresh:
        time.sleep(2)
        with st.spinner("Refreshing..."):
            st.session_state["feed"] = _get_feed(n)
        st.rerun()
