"""
dashboard.py
------------
Owns two of the four prototype screens:
  1. Attack Discovery Dashboard  (visualizes Pillar 1 - Identify)
  2. Generation Studio           (drives Pillar 2 - Generate)

Swap-in point for real teammate code:
    Replace the try/except imports below once Member 1 / Member 2 land
    their files in identify/ and generate/.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from web_app import theme

try:
    from identify.attack_database import get_all_attacks as generate_attack_taxonomy
except ImportError:
    from web_app.mock_data import generate_attack_taxonomy

try:
    from generate.transaction_sim import simulate_transactions as generate_synthetic_transactions
except ImportError:
    from web_app.mock_data import generate_synthetic_transactions


@st.cache_data
def _load_attacks():
    return generate_attack_taxonomy()


def render_attack_discovery():
    st.subheader("Attack Discovery Dashboard")
    st.caption("LIVE VIEW · THREAT DATABASE PRODUCED BY THE IDENTIFY PILLAR")

    df = _load_attacks()

    col1, col2, col3 = st.columns(3)
    with col1:
        categories = st.multiselect("Category", sorted(df["category"].unique()))
    with col2:
        severities = st.multiselect("Severity", sorted(df["severity"].unique()))
    with col3:
        capabilities = st.multiselect("GenAI Capability", sorted(df["genai_capability"].unique()))

    filtered = df.copy()
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if severities:
        filtered = filtered[filtered["severity"].isin(severities)]
    if capabilities:
        filtered = filtered[filtered["genai_capability"].isin(capabilities)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Attacks Shown", len(filtered))
    m2.metric("Total Discovered", len(df))
    m3.metric("Critical Severity", int((filtered["severity"] == "Critical").sum()))

    fig = px.scatter(
        filtered, x="feasibility", y="impact_score", color="category",
        size="priority", hover_name="name", hover_data=["severity", "attack_id"],
        title="Attack Landscape · Feasibility vs. Impact",
        color_discrete_sequence=theme.COLORWAY,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color=theme.SURFACE), opacity=0.88))
    # Quadrant guides — turns a bare scatter into a prioritization map
    fig.add_hline(y=filtered["impact_score"].median(), line_dash="dot", line_color=theme.BORDER, line_width=1)
    fig.add_vline(x=filtered["feasibility"].median(), line_dash="dot", line_color=theme.BORDER, line_width=1)
    quad_labels = [
        (0.98, 0.98, "PRIORITY THREATS", "right", "top"),
        (0.02, 0.98, "EMERGING", "left", "top"),
        (0.98, 0.02, "CONTAIN", "right", "bottom"),
        (0.02, 0.02, "LOW PRIORITY", "left", "bottom"),
    ]
    for xr, yr, label, xa, ya in quad_labels:
        fig.add_annotation(xref="x domain", yref="y domain", x=xr, y=yr, text=label,
                            showarrow=False, xanchor=xa, yanchor=ya,
                            font=dict(family=theme.FONT_MONO, size=10, color=theme.MUTED))
    fig.update_layout(
        height=560,
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    st.plotly_chart(fig, use_container_width=True)

    table_cols = ["attack_id", "name", "category", "severity", "genai_capability", "priority", "discovered_on"]
    styled = filtered[table_cols].style.map(theme.tier_style, subset=["severity"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect an attack", filtered["attack_id"].tolist() if len(filtered) else [])
    if selected:
        row = filtered[filtered["attack_id"] == selected].iloc[0]
        with st.expander(f"Details — {row['name']}", expanded=True):
            st.write(row["description"])
            st.json({
                "attack_id": row["attack_id"],
                "category": row["category"],
                "severity": row["severity"],
                "genai_capability": row["genai_capability"],
                "feasibility": row["feasibility"],
                "impact_score": row["impact_score"],
            })

    # ── Live Threat Miner with Groq Backend ──────────────────────────
    with st.expander("⚡ Mine Threat with Groq API (/api/identify)", expanded=False):
        st.caption("Generate new synthetic threat models using your deployed Groq LLM on Render.")
        from web_app.api_client import identify_threat_api
        tm_col1, tm_col2, tm_col3 = st.columns(3)
        with tm_col1:
            tm_pattern = st.text_input("Fraud Pattern", value="synthetic_identity", key="tm_pattern")
        with tm_col2:
            tm_capability = st.text_input("GenAI Capability", value="deepfake", key="tm_capability")
        with tm_col3:
            tm_vuln = st.text_input("Payment Vulnerability", value="otp_bypass", key="tm_vuln")

        if st.button("Generate Threat Model with Groq", type="primary", key="btn_mine_groq"):
            with st.spinner("Querying Groq LLM on Render backend..."):
                mine_res = identify_threat_api(
                    fraud_pattern=tm_pattern,
                    genai_capability=tm_capability,
                    payment_vulnerability=tm_vuln,
                )
            if "error" in mine_res:
                st.error(f"Threat mining failed: {mine_res['error']}")
            else:
                st.session_state["last_mined_threat"] = mine_res
                st.success("New threat vector discovered by Groq backend!")

        if "last_mined_threat" in st.session_state:
            mt = st.session_state["last_mined_threat"]
            st.markdown(f"#### {mt.get('attack_name', 'Mined Threat')}")
            st.write(mt.get("description", ""))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Risk Score", mt.get("risk_score", "N/A"))
            mc2.metric("Severity", f"{mt.get('severity_score', 'N/A')}/10")
            mc3.metric("Feasibility", f"{mt.get('feasibility_score', 'N/A')}/10")
            mc4.metric("Novelty", f"{mt.get('novelty_score', 'N/A')}/10")

            if mt.get("attack_signals"):
                st.markdown("**Key Attack Signals Detected:**")
                for sig in mt["attack_signals"]:
                    st.markdown(f"- `{sig}`")

    return filtered



def render_generation_studio(attacks_df: pd.DataFrame):
    st.subheader("Generation Studio")
    st.caption("PILLAR 2 · TURN A DISCOVERED ATTACK INTO SYNTHETIC FRAUD DATA")

    if attacks_df.empty:
        st.info("No attacks match your current filters above — adjust filters to pick one here.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        attack_name = st.selectbox("Attack type", attacks_df["name"].tolist())
    with col2:
        volume = st.slider("Volume", 100, 5000, 500, step=100)
    with col3:
        sophistication = st.slider("Sophistication", 0.0, 1.0, 0.5, step=0.05)

    if st.button("Generate Synthetic Transactions", type="primary"):
        with st.spinner("Generating synthetic fraud data..."):
            data = generate_synthetic_transactions(attack_name, volume, sophistication)
        st.session_state["last_generated"] = data
        st.success(f"Generated {len(data):,} synthetic transactions for '{attack_name}'.")

    data = st.session_state.get("last_generated")
    if data is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(data, x="amount", nbins=40, title="Transaction Amount Distribution",
                                 color_discrete_sequence=[theme.BRAND])
            fig1.update_traces(marker_line_width=0, opacity=0.9)
            fig1.update_layout(height=360, transition=dict(duration=450, easing="cubic-in-out"))
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.histogram(data, x="hour_of_day", nbins=24, title="Timing Pattern · Hour of Day",
                                 color_discrete_sequence=[theme.ACCENT_TEAL])
            fig2.update_traces(marker_line_width=0, opacity=0.9)
            fig2.update_layout(height=300, transition=dict(duration=450, easing="cubic-in-out"))
            st.plotly_chart(fig2, use_container_width=True)

        geo_counts = data["geo"].value_counts().reset_index()
        geo_counts.columns = ["geo", "count"]
        geo_counts = geo_counts.sort_values("count", ascending=True)
        fig3 = px.bar(geo_counts, x="count", y="geo", orientation="h", title="Geographic Distribution",
                       color_discrete_sequence=[theme.SIGNATURE])
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(
            yaxis_title=None,
            height=480,
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(data.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download as CSV", data.to_csv(index=False),
            file_name=f"synthetic_{attack_name.replace(' ', '_')}.csv", mime="text/csv",
        )
