"""
api_client.py
-------------
Helper utilities for communicating with the deployed PhantomGuard DEFEND REST API (e.g. on Render).
"""

from __future__ import annotations

import os
import requests
import streamlit as st

DEFAULT_TIMEOUT = 50  # Render free tier cold-start allowance
DEFAULT_BACKEND_URL = "https://phantomguard-1.onrender.com"


def get_backend_url() -> str:
    """
    Resolve the backend API URL in priority order:
      1. Session state override (configured directly in the UI sidebar)
      2. Streamlit secrets (.streamlit/secrets.toml or Streamlit Cloud)
      3. Environment variables (DEFEND_API_URL or BACKEND_URL)
      4. Default deployed Render URL
    """
    # 1. UI override in session state
    if st.session_state.get("custom_backend_url"):
        val = str(st.session_state["custom_backend_url"]).strip()
        if val:
            return val.rstrip("/")

    # 2. Streamlit secrets (.streamlit/secrets.toml)
    try:
        if "DEFEND_API_URL" in st.secrets:
            val = str(st.secrets["DEFEND_API_URL"]).strip()
            if val:
                return val.rstrip("/")
        if "BACKEND_URL" in st.secrets:
            val = str(st.secrets["BACKEND_URL"]).strip()
            if val:
                return val.rstrip("/")
    except Exception:
        pass

    # 3. Environment variables
    env_url = os.environ.get("DEFEND_API_URL") or os.environ.get("BACKEND_URL")
    if env_url:
        return env_url.strip().rstrip("/")

    # 4. Default deployed backend
    return DEFAULT_BACKEND_URL



def check_backend_health(url: str | None = None) -> dict:
    """
    Ping health endpoints of the backend (/api/health, /health, /).
    Returns: {"ok": bool, "status_code": int, "message": str, "data": dict, "url": str, "endpoint": str}
    """
    target = (url or get_backend_url()).strip().rstrip("/")
    if not target:
        return {"ok": False, "status_code": 0, "message": "No backend URL configured", "data": {}, "url": "", "endpoint": ""}

    if not target.startswith("http://") and not target.startswith("https://"):
        target = f"https://{target}"

    endpoints = ["/api/health", "/health", "/"]
    last_err = None

    for ep in endpoints:
        try:
            resp = requests.get(f"{target}{ep}", timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": resp.text}
                return {
                    "ok": True,
                    "status_code": 200,
                    "message": "Connected",
                    "data": data,
                    "url": target,
                    "endpoint": ep,
                }
        except requests.exceptions.Timeout:
            last_err = "Request timed out — Render service may be spinning up (allow ~30-50s)"
        except requests.exceptions.ConnectionError:
            last_err = "Could not connect to backend host (check URL)"
        except Exception as e:
            last_err = str(e)

    return {
        "ok": False,
        "status_code": 0,
        "message": last_err or "Backend health endpoints returned non-200",
        "data": {},
        "url": target,
        "endpoint": "",
    }


def get_backend_stats(url: str | None = None) -> dict:
    """Fetch stats from /api/stats if available."""
    target = (url or get_backend_url()).strip().rstrip("/")
    if not target:
        return {}
    if not target.startswith("http://") and not target.startswith("https://"):
        target = f"https://{target}"
    try:
        resp = requests.get(f"{target}/api/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def get_backend_history(url: str | None = None) -> list[dict]:
    """Fetch scan history from /api/history if available."""
    target = (url or get_backend_url()).strip().rstrip("/")
    if not target:
        return []
    if not target.startswith("http://") and not target.startswith("https://"):
        target = f"https://{target}"
    try:
        resp = requests.get(f"{target}/api/history", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def scan_text_api(text: str, category: str = "user", evolve: bool = False, url: str | None = None) -> dict:
    """Call /api/scan on the deployed Render backend."""
    target = (url or get_backend_url()).strip().rstrip("/")
    if not target:
        return {"error": "No backend URL configured"}
    if not target.startswith("http://") and not target.startswith("https://"):
        target = f"https://{target}"
    try:
        resp = requests.post(
            f"{target}/api/scan",
            json={"text": text, "category": category, "evolve": evolve},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def render_backend_sidebar():
    """
    Renders a dedicated Backend Connection widget in the Streamlit sidebar.
    Allows testing, viewing, and live-editing the Render backend URL.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌐 Backend Connection (Render)")

    current_configured = get_backend_url()
    input_val = st.sidebar.text_input(
        "Render Backend URL",
        value=current_configured,
        placeholder="https://your-app.onrender.com",
        help="Paste the URL of your deployed Render FastAPI backend here, or configure it in .streamlit/secrets.toml",
    )

    if input_val != current_configured:
        st.session_state["custom_backend_url"] = input_val.strip()
        st.session_state.pop("backend_health_status", None)
        st.rerun()

    active_url = get_backend_url()

    col_btn, col_status = st.sidebar.columns([1, 1])
    with col_btn:
        test_clicked = st.button("Test Health", use_container_width=True)

    if test_clicked or "backend_health_status" in st.session_state:
        if test_clicked:
            with st.spinner("Pinging Render backend..."):
                st.session_state["backend_health_status"] = check_backend_health(active_url)

        res = st.session_state.get("backend_health_status", {})
        if res.get("ok"):
            d = res.get("data", {})
            label = d.get("model") or d.get("message") or d.get("status") or "Connected"
            st.sidebar.success(f"🟢 **Online**: {label}")
            stats = get_backend_stats(active_url)
            if stats and "total" in stats:
                st.sidebar.caption(f"Scans recorded: `{stats['total']}` | Failed: `{stats.get('failed_attacks', 0)}`")
        elif active_url:
            st.sidebar.warning(f"⚠️ {res.get('message', 'Unreachable')}")
        else:
            st.sidebar.info("💡 Running in Demo Mode (mock data)")
    elif active_url:
        st.sidebar.caption(f"Configured: `{active_url}`")
    else:
        st.sidebar.caption("No backend URL set. Using mock fallback.")


