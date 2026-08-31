"""
api.py — PhantomGuard DEFEND REST API
======================================
A FastAPI backend that serves the trained DEFEND ensemble as HTTP endpoints.
Deploy this separately (Render, Railway, AWS, etc.) and point the frontend
at its URL.

Endpoints:
    GET  /health                → {"status": "ok", "models_loaded": true}
    POST /api/score             → score a single transaction
    POST /api/score-batch       → score multiple transactions
    GET  /api/live-feed?n=40    → get a scored feed of n random transactions
    GET  /api/metrics           → training metrics summary

Run locally:
    pip install fastapi uvicorn
    python defend/api.py
    # or: uvicorn defend.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Make sibling modules importable ────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluator import get_live_scored_feed, score_transaction, _generate_row, _ModelBundle, PROFILES  # noqa: E402

# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="PhantomGuard DEFEND API",
    description="Real-time fraud scoring powered by XGBoost + TCN + GNN + Meta-classifier ensemble",
    version="1.0.0",
)

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────
class TransactionInput(BaseModel):
    """A single transaction to score. All fields match the training feature set."""
    amount: float = 100.0
    currency: str = "USD"
    channel: str = "mobile_app"
    payment_method: str = "credit_card"
    merchant_category_code: int = 5411
    merchant_category_desc: str = "Grocery Stores"
    merchant_risk_score: float = 0.15
    device_type: str = "android_mobile"
    device_fingerprint_score: float = 0.85
    ip_country: str = "US"
    billing_country: str = "US"
    is_cross_border: bool = False
    hour_of_day: int = 14
    is_night_time: bool = False
    time_since_last_txn_sec: float = 3600.0
    txn_velocity_1h: int = 0
    txn_velocity_24h: int = 1
    amount_vs_user_avg_ratio: float = 1.2
    account_age_days: int = 365
    account_creation_channel: str = "web_remote"
    kyc_verification_level: str = "basic_document"
    credit_score_band: str = "prime"
    historical_avg_amount: float = 80.0
    historical_txn_count: int = 200
    num_devices_used: int = 1
    num_linked_accounts: int = 0
    behavioral_score: float = 0.85
    login_anomaly_score: float = 0.05
    password_reset_recent: bool = False
    mfa_enabled: bool = True
    prior_fraud_flags: int = 0
    shared_device_n_accounts: int = 1
    shared_ip_n_accounts: int = 1
    num_beneficiaries_30d: int = 1
    beneficiary_account_age_days: float = 200.0
    text_similarity_to_phishing_corpus: float | None = None
    llm_generated_content_prob: float | None = None
    voice_authenticity_score: float | None = None
    deepfake_video_score: float | None = None
    document_authenticity_score: float | None = None
    image_manipulation_score: float | None = None
    refund_count_30d: int = 0
    refund_to_purchase_ratio: float | None = None
    structuring_score: float = 0.05


class ScoreResponse(BaseModel):
    xgboost: float
    tcn: float
    gnn: float
    meta: float
    verdict: str
    risk_tier: str


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionInput]


# ── Startup: warm-load models ─────────────────────────────────────────
@app.on_event("startup")
def load_models():
    _ModelBundle.get()


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _ModelBundle._instance is not None,
        "model": "PhantomGuard DEFEND v1 (XGBoost + TCN + GNN + Meta)",
    }


@app.post("/api/score", response_model=ScoreResponse)
def score_single(txn: TransactionInput):
    """Score a single transaction through the full four-model ensemble."""
    import numpy as np

    row = txn.model_dump()
    # Convert None → NaN for the model
    for key, val in row.items():
        if val is None:
            row[key] = np.nan

    m = _ModelBundle.get()
    # Only pass feature columns the model expects
    filtered = {k: row[k] for k in m.feature_columns if k in row}

    scores = score_transaction(filtered)
    meta_p = scores["meta"]

    if meta_p >= 0.85:
        verdict, tier = "BLOCKED", "Critical"
    elif meta_p >= 0.60:
        verdict, tier = "REVIEW", "High"
    elif meta_p >= 0.30:
        verdict, tier = "MONITOR", "Medium"
    else:
        verdict, tier = "APPROVED", "Low"

    return ScoreResponse(
        xgboost=round(scores["xgboost"], 4),
        tcn=round(scores["tcn"], 4),
        gnn=round(scores["gnn"], 4),
        meta=round(meta_p, 4),
        verdict=verdict,
        risk_tier=tier,
    )


@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    """Score multiple transactions at once."""
    results = []
    for txn in req.transactions:
        result = score_single(txn)
        results.append(result.model_dump())
    return {"results": results, "count": len(results)}


@app.get("/api/live-feed")
def live_feed(n: int = Query(default=40, ge=1, le=200)):
    """
    Generate and score *n* random transactions. Returns the same shape
    the Streamlit frontend expects — useful for demo / testing.
    """
    df = get_live_scored_feed(n)
    return {"transactions": df.to_dict(orient="records"), "count": len(df)}


@app.get("/api/metrics")
def metrics_summary():
    """Return the training metrics from the last model run."""
    metrics_path = _PROJECT_ROOT / "artifacts" / "defend_full_tcn" / "metrics.json"
    if not metrics_path.exists():
        return {"error": "metrics.json not found"}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


# ── Run directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
