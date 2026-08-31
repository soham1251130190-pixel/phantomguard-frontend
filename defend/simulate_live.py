"""
PhantomGuard DEFEND — Live Transaction Simulation
===================================================
Loads the trained four-model architecture and scores randomly generated
transactions in real time, printing colour-coded results to the terminal.

Usage:
    python defend/simulate_live.py                          # default 20 txns
    python defend/simulate_live.py --count 50 --delay 0.3   # 50 txns, faster
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Adjust sys.path so common / model imports work regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_neighbor_means, build_sequences  # noqa: E402
from gnn_model import GraphSAGEClassifier  # noqa: E402
from meta_classifier import stack_probabilities  # noqa: E402
from tcn_model import TCNClassifier  # noqa: E402

# ---------------------------------------------------------------------------
# Fake‑transaction generator
# ---------------------------------------------------------------------------

CHANNELS = ["card_present", "mobile_app", "card_not_present_online", "p2p_transfer", "atm_withdrawal"]
PAYMENT_METHODS = ["credit_card", "debit_card", "upi", "digital_wallet", "bank_transfer", "prepaid_card"]
DEVICE_TYPES = ["android_mobile", "ios_mobile", "windows_desktop", "mac_desktop", "linux_desktop"]
CURRENCIES = ["USD"]
COUNTRIES = ["US", "GB", "IN", "JP", "DE", "BR", "SG", "NL", "AU", "CA", "FR", "VN", "PK", "BD"]
ACCOUNT_CHANNELS = ["web_remote", "in_branch", "agent_assisted", "api_partner_onboarding"]
KYC_LEVELS = ["unverified", "basic_document", "enhanced_verification", "video_kyc"]
CREDIT_BANDS = ["thin_file", "sub_prime", "near_prime", "prime", "super_prime"]
MERCHANT_DESCS = [
    "Grocery Stores", "Department Stores", "Electronics Stores", "Restaurants",
    "Gas Stations", "Insurance", "Liquor Stores", "Hotels/Motels",
    "Financial Institutions - Merchandise/Services", "Drug Stores",
]

ATTACK_PROFILES: dict[str, dict] = {
    "legitimate": {
        "weight": 0.70,
        "amount_range": (5, 500),
        "velocity_1h": (0, 1),
        "velocity_24h": (0, 3),
        "avg_ratio": (0.5, 2.0),
        "device_fp": (0.70, 0.99),
        "behavioral": (0.60, 0.99),
        "login_anomaly": (0.0, 0.15),
        "cross_border": 0.05,
        "password_reset": 0.02,
        "account_age": (90, 3000),
        "text_sim": None,
        "llm_prob": None,
    },
    "ai_phishing": {
        "weight": 0.06,
        "amount_range": (50, 2000),
        "velocity_1h": (1, 5),
        "velocity_24h": (1, 6),
        "avg_ratio": (3.0, 20.0),
        "device_fp": (0.02, 0.30),
        "behavioral": (0.30, 0.65),
        "login_anomaly": (0.50, 0.95),
        "cross_border": 0.60,
        "password_reset": 0.45,
        "account_age": (10, 800),
        "text_sim": (0.70, 0.97),
        "llm_prob": (0.70, 0.99),
    },
    "account_takeover": {
        "weight": 0.06,
        "amount_range": (100, 5000),
        "velocity_1h": (2, 8),
        "velocity_24h": (3, 8),
        "avg_ratio": (5.0, 50.0),
        "device_fp": (0.01, 0.15),
        "behavioral": (0.10, 0.50),
        "login_anomaly": (0.60, 0.98),
        "cross_border": 0.70,
        "password_reset": 0.80,
        "account_age": (200, 2500),
        "text_sim": None,
        "llm_prob": None,
    },
    "synthetic_identity_fraud": {
        "weight": 0.05,
        "amount_range": (200, 8000),
        "velocity_1h": (0, 2),
        "velocity_24h": (0, 3),
        "avg_ratio": (1.0, 5.0),
        "device_fp": (0.40, 0.80),
        "behavioral": (0.50, 0.80),
        "login_anomaly": (0.05, 0.30),
        "cross_border": 0.15,
        "password_reset": 0.05,
        "account_age": (0, 90),
        "text_sim": None,
        "llm_prob": None,
    },
    "voice_cloning_fraud": {
        "weight": 0.04,
        "amount_range": (500, 15000),
        "velocity_1h": (0, 2),
        "velocity_24h": (1, 4),
        "avg_ratio": (8.0, 80.0),
        "device_fp": (0.05, 0.40),
        "behavioral": (0.20, 0.55),
        "login_anomaly": (0.40, 0.85),
        "cross_border": 0.50,
        "password_reset": 0.35,
        "account_age": (100, 1500),
        "text_sim": None,
        "llm_prob": None,
    },
    "deepfake_identity_fraud": {
        "weight": 0.04,
        "amount_range": (300, 10000),
        "velocity_1h": (0, 1),
        "velocity_24h": (0, 3),
        "avg_ratio": (2.0, 15.0),
        "device_fp": (0.10, 0.50),
        "behavioral": (0.30, 0.65),
        "login_anomaly": (0.20, 0.60),
        "cross_border": 0.30,
        "password_reset": 0.10,
        "account_age": (0, 60),
        "text_sim": None,
        "llm_prob": None,
    },
    "merchant_fraud": {
        "weight": 0.05,
        "amount_range": (10, 3000),
        "velocity_1h": (0, 3),
        "velocity_24h": (2, 8),
        "avg_ratio": (1.0, 8.0),
        "device_fp": (0.50, 0.90),
        "behavioral": (0.60, 0.90),
        "login_anomaly": (0.0, 0.10),
        "cross_border": 0.10,
        "password_reset": 0.01,
        "account_age": (30, 1000),
        "text_sim": None,
        "llm_prob": None,
    },
}


def _rand(low: float, high: float) -> float:
    return round(random.uniform(low, high), 3)


def generate_transaction(profile_name: str, profile: dict, feature_columns: list[str]) -> dict:
    """Return a single synthetic transaction row as a dict keyed by *feature_columns*."""
    hour = random.randint(0, 23)
    ip_country = random.choice(COUNTRIES)
    billing = random.choice(COUNTRIES)
    is_cross = random.random() < profile["cross_border"]
    if is_cross:
        while ip_country == billing:
            ip_country = random.choice(COUNTRIES)
    else:
        ip_country = billing

    hist_avg = round(random.uniform(15, 200), 2)
    amount = round(random.uniform(*profile["amount_range"]), 2)

    row: dict = {
        "amount": amount,
        "currency": "USD",
        "channel": random.choice(CHANNELS),
        "payment_method": random.choice(PAYMENT_METHODS),
        "merchant_category_code": random.choice([4111, 5311, 5411, 5732, 5921, 6012, 6300, 7011, 8299]),
        "merchant_category_desc": random.choice(MERCHANT_DESCS),
        "merchant_risk_score": _rand(0.01, 0.98) if profile_name != "legitimate" else _rand(0.01, 0.40),
        "device_type": random.choice(DEVICE_TYPES),
        "device_fingerprint_score": _rand(*profile["device_fp"]),
        "ip_country": ip_country,
        "billing_country": billing,
        "is_cross_border": is_cross,
        "hour_of_day": hour,
        "is_night_time": hour <= 5,
        "time_since_last_txn_sec": round(random.uniform(10, 200000), 2),
        "txn_velocity_1h": random.randint(*profile["velocity_1h"]),
        "txn_velocity_24h": random.randint(*profile["velocity_24h"]),
        "amount_vs_user_avg_ratio": _rand(*profile["avg_ratio"]),
        "account_age_days": random.randint(*profile["account_age"]),
        "account_creation_channel": random.choice(ACCOUNT_CHANNELS),
        "kyc_verification_level": random.choice(KYC_LEVELS),
        "credit_score_band": random.choice(CREDIT_BANDS),
        "historical_avg_amount": hist_avg,
        "historical_txn_count": random.randint(0, 1200),
        "num_devices_used": random.randint(1, 3),
        "num_linked_accounts": random.randint(0, 4),
        "behavioral_score": _rand(*profile["behavioral"]),
        "login_anomaly_score": _rand(*profile["login_anomaly"]),
        "password_reset_recent": random.random() < profile["password_reset"],
        "mfa_enabled": random.choice([True, False]),
        "prior_fraud_flags": random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05])[0],
        "shared_device_n_accounts": random.randint(1, 5),
        "shared_ip_n_accounts": random.randint(1, 5),
        "num_beneficiaries_30d": random.randint(0, 6),
        "beneficiary_account_age_days": round(random.uniform(0, 900), 2),
        "text_similarity_to_phishing_corpus": _rand(*profile["text_sim"]) if profile.get("text_sim") else np.nan,
        "llm_generated_content_prob": _rand(*profile["llm_prob"]) if profile.get("llm_prob") else np.nan,
        "voice_authenticity_score": np.nan,
        "deepfake_video_score": np.nan,
        "document_authenticity_score": np.nan,
        "image_manipulation_score": np.nan,
        "refund_count_30d": random.randint(0, 3),
        "refund_to_purchase_ratio": np.nan,
        "structuring_score": _rand(0.0, 0.20) if profile_name == "legitimate" else _rand(0.0, 0.95),
    }
    # Only return columns the model expects
    return {col: row.get(col, np.nan) for col in feature_columns}


# ---------------------------------------------------------------------------
# Rich terminal helpers
# ---------------------------------------------------------------------------

def risk_colour(score: float) -> str:
    if score >= 0.85:
        return "bold red"
    if score >= 0.50:
        return "bold yellow"
    return "bold green"


def verdict_text(score: float) -> Text:
    if score >= 0.85:
        return Text("🚨 BLOCKED", style="bold white on red")
    if score >= 0.50:
        return Text("⚠️  REVIEW", style="bold black on yellow")
    return Text("✅ APPROVED", style="bold white on green")


def build_header(console_width: int) -> Panel:
    title = Text()
    title.append("⛨ ", style="bold cyan")
    title.append("PHANTOM", style="bold bright_white")
    title.append("GUARD", style="bold cyan")
    title.append("  DEFEND", style="bold bright_magenta")
    title.append("  ·  Live Transaction Monitor", style="dim white")
    return Panel(
        Align.center(title),
        box=box.DOUBLE_EDGE,
        style="bright_cyan",
        padding=(0, 2),
    )


def build_stats_panel(stats: dict) -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column(style="bold")
    tbl.add_row("Processed", str(stats["total"]))
    tbl.add_row("Approved", Text(str(stats["approved"]), style="green"))
    tbl.add_row("Reviewed", Text(str(stats["review"]), style="yellow"))
    tbl.add_row("Blocked", Text(str(stats["blocked"]), style="red"))
    tbl.add_row("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
    return Panel(tbl, title="[bold]Session Stats[/bold]", border_style="bright_cyan", box=box.ROUNDED)


def build_txn_table(history: list[dict], max_rows: int = 12) -> Table:
    tbl = Table(
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        pad_edge=True,
        title="[bold bright_white]Recent Transactions[/bold bright_white]",
        title_style="",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
    )
    tbl.add_column("#", style="dim", width=4)
    tbl.add_column("Time", width=10)
    tbl.add_column("Amount", justify="right", width=10)
    tbl.add_column("Channel", width=16)
    tbl.add_column("Country", width=8, justify="center")
    tbl.add_column("XGB", justify="right", width=6)
    tbl.add_column("TCN", justify="right", width=6)
    tbl.add_column("GNN", justify="right", width=6)
    tbl.add_column("Meta", justify="right", width=6)
    tbl.add_column("Verdict", width=14, justify="center")
    tbl.add_column("Ground Truth", width=14, justify="center")

    for rec in history[-max_rows:]:
        meta_score = rec["meta"]
        tbl.add_row(
            str(rec["idx"]),
            rec["time"],
            f"${rec['amount']:,.2f}",
            rec["channel"],
            rec["country"],
            Text(f"{rec['xgb']:.2f}", style=risk_colour(rec["xgb"])),
            Text(f"{rec['tcn']:.2f}", style=risk_colour(rec["tcn"])),
            Text(f"{rec['gnn']:.2f}", style=risk_colour(rec["gnn"])),
            Text(f"{rec['meta']:.2f}", style=risk_colour(meta_score)),
            verdict_text(meta_score),
            Text(rec["truth"], style="red" if rec["truth"] != "legitimate" else "green"),
        )
    return tbl


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def load_models(artifact_path: str):
    bundle = joblib.load(artifact_path)
    preprocessor = bundle["preprocessor"]
    xgb_model = bundle["xgboost"]
    feature_columns = bundle["feature_columns"]
    seq_len = bundle["sequence_length"]
    max_neighbors = bundle["max_neighbors"]

    import torch
    tcn_model = TCNClassifier(bundle["tcn_feature_count"])
    tcn_model.load_state_dict(bundle["tcn_state_dict"])
    tcn_model.eval()

    gnn_model = GraphSAGEClassifier(bundle["gnn_feature_count"])
    gnn_model.load_state_dict(bundle["gnn_state_dict"])
    gnn_model.eval()

    meta_model = bundle["meta_classifier"]
    return preprocessor, xgb_model, tcn_model, gnn_model, meta_model, feature_columns, seq_len, max_neighbors


def score_single(row_dict, preprocessor, xgb_model, tcn_model, gnn_model, meta_model, feature_columns, seq_len, max_neighbors):
    """Score one transaction through all four models."""
    import torch

    df = pd.DataFrame([row_dict])
    # Need a dummy timestamp for sequence builder
    df["timestamp"] = datetime.now(timezone.utc).isoformat()
    df["user_id"] = "SIM_USER"

    x_raw = df[feature_columns]
    x = preprocessor.transform(x_raw).astype(np.float32)

    # XGBoost
    xgb_prob = float(xgb_model.predict_proba(x)[:, 1][0])

    # TCN
    seq = np.zeros((1, x.shape[1], seq_len), dtype=np.float32)
    seq[:, :, -1] = x
    with torch.no_grad():
        tcn_logit = tcn_model(torch.tensor(seq, dtype=torch.float32))
        tcn_prob = float(torch.sigmoid(tcn_logit).item())

    # GNN (self = neighbor for single txn)
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        gnn_logit = gnn_model(x_t, x_t)  # no real neighbors in simulation
        gnn_prob = float(torch.sigmoid(gnn_logit).item())

    # Meta-classifier
    base = stack_probabilities(
        np.array([xgb_prob]),
        np.array([tcn_prob]),
        np.array([gnn_prob]),
    )
    meta_prob = float(meta_model.predict_proba(base)[:, 1][0])
    return xgb_prob, tcn_prob, gnn_prob, meta_prob


def main():
    parser = argparse.ArgumentParser(description="PhantomGuard DEFEND live simulation")
    parser.add_argument("--artifact", default="artifacts/defend_full_tcn/phantomguard_full_architecture.joblib")
    parser.add_argument("--count", type=int, default=20, help="Number of transactions to simulate")
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds between transactions")
    args = parser.parse_args()

    console = Console()

    # ── Load models ──────────────────────────────────────────────────
    with console.status("[bold cyan]Loading PhantomGuard DEFEND models…[/bold cyan]", spinner="dots"):
        preprocessor, xgb_model, tcn_model, gnn_model, meta_model, feature_columns, seq_len, max_neighbors = (
            load_models(args.artifact)
        )
    console.print("[bold green]✓[/bold green] Models loaded successfully.\n")
    time.sleep(0.5)

    # ── Weighted attack selection ────────────────────────────────────
    attack_names = list(ATTACK_PROFILES.keys())
    attack_weights = [ATTACK_PROFILES[n]["weight"] for n in attack_names]

    history: list[dict] = []
    stats = {"total": 0, "approved": 0, "review": 0, "blocked": 0, "avg_latency_ms": 0.0}
    total_latency = 0.0

    with Live(console=console, refresh_per_second=8, screen=True) as live:
        for i in range(1, args.count + 1):
            # Pick attack profile
            profile_name = random.choices(attack_names, weights=attack_weights, k=1)[0]
            profile = ATTACK_PROFILES[profile_name]
            row = generate_transaction(profile_name, profile, feature_columns)

            t0 = time.perf_counter()
            xgb_p, tcn_p, gnn_p, meta_p = score_single(
                row, preprocessor, xgb_model, tcn_model, gnn_model, meta_model, feature_columns, seq_len, max_neighbors,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            total_latency += latency_ms

            # Update stats
            stats["total"] = i
            stats["avg_latency_ms"] = total_latency / i
            if meta_p >= 0.85:
                stats["blocked"] += 1
            elif meta_p >= 0.50:
                stats["review"] += 1
            else:
                stats["approved"] += 1

            now = datetime.now()
            history.append({
                "idx": i,
                "time": now.strftime("%H:%M:%S"),
                "amount": row.get("amount", 0),
                "channel": row.get("channel", "?"),
                "country": f"{row.get('billing_country', '?')}→{row.get('ip_country', '?')}",
                "xgb": xgb_p,
                "tcn": tcn_p,
                "gnn": gnn_p,
                "meta": meta_p,
                "truth": profile_name,
            })

            # Build layout
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3),
            )
            layout["header"].update(build_header(console.width))
            layout["body"].split_row(
                Layout(build_txn_table(history), ratio=4),
                Layout(build_stats_panel(stats), ratio=1),
            )
            progress_pct = i / args.count
            bar_filled = int(progress_pct * 40)
            bar = f"  [bright_cyan]{'━' * bar_filled}[/bright_cyan][dim]{'╌' * (40 - bar_filled)}[/dim]  {i}/{args.count}"
            layout["footer"].update(Panel(bar, box=box.ROUNDED, border_style="dim"))

            live.update(layout)
            time.sleep(args.delay)

    # ── Final summary ────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold]Simulation complete.[/bold]\n"
        f"  Transactions: {stats['total']}\n"
        f"  [green]Approved: {stats['approved']}[/green]  "
        f"[yellow]Review: {stats['review']}[/yellow]  "
        f"[red]Blocked: {stats['blocked']}[/red]\n"
        f"  Avg scoring latency: {stats['avg_latency_ms']:.1f} ms",
        title="[bold bright_cyan]⛨ Session Summary[/bold bright_cyan]",
        border_style="bright_cyan",
        box=box.DOUBLE_EDGE,
    ))


if __name__ == "__main__":
    main()
