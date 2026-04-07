import os
import csv
import json
import base64
import requests
import schedule
import time
import asyncio
import pandas as pd
import yfinance as yf
import numpy as np
import anthropic
from datetime import datetime, date, timezone, timedelta
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

# ─────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
UW_TOKEN         = os.getenv("UW_TOKEN")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
CHAT_ID          = int(os.getenv("CHAT_ID", "0"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO      = os.getenv("GITHUB_REPO", "coding101010rizz/trading-bots")
TICKER           = "SPY"

# ─────────────────────────────────────────────
# MULTI-TICKER CONFIG
# Option C architecture: secondary tickers only
# suggested when SPY grade is D or F.
# Bot ranks all tickers and picks the best one.
#
# IV_TICKER: volatility index per ticker
#   SPY  → ^VIX  (CBOE VIX)
#   QQQ  → ^VXN  (Nasdaq VIX)
#   TSLA → None  (use IV rank from UW API)
#   NVDA → None  (use IV rank from UW API)
#   GOOGL→ None  (use IV rank from UW API)
# ─────────────────────────────────────────────
SECONDARY_TICKERS = {
    "QQQ":   {"iv_ticker": "^VXN",  "name": "QQQ",   "vol_history_key": "qqq_vol_gex_history"},
    "TSLA":  {"iv_ticker": None,    "name": "TSLA",  "vol_history_key": "tsla_vol_gex_history"},
    "NVDA":  {"iv_ticker": None,    "name": "NVDA",  "vol_history_key": "nvda_vol_gex_history"},
    "GOOGL": {"iv_ticker": None,    "name": "GOOGL", "vol_history_key": "googl_vol_gex_history"},
}
SPY_WEAK_GRADES = ("D", "F")   # grades that trigger multi-ticker check
MAX_DAY_TRADES  = 3             # PDT limit — adjust if account > $25k

# Anthropic client — AI verification + alert writing
anthropic_client = (
    anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if ANTHROPIC_API_KEY else None
)

# ─────────────────────────────────────────────
# TIMEZONE — all times PDT (UTC-7)
# ─────────────────────────────────────────────
PDT = timezone(timedelta(hours=-7))

def now_pdt():
    """Always returns current time in PDT regardless of server timezone."""
    return datetime.now(timezone.utc).astimezone(PDT)

def is_market_open():
    pdt = now_pdt()
    if pdt.weekday() >= 5:
        return False
    o = pdt.replace(hour=6, minute=30, second=0, microsecond=0)
    c = pdt.replace(hour=13, minute=0, second=0, microsecond=0)
    return o <= pdt <= c

def is_overnight_window():
    """
    True during genuine overnight hours only.
    Skips 1pm-6pm PDT to avoid post-EOD spam.
    Active: weekday evenings 6pm-6am, all weekend.
    """
    pdt = now_pdt()
    h = pdt.hour
    weekday = pdt.weekday()
    if weekday >= 5:
        return True
    return h >= 18 or h < 6

# ─────────────────────────────────────────────
# LOGGING SYSTEM v3 — Full ML Dataset
# ─────────────────────────────────────────────
LOG_FILE = "spy_gex_log.csv"
LOG_HEADERS = [
    "date", "time", "price",
    "oi_gex_raw", "vol_gex_raw",
    "oi_gex_m", "vol_gex_m",
    "ratio", "gex_state", "regime",
    "conviction_score", "grade",
    "vix", "vvix", "vix_term",
    "tick_approx", "inventory_bias",
    "unwind_score", "open_drive",
    "vanna_target", "charm_target",
    "calendar_flags", "days_to_opex",
    "opex_cycle_phase",          # NEW: early/mid/late/opex_week/opex_day
    "vwap_distance", "price_vs_open", "session_range",
    "vol_gex_velocity", "vol_gex_direction",
    "regime_transitions", "vwap_breaks",
    "gamma_wall_above", "gamma_wall_below", "time_of_day",
    "news_sentiment", "news_score",
    "catalyst_type", "catalyst_strength", "macro_override",
    "session_type",              # MARKET / OVERNIGHT / WEEKEND
    "futures_direction",         # UP / DOWN / FLAT
    "overnight_vix_move",        # VIX change since close
    "overnight_news_flag",       # MAJOR_EVENT / MINOR / NONE
    "gap_direction",             # UP / DOWN / NONE
    "gap_size",                  # points SPY gapped at open
    "gap_type",                  # DIRECTIONAL/FADE_THEN_STATIC/FULL_FADE/GAP_AND_REVERSE/STATIC/UNKNOWN
    "gap_conviction",            # 0-100 confidence in gap_type classification
    "vol_gex_velocity_alert",    # YES if velocity crossed threshold
    "outcome_direction", "outcome_points",
    "signal_correct", "max_move_up", "max_move_down",
    "claude_verdict", "claude_confidence",
    "claude_reasoning", "combined_score",
    # Open candle analysis — populated on 6:35am row only, blank all others
    "open_candle_type",        # STRONG_BEAR/STRONG_BULL/REJECTION_WICK_TOP/REJECTION_WICK_BOTTOM/MODERATE_BEAR/MODERATE_BULL/DOJI/UNCLEAR
    "open_candle_body_pct",    # body as % of full candle range
    "open_candle_upper_wick",  # upper wick as % of range
    "open_candle_lower_wick",  # lower wick as % of range
    "open_candle_vol_ratio",   # volume vs 10-candle average (e.g. 2.3 = 2.3x)
    "open_candle_vwap_pos",    # ABOVE / BELOW / AT
    "open_candle_confluence",  # CONFIRMED / CONFLICT / WAIT / NEUTRAL
    "notes"
]

# ─────────────────────────────────────────────
# ECONOMIC CALENDARS
# ─────────────────────────────────────────────
FED_DATES_2026 = [
    date(2026, 1, 29), date(2026, 3, 19),
    date(2026, 5, 7),  date(2026, 6, 18),
    date(2026, 7, 30), date(2026, 9, 17),
    date(2026, 11, 5), date(2026, 12, 16),
]

OPEX_DATES = [
    date(2026, 1, 16), date(2026, 2, 20), date(2026, 3, 21),
    date(2026, 4, 17), date(2026, 5, 15), date(2026, 6, 20),
    date(2026, 7, 17), date(2026, 8, 21), date(2026, 9, 18),
    date(2026, 10, 16), date(2026, 11, 20), date(2026, 12, 18),
]

QUARTER_END_DATES = [
    date(2026, 3, 31), date(2026, 6, 30),
    date(2026, 9, 30), date(2026, 12, 31),
]

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
state = {
    # Core GEX state
    "previous_gex_state": None,
    "previous_ratio": None,
    "vwap_alert_sent": False,
    # Intelligence state
    "previous_vol_gex": None,
    "previous_oi_gex": None,
    "vol_gex_history": [],
    "regime": None,              # FIX: only defined once now
    "previous_regime": None,
    "velocity_score_sent": False,
    "consolidation_alert_sent": False,
    "hedge_unwind_alert_sent": False,
    "open_price": None,
    "open_iv": None,
    "open_volume": None,
    "open_time_prices": [],
    "last_conviction_score": None,
    # Precision tracking
    "tick_history": [],
    "vix_history": [],
    "inventory_bias": "NEUTRAL",
    "open_drive_detected": False,
    "session_high": None,
    "session_low": None,
    # Alert timing
    "last_unwind_alert_time": 0,
    "last_summary_time": 0,
    "consolidation_gex_state": None,
    # New modules
    "last_heartbeat": 0,
    "doji_transition_sent": False,
    "last_wall_alert_price": 0,
    "current_vanna_target": None,
    "current_charm_target": None,
    "regime_transitions_today": 0,
    "vwap_breaks_today": 0,
    "session_vwap": None,
    "eod_fired_today": False,
    # News cache — updated inside log_reading, available for AI verification
    "last_news_sentiment": "NEUTRAL",
    "last_catalyst_type": "NONE",
    "last_macro_override": "NO",
    # Telegram dedup — FIX: prevents re-processing old updates (stops spam loop)
    "telegram_last_update_id": 0,
    # GitHub persistence
    "github_csv_sha": "",
    "last_git_push": 0,
    # Overnight monitoring
    "overnight_report_sent": False,
    "last_overnight_check": 0,
    "overnight_gex_snapshot": None,
    "overnight_vix_close": None,
    "overnight_alerts_today": 0,
    # Gap tracking
    "prev_session_close": None,
    "gap_direction": "NONE",
    "gap_size": 0,
    "gap_type": "UNKNOWN",
    "gap_conviction": 0,
    "gap_type_sent": False,       # prevent re-sending gap classification alert
    # Dedup gate for log_reading
    "last_logged_row": {},
    # Vol GEX snapshots for gap classification
    "open_vol_gex_snapshot": None,   # Vol GEX at market open (6:30am)
    "overnight_vol_gex_close": None, # Vol GEX at yesterday's close (set at EOD)
    # Vol GEX velocity alert
    "vol_gex_velocity_alert_sent": False,
    "last_vol_gex_velocity": 0,
    # Gap fill alert
    "gap_fill_alert_sent": False,
    # Open candle analysis
    "open_candle_analyzed":    False,
    "open_candle_type":        None,
    "open_candle_confluence":  None,
    "open_candle_body_pct":    None,
    "open_candle_upper_wick":  None,
    "open_candle_lower_wick":  None,
    "open_candle_vol_ratio":   None,
    "open_candle_vwap_pos":    None,
    # PDT trade counter
    "day_trades_used":         0,
    "day_trades_warning_sent": False,
    # Multi-ticker secondary state
    "multi_ticker_signal_sent": False,
    "qqq_vol_gex_history":      [],
    "tsla_vol_gex_history":     [],
    "nvda_vol_gex_history":     [],
    "googl_vol_gex_history":    [],
    "qqq_last_score":           0,
    "tsla_last_score":          0,
    "nvda_last_score":          0,
    "googl_last_score":         0,
    "qqq_last_regime":          None,
    "tsla_last_regime":         None,
    "nvda_last_regime":         None,
    "googl_last_regime":        None,
    "qqq_last_grade":           None,
    "tsla_last_grade":          None,
    "nvda_last_grade":          None,
    "googl_last_grade":         None,
    # Legacy key kept for backward compat
    "qqq_signal_sent":          False,
    # Liquidity zones — updated every run_job, served instantly on command
    "spy_liq_zone":             None,   # dict with low, high, target, stop zones
    "qqq_liq_zone":             None,
    "tsla_liq_zone":            None,
    "nvda_liq_zone":            None,
    "googl_liq_zone":           None,
    # Command response cache — pre-built strings, served in <1s
    # Updated by run_job and analyze_open_candle
    "cache_status":             "",
    "cache_levels":             "",     # /levels — SPY liquidity zones
    "cache_all_levels":         "",     # /levels all — all tickers
    "cache_last_updated":       0,
}

# ─────────────────────────────────────────────
# GITHUB API — persistent CSV storage
# Pure HTTP, no git binary needed
# ─────────────────────────────────────────────
def _gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

def _gh_available():
    return bool(GITHUB_TOKEN)

def pull_csv_from_github():
    """
    Startup: download CSV from GitHub via REST API and merge with local.
    Merge strategy: GitHub = source of truth, keep local rows not yet pushed.
    Version-safe: missing columns filled with "" automatically.
    """
    if not _gh_available():
        print("⚠️ GITHUB_TOKEN not set — running without persistence")
        return False
    try:
        print("🔄 Pulling CSV from GitHub...")
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOG_FILE}"
        resp = requests.get(url, headers=_gh_headers(), timeout=15)
        if resp.status_code == 404:
            print("📝 No CSV on GitHub yet — starting fresh")
            return False
        if resp.status_code != 200:
            print(f"⚠️ GitHub pull error {resp.status_code}")
            return False
        data = resp.json()
        state["github_csv_sha"] = data.get("sha", "")
        raw = base64.b64decode(data["content"]).decode("utf-8")
        lines = raw.strip().split("\n")
        github_rows = []
        if len(lines) > 1:
            reader = csv.DictReader(lines)
            for row in reader:
                for h in LOG_HEADERS:
                    if h not in row:
                        row[h] = ""
                github_rows.append(row)
        if not github_rows:
            print("📝 GitHub CSV empty — starting fresh")
            return False
        local_rows = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", newline="") as f:
                local_rows = list(csv.DictReader(f))
        gh_keys = {(r["date"], r["time"], r.get("session_type", "MARKET"))
                   for r in github_rows}
        new_local = [r for r in local_rows
                     if (r["date"], r["time"], r.get("session_type", "MARKET"))
                     not in gh_keys]
        merged = github_rows + new_local
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
        today_str = now_pdt().strftime("%Y-%m-%d")
        today_count = sum(1 for r in merged if r["date"] == today_str)
        print(f"✅ CSV restored: {len(merged)} total rows ({today_count} today)")
        return True
    except Exception as e:
        print(f"CSV pull error: {e}")
        return False

def git_commit_log(reason="scheduled"):
    """
    Push CSV to GitHub via REST API.
    Rate-limited to once per 60s for 'reading' calls.
    EOD and overnight always push immediately.
    """
    now_epoch = time.time()
    if reason == "reading" and now_epoch - state.get("last_git_push", 0) < 60:
        return
    if not _gh_available() or not os.path.exists(LOG_FILE):
        return
    try:
        today_str = now_pdt().strftime("%Y-%m-%d")
        with open(LOG_FILE, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        sha = state.get("github_csv_sha", "")
        if not sha:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOG_FILE}"
            r = requests.get(url, headers=_gh_headers(), timeout=10)
            if r.status_code == 200:
                sha = r.json().get("sha", "")
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOG_FILE}"
        payload = {
            "message": f"Auto-log [{reason}]: SPY GEX {today_str}",
            "content": encoded,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha
        resp = requests.put(url, headers=_gh_headers(), json=payload, timeout=20)
        if resp.status_code in (200, 201):
            new_sha = resp.json().get("content", {}).get("sha", "")
            if new_sha:
                state["github_csv_sha"] = new_sha
            state["last_git_push"] = now_epoch
            print(f"✅ CSV pushed to GitHub [{reason}]")
        else:
            print(f"⚠️ GitHub push failed {resp.status_code}: {resp.text[:80]}")
    except Exception as e:
        print(f"GitHub push error (data safe locally): {e}")

def init_log():
    """Startup: restore from GitHub, seed session data."""
    print("─" * 60)
    print("STARTUP: Initializing persistent storage...")
    pull_csv_from_github()
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
        print("📝 New CSV created")
    else:
        with open(LOG_FILE, "r", newline="") as f:
            existing = list(csv.DictReader(f))
        today_str = now_pdt().strftime("%Y-%m-%d")
        today_rows = [r for r in existing if r.get("date") == today_str]
        print(f"📊 CSV ready: {len(existing)} total rows, {len(today_rows)} today")
    # Seed session open/high/low from real market data
    pdt = now_pdt()
    h = pdt.hour
    if 6 <= h <= 13 or (h >= 13 and pdt.weekday() < 5):
        print("🔍 Seeding session data from yfinance...")
        sd = fetch_true_session_data()
        if sd:
            if state["open_price"] is None:
                state["open_price"] = sd["open"]
            state["session_high"] = sd["high"]
            state["session_low"]  = sd["low"]
            state["true_session_close"] = sd["close"]
            state["prev_session_close"] = sd.get("prev_close")
            print(f"   O={sd['open']} H={sd['high']} L={sd['low']} C={sd['close']}")
            if sd.get("prev_close") and sd["open"]:
                gap = round(sd["open"] - sd["prev_close"], 2)
                state["gap_size"] = abs(gap)
                state["gap_direction"] = "UP" if gap > 0.5 else "DOWN" if gap < -0.5 else "NONE"
                if abs(gap) >= 1.0:
                    print(f"   Gap detected: {state['gap_direction']} ${abs(gap):.2f}")
    print("─" * 60)

def load_historical_context(days=30):
    """
    Loads last N days of CSV data as a compact text summary for Claude.
    Injected into morning brief and EOD prompts so Claude can reference
    actual win rates and patterns — not just today's data.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return "No historical data available yet."
        with open(LOG_FILE, "r", newline="") as f:
            all_rows = list(csv.DictReader(f))
        cutoff = (now_pdt() - timedelta(days=days)).strftime("%Y-%m-%d")
        market_rows = [
            r for r in all_rows
            if r.get("date", "") >= cutoff
            and r.get("session_type", "MARKET") == "MARKET"
            and r.get("signal_correct", "") != ""
        ]
        if not market_rows:
            return f"No completed signal data in last {days} days yet."
        total = len(market_rows)
        correct  = sum(1 for r in market_rows if r.get("signal_correct") == "YES")
        partial  = sum(1 for r in market_rows if r.get("signal_correct") == "PARTIAL")
        wrong    = sum(1 for r in market_rows if r.get("signal_correct") == "NO")
        win_rate = round((correct / total) * 100, 1) if total > 0 else 0
        regime_stats = {}
        for r in market_rows:
            reg = r.get("regime", "UNKNOWN")
            if reg not in regime_stats:
                regime_stats[reg] = {"correct": 0, "total": 0}
            regime_stats[reg]["total"] += 1
            if r.get("signal_correct") == "YES":
                regime_stats[reg]["correct"] += 1
        regime_lines = []
        for reg, s in sorted(regime_stats.items(), key=lambda x: -x[1]["total"]):
            if s["total"] >= 3:
                pct = round((s["correct"] / s["total"]) * 100, 0)
                regime_lines.append(f"  {reg}: {pct}% ({s['total']} signals)")
        override_rows = [r for r in market_rows if r.get("macro_override") == "YES"]
        ovr_correct = sum(1 for r in override_rows if r.get("signal_correct") == "YES")
        ovr_rate = round((ovr_correct / len(override_rows)) * 100, 1) if override_rows else 0
        recent = sorted(market_rows, key=lambda x: (x["date"], x["time"]))[-5:]
        recent_lines = []
        for r in recent:
            recent_lines.append(
                f"  {r['date']} {r['time']}: {r.get('gex_state','?')} | "
                f"{r.get('regime','?')} | Score:{r.get('conviction_score','?')} | "
                f"Outcome:{r.get('outcome_direction','?')} {r.get('outcome_points','?')}pts | "
                f"Correct:{r.get('signal_correct','?')}"
            )
        return (
            f"HISTORICAL PERFORMANCE (last {days} days, {total} signals):\n"
            f"  Win rate: {win_rate}% (Correct:{correct} Partial:{partial} Wrong:{wrong})\n"
            f"  Macro override YES: {ovr_rate}% win rate ({len(override_rows)} signals)\n"
            f"\nREGIME WIN RATES:\n" + "\n".join(regime_lines) +
            f"\n\nRECENT SIGNALS:\n" + "\n".join(recent_lines)
        )
    except Exception as e:
        print(f"Historical context error: {e}")
        return "Historical context unavailable."

# ─────────────────────────────────────────────
# MODULE 1: CALENDAR + OPEX CYCLE AWARENESS
# ─────────────────────────────────────────────
def get_opex_cycle_phase(days_to_opex):
    """
    Returns the OPEX cycle phase and its trading implications.

    Full cycle (monthly):
    Days 1-5  → Post-OPEX: fresh options, low gamma, drift
    Days 6-10 → Mid-cycle: gamma building, 1DTE setups good
    Days 11-14→ Pre-OPEX: max gamma acceleration, best week
    Days 15   → OPEX: explosive or pin
    """
    if days_to_opex is None:
        return "UNKNOWN", 0, ""
    if days_to_opex == 0:
        return "OPEX_DAY", 20, (
            "⚡ OPEX DAY — Maximum gamma decay. "
            "Explosive move or hard pin. Best 0DTE setups."
        )
    elif days_to_opex <= 2:
        return "OPEX_WEEK", 15, (
            f"⚡ OPEX IN {days_to_opex} DAYS — Gamma acceleration zone. "
            f"Best window for 400-700% setups. Full size OK on A signals."
        )
    elif days_to_opex <= 5:
        return "PRE_OPEX", 10, (
            f"📅 OPEX IN {days_to_opex} DAYS — Elevated gamma activity. "
            f"1DTE setups ideal here. GEX effects stronger than normal."
        )
    elif days_to_opex <= 10:
        return "MID_CYCLE", 5, (
            f"📅 OPEX IN {days_to_opex} DAYS — Mid-cycle. "
            f"Good for 1DTE swing setups. 0DTE needs strong VVIX."
        )
    else:
        return "EARLY_CYCLE", 0, (
            f"📅 OPEX IN {days_to_opex} DAYS — Early cycle. "
            f"Low gamma. Size down. Wait for mid-cycle to build."
        )

def get_calendar_flags():
    today = date.today()
    flags = []
    score_bonus = 0

    if today in QUARTER_END_DATES:
        flags.append(
            "🗓️ QUARTER END TODAY\n"
            "   → Window dressing = bullish bias\n"
            "   → Hedge unwind probability: VERY HIGH"
        )
        score_bonus += 15
    else:
        for qe in QUARTER_END_DATES:
            delta = (qe - today).days
            if 0 < delta <= 2:
                flags.append(
                    f"🗓️ QUARTER END IN {delta} DAYS\n"
                    f"   → Early hedge unwind likely"
                )
                score_bonus += 10
                break

    days_to_opex = None
    for opex in sorted(OPEX_DATES):
        delta = (opex - today).days
        if delta >= 0:
            days_to_opex = delta
            break

    cycle_phase, cycle_bonus, cycle_note = get_opex_cycle_phase(days_to_opex)
    score_bonus += cycle_bonus
    if cycle_note:
        flags.append(cycle_note)

    return flags, score_bonus, days_to_opex, cycle_phase

# ─────────────────────────────────────────────
# MODULE 2: GEX FETCH
# ─────────────────────────────────────────────
def fetch_gex():
    try:
        url = f"https://api.unusualwhales.com/api/stock/{TICKER}/spot-exposures"
        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()["data"]
        if not data:
            return None, None, None
        latest = data[-1]
        oi_gex  = float(latest["gamma_per_one_percent_move_oi"])
        vol_gex = float(latest["gamma_per_one_percent_move_vol"])
        price   = float(latest["price"])
        return oi_gex, vol_gex, price
    except Exception as e:
        print(f"GEX fetch error: {e}")
        return None, None, None

# ─────────────────────────────────────────────
# MODULE 3: REGIME DETECTION
# ─────────────────────────────────────────────
def detect_regime(oi_gex, vol_gex, vol_gex_history):
    if len(vol_gex_history) < 3:
        return "INSUFFICIENT_DATA", 50
    recent  = vol_gex_history[-3:]
    roc     = recent[-1] - recent[0]
    mid_roc = recent[-1] - recent[-2]
    vol_pos = vol_gex > 0
    oi_pos  = oi_gex > 0
    if vol_pos and oi_pos:
        return "BULLISH_MOMENTUM", 95
    if vol_pos and not oi_pos:
        return "HEDGE_UNWIND_CONFIRMED", 88
    if not vol_pos:
        if roc > 0 and mid_roc > 0:
            return "HEDGE_UNWIND_EARLY", 72
        elif roc > 0 and mid_roc <= 0:
            return "TRANSITION_ZONE", 58
        else:
            roc_pct = abs(roc / recent[0]) * 100 if recent[0] != 0 else 0
            return "BEARISH_HEDGE_BUILD", int(min(85, 60 + roc_pct))
    return "NEUTRAL", 50

def get_regime_signal(regime, confidence, oi_b, vol_b):
    explanations = {
        "BULLISH_MOMENTUM":      "🟢 BULLISH MOMENTUM\nBoth OI and Vol GEX positive.\n→ Calls OK but don't expect 400%+.",
        "HEDGE_UNWIND_CONFIRMED":"🚀 HEDGE UNWIND CONFIRMED\nVol GEX flipped — put SELLING is the fuel.\n→ CALLS strongly favored. 400-700% setup.",
        "HEDGE_UNWIND_EARLY":    "🔄 EARLY HEDGE UNWIND — caught early.\nVol GEX still negative but improving fast.\n→ Prepare call entry.",
        "TRANSITION_ZONE":       "⚠️ TRANSITION ZONE\nVol GEX decelerating — could go either way.\n→ No new entries. Wait.",
        "BEARISH_HEDGE_BUILD":   "🔴 BEARISH HEDGE BUILD\nVol GEX accelerating negative.\n→ PUTS favored.",
        "NEUTRAL":               "⚪ NEUTRAL\n→ Stay out. Wait for regime.",
        "INSUFFICIENT_DATA":     "📊 COLLECTING DATA\n→ Check back in 30-45 min.",
    }
    base = explanations.get(regime, "Unknown regime")
    return f"{base}\nConfidence: {confidence}% | OI: {oi_b}B | Vol: {vol_b}B"

# ─────────────────────────────────────────────
# MODULE 4: HEDGE UNWIND DETECTOR
# ─────────────────────────────────────────────
def fetch_hedge_unwind_signals():
    try:
        url = f"https://api.unusualwhales.com/api/stock/{TICKER}/options-contracts"
        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        params = {"limit": 100, "order": "desc"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json().get("data", [])
        if not data:
            return False, 0, [], "NEUTRAL"
        signals = []
        score = 0
        for contract in data:
            try:
                ctype  = str(contract.get("type", "")).upper()
                volume = float(contract.get("volume", 0) or 0)
                oi     = float(contract.get("open_interest", 0) or 0)
                strike = float(contract.get("strike", 0) or 0)
                exec_  = str(contract.get("execution_estimate", "")).upper()
                ratio  = volume / oi if oi > 0 else 0
                is_put = "PUT" in ctype
                is_call= "CALL" in ctype
                if is_put and ratio >= 50 and volume >= 10000:
                    score += 25
                    signals.append(f"🚀 PUT HEDGE CLOSING: ${strike:.0f}P Vol/OI: {round(ratio)}x")
                elif is_put and ratio >= 10 and volume >= 5000:
                    score += 10
                    signals.append(f"⚠️ PUT CLOSING: ${strike:.0f}P Vol/OI: {round(ratio)}x")
                if is_put and "DESCENDING" in exec_ and volume >= 5000:
                    score += 15
                    signals.append(f"🔽 DESCENDING FILL PUT: ${strike:.0f}P")
                if is_call and "SWEEP" in exec_ and volume >= 5000:
                    score += 8
                    signals.append(f"📈 CALL SWEEP: ${strike:.0f}C {int(volume/1000)}K")
            except Exception:
                continue
        score = min(score, 100)
        if score >= 40:
            return True, score, signals[:6], "BULLISH — Hedge unwind active"
        elif score >= 20:
            return True, score, signals[:6], "LEANING BULLISH — Early signs"
        return False, score, signals[:6], "NEUTRAL"
    except Exception as e:
        print(f"Hedge unwind error: {e}")
        return False, 0, [], "UNAVAILABLE"

# ─────────────────────────────────────────────
# MODULE 5: VANNA / CHARM ENGINE
# ─────────────────────────────────────────────
def fetch_vanna_charm():
    try:
        url = f"https://api.unusualwhales.com/api/stock/{TICKER}/greek-exposure"
        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        v_data = requests.get(url, headers=headers,
                              params={"greek": "vanna", "expiry": "0dte"},
                              timeout=10).json().get("data", [])
        c_data = requests.get(url, headers=headers,
                              params={"greek": "charm", "expiry": "0dte"},
                              timeout=10).json().get("data", [])
        vanna_target = vanna_strength = charm_target = charm_strength = None
        if v_data:
            best = max(v_data, key=lambda x: float(x.get("vanna", 0) or 0))
            vanna_target   = float(best.get("strike", 0))
            vanna_strength = float(best.get("vanna", 0) or 0)
        if c_data:
            worst = min(c_data, key=lambda x: float(x.get("charm", 0) or 0))
            charm_target   = float(worst.get("strike", 0))
            charm_strength = float(worst.get("charm", 0) or 0)
        conflict = (vanna_target and charm_target and
                    abs(vanna_target - charm_target) <= 1.0)
        return vanna_target, vanna_strength, charm_target, charm_strength, conflict
    except Exception as e:
        print(f"Vanna/Charm error: {e}")
        return None, 0, None, 0, False

def get_vanna_charm_read(vt, vs, ct, cs, price, conflict):
    pdt = now_pdt()
    mins_since_open = (pdt.hour - 6) * 60 + pdt.minute - 30
    vanna_window = mins_since_open < 270
    mins_left = max(0, 270 - mins_since_open)
    lines = []
    if vt and price:
        dist = vt - price
        pull = "STRONG" if abs(dist) <= 2 else "MODERATE" if abs(dist) <= 5 else "WEAK"
        lines.append(f"Vanna magnet: ${vt} (${abs(dist):.2f} {'above' if dist>0 else 'below'} spot) — {pull}")
        lines.append(f"Vanna strength: {round(vs/1e6,1)}M")
    if ct:
        lines.append(f"Charm headwind: ${ct} — {round(cs/1e6,1)}M")
    if conflict:
        lines.append("⚠️ CONFLICT: Vanna + Charm stacked — CONSOLIDATION TRAP risk")
    lines.append(f"⏰ Vanna window: ~{mins_left} min left" if vanna_window
                 else "🕐 Charm dominant — vanna expired")
    return "\n".join(lines), vanna_window

# ─────────────────────────────────────────────
# MODULE 6: CONSOLIDATION DETECTOR
# ─────────────────────────────────────────────
def run_consolidation_check(current_price, current_iv, current_volume, vt, ct, conflict):
    pdt = now_pdt()
    mins = (pdt.hour - 6) * 60 + pdt.minute - 30
    if mins > 45 or mins < 5:
        return False, 0, []
    state["open_time_prices"].append(current_price)
    score = 0
    signals = []
    if vt:
        prox = abs(current_price - vt) / vt * 100
        if prox <= 0.5:
            score += 30
            signals.append(f"⚠️ Price within 0.5% of vanna ${vt} — magnetic stall")
    if state["open_iv"] and current_iv:
        iv_chg = abs(current_iv - state["open_iv"]) / state["open_iv"] * 100
        if iv_chg < 2.0:
            score += 25
            signals.append(f"⚠️ IV only {iv_chg:.1f}% from open — no conviction")
    if state["open_volume"] and current_volume:
        vol_ratio = current_volume / state["open_volume"]
        if vol_ratio < 0.7:
            score += 20
            signals.append(f"⚠️ Volume only {round(vol_ratio*100)}% of open")
    if conflict:
        score += 15
        signals.append("⚠️ Vanna + charm stacked — forces canceling")
    if len(state["open_time_prices"]) >= 4:
        prices = state["open_time_prices"]
        changes = sum(1 for i in range(1, len(prices)-1)
                      if (prices[i]-prices[i-1]) * (prices[i+1]-prices[i]) < 0)
        rng = max(prices) - min(prices)
        if changes >= 3 and rng < 1.5:
            score += 10
            signals.append(f"⚠️ {changes} direction changes, ${rng:.2f} range — choppy")
    return score >= 50, score, signals

# ─────────────────────────────────────────────
# MODULE 7: VIX / VVIX
# ─────────────────────────────────────────────
def fetch_vix_data():
    try:
        vix_h  = yf.Ticker("^VIX").history(period="5d", interval="1d")
        vvix_h = yf.Ticker("^VVIX").history(period="5d", interval="1d")
        vix3m_h= yf.Ticker("^VIX3M").history(period="2d", interval="1d")
        vix_spot  = float(vix_h["Close"].iloc[-1])  if not vix_h.empty  else None
        vvix_val  = float(vvix_h["Close"].iloc[-1]) if not vvix_h.empty else None
        vix3m_val = float(vix3m_h["Close"].iloc[-1])if not vix3m_h.empty else None
        if vix_spot:
            state["vix_history"].append(vix_spot)
            if len(state["vix_history"]) > 5:
                state["vix_history"].pop(0)
        if vix_spot and vix3m_val:
            if vix_spot > vix3m_val * 1.02:
                vix_term, term_sig = "BACKWARDATION", "⚡ BACKWARDATION — Fear spike. Explosive moves."
            elif vix_spot < vix3m_val * 0.98:
                vix_term, term_sig = "CONTANGO", "😴 CONTANGO — Calm market. Chop risk."
            else:
                vix_term, term_sig = "FLAT", "⚠️ FLAT — Neutral term structure."
        else:
            vix_term, term_sig = "UNKNOWN", "Term structure unavailable"
        vix_momentum = ""
        if len(state["vix_history"]) >= 3:
            vix_roc = state["vix_history"][-1] - state["vix_history"][0]
            vix_momentum = (" ↑ RISING — fear building" if vix_roc > 1.5
                            else " ↓ FALLING — IV crush, vanna fuel" if vix_roc < -1.5
                            else " → STABLE")
        if vix_spot:
            if vix_spot >= 30:
                vix_sig = f"🔴 EXTREME FEAR ({round(vix_spot,1)}){vix_momentum}"
            elif vix_spot >= 22:
                vix_sig = f"🟠 ELEVATED ({round(vix_spot,1)}){vix_momentum}"
            elif vix_spot >= 16:
                vix_sig = f"🟡 MODERATE ({round(vix_spot,1)}){vix_momentum}"
            else:
                vix_sig = f"🟢 LOW ({round(vix_spot,1)}){vix_momentum} — chop risk"
        else:
            vix_sig = "Unavailable"
        if vvix_val:
            if vvix_val >= 100:
                vvix_sig = f"🔥 EXPLOSIVE ({round(vvix_val,1)}) — Velocity day likely."
            elif vvix_val >= 90:
                vvix_sig = f"⚡ ACTIVE ({round(vvix_val,1)}) — Good momentum"
            elif vvix_val >= 85:
                vvix_sig = f"⚠️ BORDERLINE ({round(vvix_val,1)})"
            else:
                vvix_sig = f"😴 QUIET ({round(vvix_val,1)}) — Chop day likely."
        else:
            vvix_sig = "Unavailable"
        return vix_spot, vvix_val, vix_term, term_sig, vix_sig, vvix_sig
    except Exception as e:
        print(f"VIX error: {e}")
        return None, None, "UNKNOWN", "Unavailable", "Unavailable", "Unavailable"

# ─────────────────────────────────────────────
# MODULE 8: TICK + INVENTORY
# ─────────────────────────────────────────────
def fetch_tick_and_inventory():
    try:
        spy = yf.download("SPY", period="1d", interval="1m", progress=False)
        if spy.empty or len(spy) < 10:
            return "UNAVAILABLE", 0, "NEUTRAL", False
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        closes  = spy["Close"].iloc[-10:].values.flatten().astype(float)
        opens   = spy["Open"].iloc[-10:].values.flatten().astype(float)
        volumes = spy["Volume"].iloc[-10:].values.flatten().astype(float)
        up   = sum(1 for c, o in zip(closes, opens) if c > o)
        down = sum(1 for c, o in zip(closes, opens) if c < o)
        tick_approx = (up - down) * 100
        state["tick_history"].append(tick_approx)
        if len(state["tick_history"]) > 6:
            state["tick_history"].pop(0)
        s_bull = (len(state["tick_history"]) >= 3 and
                  all(t > 300 for t in state["tick_history"][-3:]))
        s_bear = (len(state["tick_history"]) >= 3 and
                  all(t < -300 for t in state["tick_history"][-3:]))
        state["session_high"] = float(spy["High"].values.max())
        state["session_low"]  = float(spy["Low"].values.min())
        avg_vol    = float(np.mean(volumes[:-3])) if len(volumes) > 3 else 0
        recent_vol = float(np.mean(volumes[-3:]))
        volume_surge = recent_vol > avg_vol * 1.5
        pdt = now_pdt()
        mins_since_open = (pdt.hour - 6) * 60 + pdt.minute - 30
        open_drive = False
        if mins_since_open <= 30 and volume_surge and (s_bull or s_bear):
            open_drive = True
            state["open_drive_detected"] = True
        try:
            spy["vwap"] = ((spy["Close"] * spy["Volume"]).cumsum()
                           / spy["Volume"].cumsum())
            cp   = float(spy["Close"].iloc[-1])
            vwap = float(spy["vwap"].iloc[-1])
            state["session_vwap"] = vwap
            if cp > vwap * 1.002:
                inv_bias = "BULL ZONE"; state["inventory_bias"] = "BULL"
            elif cp < vwap * 0.998:
                inv_bias = "BEAR ZONE"; state["inventory_bias"] = "BEAR"
            else:
                inv_bias = "NEUTRAL (100%)"; state["inventory_bias"] = "NEUTRAL"
        except Exception:
            inv_bias = "NEUTRAL"
        if tick_approx >= 600 or s_bull:
            tick_sig = f"📈 STRONG BUYING (TICK ~+{tick_approx})" + (" — Open drive!" if open_drive else "")
        elif tick_approx >= 200:
            tick_sig = f"🟡 MILD BUYING (TICK ~+{tick_approx})"
        elif tick_approx <= -600 or s_bear:
            tick_sig = f"📉 STRONG SELLING (TICK ~{tick_approx})" + (" — Open drive DOWN!" if open_drive else "")
        elif tick_approx <= -200:
            tick_sig = f"🟡 MILD SELLING (TICK ~{tick_approx})"
        else:
            tick_sig = f"⚪ NEUTRAL (TICK ~{tick_approx}) — Inventory: {inv_bias}"
        return tick_sig, tick_approx, inv_bias, open_drive
    except Exception as e:
        print(f"TICK/Inventory error: {e}")
        return "UNAVAILABLE", 0, "NEUTRAL", False

# ─────────────────────────────────────────────
# MODULE 9: CONVICTION SCORER
# ─────────────────────────────────────────────
def score_conviction(vix_spot, vvix_val, vix_term, vol_gex, prev_vol_gex,
                     regime, unwind_score, cal_bonus, vanna_window,
                     conflict, ratio, tick_approx, inv_bias, open_drive):
    score = 0
    checklist = []
    if vvix_val:
        if vvix_val >= 100:
            score += 25; checklist.append(f"✅ VVIX {round(vvix_val,1)} ≥ 100 — Velocity (+25)")
        elif vvix_val >= 90:
            score += 18; checklist.append(f"✅ VVIX {round(vvix_val,1)} Active (+18)")
        elif vvix_val >= 85:
            score += 10; checklist.append(f"⚠️ VVIX {round(vvix_val,1)} Borderline (+10)")
        else:
            checklist.append(f"❌ VVIX {round(vvix_val,1)} < 85 — Chop risk (+0)")
    if vix_term == "BACKWARDATION":
        score += 15; checklist.append("✅ VIX Backwardation (+15)")
    elif vix_term == "FLAT":
        score += 7;  checklist.append("⚠️ VIX Flat (+7)")
    else:
        checklist.append("❌ VIX Contango (+0)")
    if len(state["vix_history"]) >= 3:
        vix_roc = state["vix_history"][-1] - state["vix_history"][0]
        if abs(vix_roc) > 1.5:
            score += 5; checklist.append("✅ VIX momentum (+5)")
    regime_pts = {
        "HEDGE_UNWIND_CONFIRMED": 20, "BULLISH_MOMENTUM": 18,
        "BEARISH_HEDGE_BUILD": 16,    "HEDGE_UNWIND_EARLY": 14,
        "TRANSITION_ZONE": 8,         "NEUTRAL": 0, "INSUFFICIENT_DATA": 0,
    }
    rpts = regime_pts.get(regime, 0)
    score += rpts
    checklist.append(f"{'✅' if rpts>=14 else '⚠️' if rpts>=8 else '❌'} Regime: {regime} (+{rpts})")
    if unwind_score >= 40:
        score += 15; checklist.append(f"✅ Hedge unwind {unwind_score}/100 (+15)")
    elif unwind_score >= 20:
        score += 8;  checklist.append(f"⚠️ Early unwind {unwind_score}/100 (+8)")
    else:
        checklist.append("❌ No unwind (+0)")
    cal_pts = min(cal_bonus, 15)
    score += cal_pts
    if cal_pts >= 5:
        checklist.append(f"✅ Calendar +{cal_pts}")
    if vanna_window:
        score += 10; checklist.append("✅ Vanna window open (+10)")
    else:
        checklist.append("❌ Vanna window closed (+0)")
    if abs(tick_approx) >= 600:
        score += 10; checklist.append(f"✅ TICK strong (+10)")
    elif abs(tick_approx) >= 300:
        score += 5;  checklist.append("⚠️ TICK moderate (+5)")
    else:
        checklist.append(f"❌ TICK neutral (+0)")
    if inv_bias in ["BULL ZONE", "BEAR ZONE"]:
        score += 5; checklist.append(f"✅ Inventory: {inv_bias} (+5)")
    else:
        checklist.append("❌ Inventory: NEUTRAL (+0)")
    if open_drive:
        score += 10; checklist.append("🚀 OPEN DRIVE (+10)")
    if conflict:
        score -= 20; checklist.append("🚨 Vanna/charm conflict (-20)")
    score = max(0, min(100, score))
    if score >= 80:
        grade, rec = "A+ 🔥", "FULL SIZE. 400-700% day confirmed."
    elif score >= 65:
        grade, rec = "B+ ✅", "NORMAL SIZE. 200-400% realistic."
    elif score >= 50:
        grade, rec = "C ⚠️", "HALF SIZE. Wait for open confirmation."
    elif score >= 35:
        grade, rec = "D 🔴", "MINIMAL or sit out. High chop risk."
    else:
        grade, rec = "F ❌", "DO NOT TRADE. Theta destroys premium."
    return score, grade, rec, checklist

# ─────────────────────────────────────────────
# MODULE 10: NEWS SENTIMENT — UW API
# ─────────────────────────────────────────────
def fetch_news_sentiment():
    try:
        if not UW_TOKEN:
            return "NEUTRAL", 50, "NONE", 0, "NO"
        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        r = requests.get("https://api.unusualwhales.com/api/news/headlines",
                         headers=headers, params={"major_only": "true", "limit": 20},
                         timeout=8)
        if r.status_code != 200:
            return "NEUTRAL", 50, "NONE", 0, "NO"
        data = r.json().get("data", [])
        if not data:
            return "NEUTRAL", 50, "NONE", 0, "NO"
        bull = bear = major = geo = fed = tariff = 0
        geo_w    = ["iran","war","hormuz","military","attack","strike","nato","conflict","missile","ceasefire","nuclear","houthi"]
        fed_w    = ["federal reserve","fed","powell","fomc","interest rate","rate hike","rate cut","monetary policy"]
        tariff_w = ["tariff","trade war","import tax","liberation day","trade deal","sanctions","reciprocal tariff"]
        for item in data[:20]:
            hl  = (item.get("headline") or "").lower()
            uw_s= (item.get("sentiment") or "").lower()
            maj = item.get("is_major", False)
            if maj: major += 1
            if uw_s == "positive": bull += (2 if maj else 1)
            elif uw_s == "negative": bear += (2 if maj else 1)
            for w in geo_w:
                if w in hl: geo += 1; break
            for w in fed_w:
                if w in hl: fed += 1; break
            for w in tariff_w:
                if w in hl: tariff += 1; break
        today = date.today()
        if today in FED_DATES_2026 or fed >= 3:
            cat, strength = "FED",    min(100, fed * 15 + 50)
        elif geo >= 3:
            cat, strength = "GEO",    min(100, geo * 12 + 40)
        elif tariff >= 3:
            cat, strength = "TARIFF", min(100, tariff * 12 + 40)
        elif today in OPEX_DATES:
            cat, strength = "OPEX",   60
        else:
            cat, strength = "NONE",   0
        if major >= 3:
            strength = min(100, strength + 15)
        total = bull + bear
        if total == 0:
            sent, ns = "NEUTRAL", 50
        elif bull > bear * 1.5:
            sent, ns = "BULLISH", min(100, int(50 + (bull/total)*50))
        elif bear > bull * 1.5:
            sent, ns = "BEARISH", max(0, int(50 - (bear/total)*50))
        else:
            sent, ns = "NEUTRAL", 50
        override = "YES" if (
            strength >= 60 or major >= 5 or
            (cat == "FED" and strength >= 50) or
            (cat == "GEO" and geo >= 4) or
            (cat == "TARIFF" and strength >= 60)
        ) else "NO"
        print(f"📰 News: {sent} | {cat} ({strength}) | Override:{override} | Bull:{bull} Bear:{bear}")
        return sent, ns, cat, strength, override
    except Exception as e:
        print(f"News sentiment error: {e}")
        return "NEUTRAL", 50, "NONE", 0, "NO"

# ─────────────────────────────────────────────
# MODULE 11: FUTURES PRE-MARKET CHECK
# ─────────────────────────────────────────────
def fetch_futures_direction():
    """
    Checks ES futures direction and gap at open.
    Predicts gap direction before market opens.
    Returns: direction (UP/DOWN/FLAT), change_pct, price
    """
    try:
        es = yf.Ticker("ES=F").history(period="2d", interval="5m")
        if es.empty:
            return "FLAT", 0.0, None
        if isinstance(es.columns, pd.MultiIndex):
            es.columns = es.columns.get_level_values(0)
        current = float(es["Close"].iloc[-1])
        # Compare to previous regular session close (~1pm PDT = 8pm UTC)
        prev_candidates = es[es.index.hour == 20]
        prev_close = (float(prev_candidates["Close"].iloc[-1])
                      if not prev_candidates.empty
                      else float(es["Close"].iloc[0]))
        chg_pct = round(((current - prev_close) / prev_close) * 100, 2)
        direction = "UP" if chg_pct > 0.3 else "DOWN" if chg_pct < -0.3 else "FLAT"
        return direction, chg_pct, round(current, 2)
    except Exception as e:
        print(f"Futures fetch error: {e}")
        return "FLAT", 0.0, None

# ─────────────────────────────────────────────
# MODULE 12: GAP CLASSIFICATION ENGINE
# ─────────────────────────────────────────────
# Gap types (what you will actually see in the market):
#
# DIRECTIONAL (gap and go):
#   Market gaps, holds above/below open, trends all day.
#   Institutions agree with the gap direction.
#   Entry: calls/puts at open, hold through day.
#
# FADE_THEN_STATIC (gap up/down, fade to VWAP, then chop):
#   Most common type. Market gaps, reverses to VWAP,
#   then oscillates near VWAP rest of day.
#   Entry: fade the gap direction back to VWAP,
#   then no trade — chop kills premium.
#
# FULL_FADE (gap fully fills to prev close):
#   Gap opens, entire move reverses back to prev session close.
#   Institutions used the gap to exit positions.
#   Entry: aggressive fade, hold to prev close target.
#
# GAP_AND_REVERSE (gap one direction, full reversal other way):
#   Rarest. Gap up $5, then falls $8 below prev close.
#   Usually a macro shock or surprise news.
#   Entry: wait for first 15min then trade the reversal.
#
# STATIC (tiny gap, market pins near open all day):
#   Gap < $1.50, no meaningful overnight catalyst.
#   Vol GEX flat, no conviction either way.
#   Entry: no trade — theta decays premium fast.
# ─────────────────────────────────────────────

def classify_gap(vol_gex, vix_change, news_sentiment,
                 macro_override, catalyst_type, catalyst_strength,
                 futures_chg):
    """
    Classifies the overnight gap at market open using
    all available signals. Returns (gap_type, conviction, explanation).

    Called once per session at 6:30am PDT open.
    Stored in state and logged to CSV every row.
    """
    gap_size  = state.get("gap_size", 0)
    gap_dir   = state.get("gap_direction", "NONE")
    open_vgex = state.get("open_vol_gex_snapshot")  # Vol GEX at open
    prev_vgex = state.get("overnight_vol_gex_close") # Vol GEX at yesterday close

    if gap_dir == "NONE" or gap_size < 0.75:
        state["gap_type"]      = "STATIC"
        state["gap_conviction"] = 85
        return "STATIC", 85, (
            "Gap < $0.75 — no meaningful overnight move.\n"
            "Market likely pins near open. "
            "Theta decays premium fast in this environment.\n"
            "→ No gap trade. Wait for GEX regime signal."
        )

    hold_score = 0
    fade_score = 0
    signals_hold = []
    signals_fade = []

    # ── SIGNAL 1: Vol GEX direction vs gap direction ──
    # This is the single most important signal.
    # If institutions are buying puts (negative Vol GEX)
    # while futures are positive — they don't believe the gap.
    if open_vgex is not None:
        vgex_b = round(open_vgex / 1e9, 2)
        if gap_dir == "UP":
            if open_vgex < -5e9:
                fade_score += 35
                signals_fade.append(
                    f"Vol GEX deeply negative ({vgex_b}B) despite gap UP "
                    f"— institutions still holding put protection, "
                    f"don't believe the move"
                )
            elif open_vgex < 0:
                fade_score += 20
                signals_fade.append(
                    f"Vol GEX still negative ({vgex_b}B) vs gap UP "
                    f"— mild bearish disagreement"
                )
            else:
                hold_score += 30
                signals_hold.append(
                    f"Vol GEX positive ({vgex_b}B) matches gap UP "
                    f"— institutions agreeing with move"
                )
        elif gap_dir == "DOWN":
            if open_vgex > 5e9:
                fade_score += 35
                signals_fade.append(
                    f"Vol GEX strongly positive ({vgex_b}B) despite gap DOWN "
                    f"— put buying stopped, institutions not believing bear move"
                )
            elif open_vgex > 0:
                fade_score += 20
                signals_fade.append(
                    f"Vol GEX positive ({vgex_b}B) vs gap DOWN "
                    f"— mild bullish disagreement with gap"
                )
            else:
                hold_score += 30
                signals_hold.append(
                    f"Vol GEX negative ({vgex_b}B) matches gap DOWN "
                    f"— institutions agreeing with bearish move"
                )

    # ── SIGNAL 2: Vol GEX improvement overnight ──
    # If Vol GEX improved overnight (less negative or flipped positive)
    # alongside a gap up — two forces aligning = strong hold signal.
    if open_vgex is not None and prev_vgex is not None:
        overnight_change = open_vgex - prev_vgex
        if gap_dir == "UP" and overnight_change > 2e9:
            hold_score += 20
            signals_hold.append(
                f"Vol GEX improved +${round(overnight_change/1e9,1)}B overnight "
                f"alongside gap UP — institutional put closing confirms gap"
            )
        elif gap_dir == "DOWN" and overnight_change < -2e9:
            hold_score += 20
            signals_hold.append(
                f"Vol GEX worsened ${round(overnight_change/1e9,1)}B overnight "
                f"alongside gap DOWN — institutional put buying confirms gap"
            )
        elif gap_dir == "UP" and overnight_change < -2e9:
            fade_score += 15
            signals_fade.append(
                f"Vol GEX worsened ${round(overnight_change/1e9,1)}B overnight "
                f"despite gap UP — smart money adding puts into strength"
            )

    # ── SIGNAL 3: VIX behavior overnight ──
    # Fear leaving = institutions comfortable = gap holds
    # Fear building = someone knows something = gap may fade or reverse
    if vix_change is not None:
        if gap_dir == "UP":
            if vix_change < -1.5:
                hold_score += 20
                signals_hold.append(
                    f"VIX fell {abs(vix_change):.1f}pts overnight — "
                    f"fear leaving as market gaps up = real conviction"
                )
            elif vix_change > 1.5:
                fade_score += 20
                signals_fade.append(
                    f"VIX RISING +{vix_change:.1f}pts despite gap UP — "
                    f"someone is buying protection into strength = trap signal"
                )
            else:
                fade_score += 5
                signals_fade.append(
                    f"VIX flat overnight ({vix_change:+.1f}pts) — "
                    f"no fear compression to fuel gap continuation"
                )
        elif gap_dir == "DOWN":
            if vix_change > 2.0:
                hold_score += 20
                signals_hold.append(
                    f"VIX spiked +{vix_change:.1f}pts with gap DOWN — "
                    f"real fear building, institutions hedging hard"
                )
            elif vix_change < -1.5:
                fade_score += 20
                signals_fade.append(
                    f"VIX FALLING despite gap DOWN — "
                    f"fear leaving as price drops = no real conviction in bear move"
                )

    # ── SIGNAL 4: News / macro override ──
    if macro_override == "YES":
        if catalyst_strength >= 70:
            if (gap_dir == "UP" and news_sentiment == "BULLISH") or \
               (gap_dir == "DOWN" and news_sentiment == "BEARISH"):
                hold_score += 15
                signals_hold.append(
                    f"Macro catalyst ({catalyst_type} strength {catalyst_strength}) "
                    f"aligns with gap direction — fundamental driver present"
                )
            else:
                fade_score += 20
                signals_fade.append(
                    f"Macro override YES but sentiment contradicts gap — "
                    f"news-driven gaps with conflicting signals fade hard"
                )
        else:
            fade_score += 10
            signals_fade.append(
                f"Macro override active ({catalyst_type}) — "
                f"macro gaps fade more often than technical gaps"
            )

    # ── SIGNAL 5: Gap size vs conviction ──
    # Large gaps (>$5) are harder to hold than small gaps ($1-3)
    if gap_size >= 8:
        fade_score += 20
        signals_fade.append(
            f"Large gap ${gap_size:.2f} — gaps >$8 fill completely "
            f"within same session 70%+ of the time"
        )
    elif gap_size >= 5:
        fade_score += 10
        signals_fade.append(
            f"Gap ${gap_size:.2f} — medium gap, fade to at least "
            f"VWAP expected before any continuation"
        )
    elif gap_size >= 2:
        hold_score += 5
        signals_hold.append(
            f"Gap ${gap_size:.2f} — small gap, easier to hold "
            f"with institutional backing"
        )

    # ── SIGNAL 6: Futures magnitude ──
    abs_fut = abs(futures_chg) if futures_chg else 0
    if abs_fut >= 1.5:
        fade_score += 10
        signals_fade.append(
            f"Futures moved {abs_fut:.1f}% overnight — "
            f"large futures moves mean early birds already positioned, "
            f"latecomers fade the open"
        )

    # ── CLASSIFY ──
    total = hold_score + fade_score
    if total == 0:
        state["gap_type"]       = "UNKNOWN"
        state["gap_conviction"] = 40
        return "UNKNOWN", 40, "Insufficient data to classify gap."

    hold_pct = hold_score / total
    conviction = min(95, int(max(hold_score, fade_score) / total * 100))

    hold_lines = "\n  → ".join(signals_hold) if signals_hold else "None"
    fade_lines = "\n  → ".join(signals_fade) if signals_fade else "None"

    # ── Determine type based on scores + magnitude ──
    if hold_pct >= 0.65:
        gap_type = "DIRECTIONAL"
        state["gap_type"]       = gap_type
        state["gap_conviction"] = conviction
        return gap_type, conviction, (
            f"HOLD signals ({hold_score}pts):\n  → {hold_lines}\n\n"
            f"FADE signals ({fade_score}pts):\n  → {fade_lines}"
        )
    elif hold_pct <= 0.35 and gap_size >= 5:
        gap_type = "FULL_FADE"
        state["gap_type"]       = gap_type
        state["gap_conviction"] = conviction
        return gap_type, conviction, (
            f"FADE signals ({fade_score}pts):\n  → {fade_lines}\n\n"
            f"HOLD signals ({hold_score}pts):\n  → {hold_lines}"
        )
    elif hold_pct <= 0.35:
        gap_type = "FADE_THEN_STATIC"
        state["gap_type"]       = gap_type
        state["gap_conviction"] = conviction
        return gap_type, conviction, (
            f"FADE signals ({fade_score}pts):\n  → {fade_lines}\n\n"
            f"HOLD signals ({hold_score}pts):\n  → {hold_lines}"
        )
    else:
        # Ambiguous — slight lean either way
        if hold_score >= fade_score:
            gap_type = "DIRECTIONAL"
        else:
            gap_type = "FADE_THEN_STATIC"
        conviction = max(35, conviction - 20)  # lower conviction when ambiguous
        state["gap_type"]       = gap_type
        state["gap_conviction"] = conviction
        return gap_type, conviction, (
            f"AMBIGUOUS (leaning {gap_type})\n"
            f"HOLD signals ({hold_score}pts):\n  → {hold_lines}\n\n"
            f"FADE signals ({fade_score}pts):\n  → {fade_lines}"
        )


def build_gap_alert(gap_type, conviction, detail, vol_gex, vix_change,
                    news_sentiment, catalyst_type):
    """
    Writes the plain English gap classification alert
    that fires at 6:30am when market opens.
    """
    gap_dir  = state.get("gap_direction", "NONE")
    gap_size = state.get("gap_size", 0)
    open_px  = state.get("open_price")
    prev_cl  = state.get("prev_session_close")
    now_str  = now_pdt().strftime("%H:%M")

    dir_word  = "UP" if gap_dir == "UP" else "DOWN"
    dir_emoji = "🟢" if gap_dir == "UP" else "🔴"

    # Type-specific plain English explanation
    type_blocks = {

        "DIRECTIONAL": (
            f"🚀 GAP {dir_word} — DIRECTIONAL SETUP\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Gap: {dir_emoji} ${gap_size:.2f} "
            f"(open ${open_px} vs prev close ${prev_cl})\n"
            f"Conviction: {conviction}%\n\n"
            f"What this means:\n"
            f"The gap is backed by real institutional agreement.\n"
            f"When you see a DIRECTIONAL gap, institutions are\n"
            f"positioning the same way the gap is moving.\n"
            f"They are NOT going to fade it — they're adding.\n\n"
            f"What to watch at open:\n"
            f"→ First 3-5 candles stay {'above' if gap_dir=='UP' else 'below'} VWAP\n"
            f"→ Volume above average on first candle\n"
            f"→ No immediate rejection back through open price\n\n"
            f"Trade:\n"
            f"{'→ CALLS. Enter on VWAP hold after first 5min.' if gap_dir=='UP' else '→ PUTS. Enter on VWAP rejection after first 5min.'}\n"
            f"Target: vanna level (check morning brief).\n"
            f"Stop: close back through open price.\n\n"
            f"Risk: If price immediately reverses below open\n"
            f"in the first candle — this is a trap. Exit fast."
        ),

        "FADE_THEN_STATIC": (
            f"↩️ GAP {dir_word} — FADE THEN CHOP SETUP\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Gap: {dir_emoji} ${gap_size:.2f} "
            f"(open ${open_px} vs prev close ${prev_cl})\n"
            f"Conviction: {conviction}%\n\n"
            f"What this means:\n"
            f"The gap opens {'up' if gap_dir=='UP' else 'down'} but institutions\n"
            f"{'are still holding puts — they don' + chr(39) + 't believe the bull move.' if gap_dir=='UP' else 'stopped adding puts — they don' + chr(39) + 't believe the bear move.'}\n"
            f"Price will likely reverse back toward VWAP within\n"
            f"the first 15-30 minutes, then chop near VWAP\n"
            f"for the rest of the day.\n\n"
            f"What to watch at open:\n"
            f"→ Price fails to hold {'above' if gap_dir=='UP' else 'below'} VWAP within 5-15min\n"
            f"→ Volume thins out after open spike\n"
            f"→ First candle shows reversal wick\n\n"
            f"Trade:\n"
            f"{'→ PUTS on VWAP cross. Target: VWAP. Exit there — do not hold expecting full fill.' if gap_dir=='UP' else '→ CALLS on VWAP cross. Target: VWAP. Exit there — do not hold expecting full fill.'}\n"
            f"After reaching VWAP: NO new trades.\n"
            f"Market pins here rest of day — theta destroys premium.\n\n"
            f"Target: ${prev_cl} (prev close / VWAP zone)\n"
            f"Stop: price holds {'above' if gap_dir=='UP' else 'below'} VWAP for 2+ candles."
        ),

        "FULL_FADE": (
            f"🔄 GAP {dir_word} — FULL FADE SETUP\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Gap: {dir_emoji} ${gap_size:.2f} "
            f"(open ${open_px} vs prev close ${prev_cl})\n"
            f"Conviction: {conviction}%\n\n"
            f"What this means:\n"
            f"The gap opens {'up' if gap_dir=='UP' else 'down'} but every signal\n"
            f"says institutions are fading this move hard.\n"
            f"Price is expected to retrace the ENTIRE gap\n"
            f"back to yesterday's close (${prev_cl}) within the session.\n\n"
            f"Why this happens:\n"
            f"Institutions used the gap to EXIT positions.\n"
            f"{'Gap up = they sold into strength. Once they' + chr(39) + 're out, price collapses.' if gap_dir=='UP' else 'Gap down = they bought into weakness. Once they' + chr(39) + 're in, price recovers.'}\n\n"
            f"What to watch at open:\n"
            f"→ Price immediately {'sells below' if gap_dir=='UP' else 'pops above'} VWAP within 5min\n"
            f"→ Strong volume on reversal candles\n"
            f"→ Vol GEX accelerating {'negative' if gap_dir=='UP' else 'positive'}\n\n"
            f"Trade:\n"
            f"{'→ PUTS aggressively. Enter on any 5min candle that closes below VWAP.' if gap_dir=='UP' else '→ CALLS aggressively. Enter on any 5min candle that closes above VWAP.'}\n"
            f"Target: ${prev_cl:.2f} (full gap fill)\n"
            f"Stop: holds {'above' if gap_dir=='UP' else 'below'} open price ${open_px} for 15min.\n\n"
            f"⚡ High conviction fade. Size up on confirmation."
        ),

        "GAP_AND_REVERSE": (
            f"⚡ GAP {dir_word} — REVERSAL SETUP (RARE)\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Gap: {dir_emoji} ${gap_size:.2f} "
            f"(open ${open_px} vs prev close ${prev_cl})\n"
            f"Conviction: {conviction}%\n\n"
            f"What this means:\n"
            f"Gap opens {'up' if gap_dir=='UP' else 'down'} but signals are so\n"
            f"overwhelmingly against the gap that price is expected\n"
            f"to not just fill the gap but CONTINUE PAST prev close.\n"
            f"This is the rarest gap type — usually triggered by\n"
            f"a macro shock or surprise institutional positioning.\n\n"
            f"Wait for:\n"
            f"→ First 15 minutes to confirm direction\n"
            f"→ Vol GEX confirming {'negative' if gap_dir=='UP' else 'positive'}\n"
            f"→ Price through prev close ${prev_cl:.2f}\n\n"
            f"⚠️ Do NOT rush entry. Wait for 15min confirmation.\n"
            f"This setup can also trap both sides — patience required."
        ),

        "STATIC": (
            f"⚪ STATIC OPEN — NO GAP TRADE\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Gap: ${gap_size:.2f} (minimal)\n\n"
            f"What this means:\n"
            f"No meaningful overnight gap. Market opens near\n"
            f"yesterday's close. No institutional repositioning\n"
            f"overnight. Price will likely oscillate near open.\n\n"
            f"→ Skip the gap trade entirely.\n"
            f"Wait for the 7:45-8:30am GEX regime signal\n"
            f"to tell you if a directional trade sets up."
        ),

        "UNKNOWN": (
            f"❓ GAP {dir_word} — INSUFFICIENT DATA\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Gap: {dir_emoji} ${gap_size:.2f}\n\n"
            f"Cannot classify gap — Vol GEX open data unavailable.\n"
            f"→ Treat as FADE_THEN_STATIC until more data arrives.\n"
            f"Check the 7:00am GEX reading for confirmation."
        ),
    }

    base = type_blocks.get(gap_type, type_blocks["UNKNOWN"])
    footer = (f"\n\n📊 WHY THIS CLASSIFICATION:\n{detail[:600]}"
              if detail and gap_type not in ("STATIC", "UNKNOWN") else "")
    return base + footer


def check_gap_fill(current_price):
    """
    During the session, monitors whether gap is filling.
    Fires a simple progress alert when price is within
    50% of completing the gap fill.
    Separate from classify_gap which runs once at open.
    """
    try:
        prev_close = state.get("prev_session_close")
        open_price = state.get("open_price")
        if not prev_close or not open_price:
            return False, ""
        gap = round(open_price - prev_close, 2)
        if abs(gap) < 1.0:
            return False, ""
        gap_dir = "UP" if gap > 0 else "DOWN"
        dist_to_fill = abs(current_price - prev_close)
        half_gap = abs(gap) * 0.5
        if dist_to_fill <= half_gap:
            gap_type = state.get("gap_type", "UNKNOWN")
            if gap_dir == "UP" and current_price < open_price:
                return True, (
                    f"📉 GAP FILL IN PROGRESS — SPY\n"
                    f"Gap {gap_dir} ${abs(gap):.2f} | Type: {gap_type}\n"
                    f"Price ${current_price:.2f} heading toward fill target ${prev_close:.2f}\n"
                    f"${dist_to_fill:.2f} remaining\n"
                    f"→ {'On track per classification' if gap_type in ('FULL_FADE','FADE_THEN_STATIC') else 'Unexpected reversal — reassess'}"
                )
            elif gap_dir == "DOWN" and current_price > open_price:
                return True, (
                    f"📈 GAP FILL IN PROGRESS — SPY\n"
                    f"Gap {gap_dir} ${abs(gap):.2f} | Type: {gap_type}\n"
                    f"Price ${current_price:.2f} heading toward fill target ${prev_close:.2f}\n"
                    f"${dist_to_fill:.2f} remaining\n"
                    f"→ {'On track per classification' if gap_type in ('FULL_FADE','FADE_THEN_STATIC') else 'Unexpected reversal — reassess'}"
                )
        return False, ""
    except Exception as e:
        print(f"Gap fill error: {e}")
        return False, ""

# ─────────────────────────────────────────────
# MODULE 13: VOL GEX VELOCITY ALERT
# ─────────────────────────────────────────────
def check_vol_gex_velocity(vol_gex):
    """
    Dedicated alert for Vol GEX rate of change.
    Catches regime flips before they complete.
    The single most predictive signal for early entries.
    """
    try:
        history = state.get("vol_gex_history", [])
        if len(history) < 3:
            return
        recent = history[-3:]
        # Rate of change over last 3 readings
        roc_total = recent[-1] - recent[0]
        roc_recent= recent[-1] - recent[-2]
        prev_vel  = state.get("last_vol_gex_velocity", 0)
        state["last_vol_gex_velocity"] = roc_recent
        # Already sent and nothing dramatic changed
        if state.get("vol_gex_velocity_alert_sent"):
            # Reset if velocity reversed
            if (prev_vel > 0 and roc_recent < 0) or (prev_vel < 0 and roc_recent > 0):
                state["vol_gex_velocity_alert_sent"] = False
            else:
                return
        # Threshold: velocity equivalent to $50M+ move per reading
        threshold = 50e6 * 6.31 * 650  # approx $50M notional in raw GEX units
        if abs(roc_total) < threshold:
            return
        now_str = now_pdt().strftime("%H:%M")
        pct_chg = round((abs(roc_total) / abs(recent[0])) * 100, 1) if recent[0] != 0 else 0
        if roc_total > 0:
            direction = "IMPROVING (less negative → bullish)"
            emoji = "📈"
            implication = "Vol GEX acceleration bullish → Hedge unwind may be starting"
        else:
            direction = "ACCELERATING NEGATIVE (more bearish)"
            emoji = "📉"
            implication = "Vol GEX acceleration bearish → Institutions adding puts fast"
        alert(
            f"{emoji} VOL GEX VELOCITY ALERT — SPY\n"
            f"{'─'*35}\n"
            f"{now_str} PDT\n\n"
            f"Direction: {direction}\n"
            f"Rate of change: {pct_chg}% over last 3 readings\n"
            f"Current: {round(recent[-1]/1e9,2)}B | Was: {round(recent[0]/1e9,2)}B\n\n"
            f"💡 {implication}\n\n"
            f"⚠️ NOT confirmed yet — watch next reading for continuation"
        )
        state["vol_gex_velocity_alert_sent"] = True
        print(f"Vol GEX velocity alert: {direction} at {now_str}")
    except Exception as e:
        print(f"Vol GEX velocity error: {e}")

# ─────────────────────────────────────────────
# MODULE 14: OVERNIGHT MONITORING
# ─────────────────────────────────────────────
def fetch_overnight_data():
    result = {
        "futures_direction": "FLAT", "futures_change_pct": 0.0, "futures_price": None,
        "vix_current": None, "vix_change": None, "vix_direction": "STABLE",
        "news_sentiment": "NEUTRAL", "catalyst_type": "NONE",
        "catalyst_strength": 0, "macro_override": "NO",
        "overnight_news_flag": "NONE",
    }
    try:
        fut_dir, fut_chg, fut_price = fetch_futures_direction()
        result["futures_direction"]  = fut_dir
        result["futures_change_pct"] = fut_chg
        result["futures_price"]      = fut_price
    except Exception as e:
        print(f"Overnight futures error: {e}")
    try:
        vix = yf.Ticker("^VIX").history(period="2d", interval="5m")
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vc = float(vix["Close"].iloc[-1])
            result["vix_current"] = round(vc, 2)
            vix_close = state.get("overnight_vix_close")
            if vix_close:
                chg = round(vc - vix_close, 2)
                result["vix_change"] = chg
                result["vix_direction"] = ("RISING" if chg > 1.5
                                           else "FALLING" if chg < -1.5
                                           else "STABLE")
    except Exception as e:
        print(f"Overnight VIX error: {e}")
    sent, ns, cat, strength, override = fetch_news_sentiment()
    result["news_sentiment"]   = sent
    result["catalyst_type"]    = cat
    result["catalyst_strength"]= strength
    result["macro_override"]   = override
    result["overnight_news_flag"] = ("MAJOR_EVENT" if override == "YES" and strength >= 70
                                     else "MINOR" if strength >= 40
                                     else "NONE")
    return result

def run_overnight_check():
    if not is_overnight_window():
        return
    pdt = now_pdt()
    now_epoch = time.time()
    h = pdt.hour
    if state.get("overnight_alerts_today", 0) >= 3:
        return
    if now_epoch - state.get("last_overnight_check", 0) < 7200:
        return
    try:
        overnight = fetch_overnight_data()
        fut_dir  = overnight.get("futures_direction", "FLAT")
        fut_chg  = overnight.get("futures_change_pct", 0) or 0
        vix_curr = overnight.get("vix_current")
        vix_chg  = overnight.get("vix_change") or 0
        vix_dir  = overnight.get("vix_direction", "STABLE")
        news_flag= overnight.get("overnight_news_flag", "NONE")
        cat      = overnight.get("catalyst_type", "NONE")
        strength = overnight.get("catalyst_strength", 0)
        sent     = overnight.get("news_sentiment", "NEUTRAL")
        sig_futures = abs(fut_chg) >= 1.0
        sig_vix     = abs(vix_chg) >= 3.0
        major_event = news_flag == "MAJOR_EVENT" and strength >= 70
        evening_summary = (19 <= h <= 20 and not state.get("overnight_report_sent"))
        if not (sig_futures or sig_vix or major_event or evening_summary):
            state["last_overnight_check"] = now_epoch
            return
        now_str   = pdt.strftime("%H:%M")
        alert_type= "major_overnight_event" if major_event else "overnight_update"
        written   = write_overnight_alert_with_claude(overnight, alert_type)
        fut_emoji = "🟢" if fut_dir == "UP" else "🔴" if fut_dir == "DOWN" else "⚪"
        vix_emoji = "😨" if vix_dir == "RISING" else "😌" if vix_dir == "FALLING" else "😐"
        event_emoji = "🚨" if major_event else "🌙"
        futures_str = f"{fut_emoji} ES Futures: {fut_dir} {fut_chg:+.2f}%"
        if overnight.get("futures_price"):
            futures_str += f" (${overnight['futures_price']})"
        vix_str = f"{vix_emoji} VIX: {vix_curr} ({vix_dir})"
        if vix_chg and state.get("overnight_vix_close"):
            vix_str += f" {vix_chg:+.2f} since close"
        cat_str = (f"📰 {cat} | {sent} | Strength: {strength}/100"
                   if cat != "NONE" else "📰 No major catalyst")
        if written:
            alert(f"{event_emoji} OVERNIGHT UPDATE — SPY\n{'─'*35}\n{now_str} PDT\n\n"
                  f"{futures_str}\n{vix_str}\n{cat_str}\n\n{written}")
        else:
            bias = ("BULLISH LEAN" if fut_dir == "UP" and sent != "BEARISH"
                    else "BEARISH LEAN" if fut_dir == "DOWN" or (sent == "BEARISH" and strength >= 50)
                    else "NEUTRAL")
            alert(f"{event_emoji} OVERNIGHT UPDATE — SPY\n{'─'*35}\n{now_str} PDT\n\n"
                  f"{futures_str}\n{vix_str}\n{cat_str}\n\n"
                  f"🌅 Tomorrow's bias: {bias}\n"
                  f"⚠️ Confirm with 6:30am GEX signal before trading.")
        state["last_overnight_check"] = now_epoch
        state["overnight_alerts_today"] = state.get("overnight_alerts_today", 0) + 1
        if evening_summary:
            state["overnight_report_sent"] = True
        log_overnight_reading(overnight)
    except Exception as e:
        print(f"Overnight check error: {e}")

def log_overnight_reading(overnight):
    try:
        pdt = now_pdt()
        row = {h: "" for h in LOG_HEADERS}
        row["date"] = pdt.strftime("%Y-%m-%d")
        row["time"] = pdt.strftime("%H:%M")
        row["price"] = overnight.get("futures_price") or ""
        row["vix"]   = overnight.get("vix_current") or ""
        row["gex_state"] = "OVERNIGHT"
        row["regime"]    = "OVERNIGHT"
        row["session_type"]       = "OVERNIGHT"
        row["futures_direction"]  = overnight.get("futures_direction", "")
        row["overnight_vix_move"] = overnight.get("vix_change") or ""
        row["overnight_news_flag"]= overnight.get("overnight_news_flag", "")
        row["news_sentiment"]     = overnight.get("news_sentiment", "")
        row["catalyst_type"]      = overnight.get("catalyst_type", "")
        row["catalyst_strength"]  = overnight.get("catalyst_strength", "")
        row["macro_override"]     = overnight.get("macro_override", "")
        row["time_of_day"]        = "OVERNIGHT"
        with open(LOG_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_HEADERS).writerow(row)
        git_commit_log(reason="overnight")
        print(f"📝 Overnight logged: {row['time']}")
    except Exception as e:
        print(f"Overnight log error: {e}")

# ─────────────────────────────────────────────
# AI VERIFICATION LAYER
# ─────────────────────────────────────────────
def verify_signal_with_claude(signal_type, price, gex_state, regime,
                               vol_gex, oi_gex, ratio, vix, vvix,
                               news_sentiment, catalyst_type,
                               macro_override, conviction_score,
                               unwind_score, vanna_target, charm_target,
                               cycle_phase="UNKNOWN"):
    if not anthropic_client:
        return "UNAVAILABLE", 0, "No API key configured", conviction_score
    try:
        vol_b = round(vol_gex / 1e9, 2)
        oi_b  = round(oi_gex / 1e9, 2)
        prompt = f"""You are an expert options flow analyst reviewing a SPY 0DTE/1DTE trading signal.

CURRENT MARKET DATA:
- SPY Price: ${price}
- GEX State: {gex_state} | Regime: {regime}
- Vol GEX: {vol_b}B | OI GEX: {oi_b}B | Ratio: {ratio:.2f}x
- VIX: {vix} | VVIX: {vvix}
- News: {news_sentiment} | Catalyst: {catalyst_type} | Override: {macro_override}
- Unwind Score: {unwind_score}/100
- Vanna Target: ${vanna_target or 'None'} | Charm: ${charm_target or 'None'}
- OPEX Cycle Phase: {cycle_phase}
- Bot Conviction: {conviction_score}/100

SIGNAL: {signal_type}

Analysis rules:
1. macro_override YES + GEO/FED catalyst → reduce confidence in structure signals
2. Vol GEX + OI GEX same sign + ratio > 1.5x → directionally confirmed
3. VVIX > 100 → velocity conditions support premium expansion
4. Unwind score > 40 → mechanical bullish pressure regardless of structure
5. OPEX_WEEK/OPEX_DAY cycle phase → amplify conviction for high-score signals
6. Contradictions between news and GEX → higher risk

Respond in this exact JSON format only:
{{
  "verdict": "CONFIRM" or "CHALLENGE" or "NEUTRAL",
  "confidence": <integer 0-100>,
  "reasoning": "<2-3 sentences max>",
  "risk_factor": "<single biggest risk>"
}}"""
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        verdict    = result.get("verdict", "NEUTRAL")
        confidence = int(result.get("confidence", 50))
        reasoning  = result.get("reasoning", "")
        risk       = result.get("risk_factor", "")
        if verdict == "CONFIRM":
            combined = min(100, int((conviction_score * 0.6) + (confidence * 0.4) * 1.15))
        elif verdict == "CHALLENGE":
            combined = max(0, int((conviction_score * 0.4) + ((100 - confidence) * 0.2) * 0.7))
        else:
            combined = conviction_score
        full_reasoning = f"{reasoning} Risk: {risk}"
        print(f"🤖 Claude: {verdict} ({confidence}%) | Combined: {combined}")
        return verdict, confidence, full_reasoning, combined
    except Exception as e:
        print(f"Claude verification error: {e}")
        return "UNAVAILABLE", 0, str(e)[:100], conviction_score

# ─────────────────────────────────────────────
# CLAUDE ALERT WRITER
# ─────────────────────────────────────────────
def write_alert_with_claude(alert_type, price, gex_state, regime,
                             vol_gex, oi_gex, ratio, vix, vvix,
                             conviction, combined, unwind_score,
                             vanna_target, charm_target, news_sentiment,
                             catalyst_type, macro_override, flow_dir,
                             previous_regime, previous_gex_state,
                             claude_verdict, extra_context=""):
    if not anthropic_client:
        return None
    try:
        vol_b = round(vol_gex / 1e9, 2)
        oi_b  = round(oi_gex / 1e9, 2)
        now_str = now_pdt().strftime("%H:%M")
        prompts = {
            "morning_report": f"""You are writing a pre-market briefing for a SPY 0DTE/1DTE options trader in California.
Time: {now_str} PDT | SPY: ${price}

Market data:
- Regime: {regime} (was {previous_regime}) | GEX: {gex_state}
- Vol: {vol_b}B | OI: {oi_b}B | Ratio: {ratio:.2f}x
- VIX: {vix} | VVIX: {vvix}
- Conviction: {conviction}/100 | AI: {combined}/100
- News: {news_sentiment} | Catalyst: {catalyst_type} | Override: {macro_override}
- Vanna: ${vanna_target or 'none'} | Unwind: {unwind_score}/100
{extra_context}

Rules: max 10 lines, plain English, no jargon, no headers, no bullets.
If historical data is provided above, reference it (e.g. "this regime has been correct X%").
Tell them what to expect and why. One clear recommendation. Name biggest risk.
End with: size and when to enter. Write like a trader to a friend. 1-2 emojis max.""",

            "regime_transition": f"""You are writing a trading alert for a SPY 0DTE/1DTE options trader.
Time: {now_str} PDT | SPY: ${price}

Change: {previous_regime} → {regime}
GEX: {gex_state} | Vol: {vol_b}B | OI: {oi_b}B | Ratio: {ratio:.2f}x
VIX: {vix} | VVIX: {vvix} | Conviction: {conviction}/100 | AI: {combined}/100
News: {news_sentiment} | Catalyst: {catalyst_type} | Override: {macro_override}
Vanna: ${vanna_target or 'none'} | AI verdict: {claude_verdict}
{extra_context}

Rules: max 8 lines, no bullets, no headers.
Line 1: what changed and why it matters NOW.
Lines 2-3: mechanical explanation in plain English.
Lines 4-5: exactly what to do — calls/puts, entry, target, stop.
Line 6: the one thing that kills this trade. 1-2 emojis max.""",

            "hedge_unwind": f"""You are writing a hedge unwind alert for a SPY 0DTE/1DTE options trader.
Time: {now_str} PDT | SPY: ${price}

Unwind score: {unwind_score}/100 | Regime: {regime}
Vol GEX: {vol_b}B | Vanna target: ${vanna_target or 'none'}
VIX: {vix} | VVIX: {vvix} | Conviction: {conviction}/100
News: {news_sentiment} | Catalyst: {catalyst_type} | AI verdict: {claude_verdict}
{extra_context}

Rules: max 7 lines. Explain why price is rising in plain English (institutions closing puts forces MMs to buy shares). Specific call target. Exit zone. One killer risk. No jargon. 1-2 emojis.""",

            "eod_summary": f"""You are writing an end of day summary for a SPY 0DTE/1DTE options trader.
{extra_context}

Rules: max 8 lines. What happened today in plain English. Was the signal right or wrong and why. What to watch for tomorrow. Any overnight risks. Be honest — if wrong, say so. 1-2 emojis. Last line: tomorrow's one-sentence bias.""",
        }
        prompt = prompts.get(alert_type, prompts["regime_transition"])
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        print(f"✍️ Claude wrote {alert_type} ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"Claude alert writer error: {e}")
        return None

def write_overnight_alert_with_claude(overnight, alert_type="overnight_update"):
    if not anthropic_client:
        return None
    try:
        pdt = now_pdt()
        fut_dir = overnight.get("futures_direction", "FLAT")
        fut_chg = overnight.get("futures_change_pct", 0)
        vix_cur = overnight.get("vix_current", "N/A")
        vix_chg = overnight.get("vix_change", 0) or 0
        vix_dir = overnight.get("vix_direction", "STABLE")
        cat     = overnight.get("catalyst_type", "NONE")
        strength= overnight.get("catalyst_strength", 0)
        sent    = overnight.get("news_sentiment", "NEUTRAL")
        override= overnight.get("macro_override", "NO")
        prompt = f"""You are writing an overnight update for a SPY 0DTE/1DTE options trader in California.
Time: {pdt.strftime('%H:%M')} PDT ({pdt.strftime('%A')})

Overnight data:
- ES Futures: {fut_dir} {fut_chg:+.2f}%
- VIX: {vix_cur} ({vix_dir}, change: {vix_chg:+.2f} since close)
- Catalyst: {cat} (strength: {strength}/100) | News: {sent} | Override: {override}

Rules: max 8 lines. What are institutions doing and why. Is fear building or leaving.
What does futures tell you about tomorrow's open. One thing to watch at 6:30am.
Plain English. 1-2 emojis. End with: tomorrow's early bias (BULLISH/BEARISH/NEUTRAL)."""
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6", max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Overnight Claude writer error: {e}")
        return None

# ─────────────────────────────────────────────
# INTRADAY FEATURES
# ─────────────────────────────────────────────
def get_intraday_features(price, vol_gex):
    try:
        vwap       = state.get("session_vwap")
        open_price = state.get("open_price")
        s_high     = state.get("session_high") or price
        s_low      = state.get("session_low")  or price
        s_high     = max(s_high, price); s_low = min(s_low, price)
        state["session_high"] = s_high; state["session_low"] = s_low
        history = state.get("vol_gex_history", [])
        velocity = vol_gex_dir = 0
        if len(history) >= 2:
            velocity    = round(vol_gex - history[-2], 4)
            vol_gex_dir = ("ACCELERATING" if abs(vol_gex) > abs(history[-2])
                           else "DECELERATING")
        else:
            vol_gex_dir = "STABLE"
        h = now_pdt().hour
        return {
            "vwap_distance": round(price - vwap, 2) if vwap else 0,
            "price_vs_open": round(price - open_price, 2) if open_price else 0,
            "session_range": round(s_high - s_low, 2),
            "vol_gex_velocity": velocity,
            "vol_gex_direction": vol_gex_dir,
            "regime_transitions": state.get("regime_transitions_today", 0),
            "vwap_breaks": state.get("vwap_breaks_today", 0),
            "gamma_wall_above": state.get("gamma_wall_above", ""),
            "gamma_wall_below": state.get("gamma_wall_below", ""),
            "time_of_day": "EARLY" if h < 8 else "MID" if h < 11 else "LATE",
        }
    except Exception as e:
        print(f"Intraday features error: {e}")
        return {k: "" for k in [
            "vwap_distance","price_vs_open","session_range",
            "vol_gex_velocity","vol_gex_direction","regime_transitions",
            "vwap_breaks","gamma_wall_above","gamma_wall_below","time_of_day"
        ]}

# ─────────────────────────────────────────────
# LOG READING
# ─────────────────────────────────────────────
def log_reading(price, oi_gex, vol_gex, oi_m, vol_m, ratio, gex_state,
                regime, conv, grade, vix_spot, vvix_val, vix_term,
                tick_approx, inv_bias, unwind_score, open_drive,
                vt, ct, cal_flags, days_opex, cycle_phase="UNKNOWN",
                claude_verdict="", claude_confidence=0,
                claude_reasoning="", combined_score=0):
    try:
        now = now_pdt()

        # ── DEDUP GATE ────────────────────────────────────────────────
        # Prevents duplicate rows when 5-min jobs (check_telegram_commands,
        # check_vwap, etc.) trigger log_reading between scheduled run_job calls.
        # Skips write if all 5 key fields are identical to the last logged row.
        current_minute = now.strftime("%H:%M")
        last = state.get("last_logged_row", {})
        if last:
            if (last.get("time")             == current_minute and
                    last.get("gex_state")    == gex_state and
                    last.get("regime")       == regime and
                    abs(float(last.get("price", 0)) - price) < 0.10 and
                    last.get("conviction_score") == str(conv)):
                print(f"⏭  Dedup skip: {current_minute} | {gex_state} | ${round(price,2)}")
                return
        # ─────────────────────────────────────────────────────────────

        # FIX: fetch news HERE so both log and AI verification use same data
        news_sentiment, news_score, catalyst_type, catalyst_strength, macro_override = \
            fetch_news_sentiment()
        # Update state cache immediately so AI verification below gets fresh values
        state["last_news_sentiment"] = news_sentiment
        state["last_catalyst_type"]  = catalyst_type
        state["last_macro_override"] = macro_override
        intraday = get_intraday_features(price, vol_gex)
        cal_summary = (
            "QUARTER_END" if "QUARTER END TODAY" in str(cal_flags) else
            "OPEX_DAY"    if "OPEX DAY" in str(cal_flags) else
            f"OPEX_IN_{days_opex}D" if days_opex else "NORMAL"
        )
        clean_grade = (grade.replace("🔥","").replace("✅","")
                       .replace("⚠️","").replace("🔴","").replace("❌","").strip())
        row = {
            "date": now.strftime("%Y-%m-%d"),
            "time": current_minute,
            "price": round(price, 2),
            "oi_gex_raw": round(oi_gex / 1e9, 4),
            "vol_gex_raw": round(vol_gex / 1e9, 4),
            "oi_gex_m": round(oi_m, 2),
            "vol_gex_m": round(vol_m, 2),
            "ratio": round(ratio, 2),
            "gex_state": gex_state,
            "regime": regime,
            "conviction_score": conv,
            "grade": clean_grade,
            "vix": round(vix_spot, 2) if vix_spot else "",
            "vvix": round(vvix_val, 2) if vvix_val else "",
            "vix_term": vix_term or "",
            "tick_approx": tick_approx,
            "inventory_bias": inv_bias,
            "unwind_score": unwind_score,
            "open_drive": "YES" if open_drive else "NO",
            "vanna_target": vt or "",
            "charm_target": ct or "",
            "calendar_flags": cal_summary,
            "days_to_opex": days_opex or "",
            "opex_cycle_phase": cycle_phase,
            "vwap_distance": intraday["vwap_distance"],
            "price_vs_open": intraday["price_vs_open"],
            "session_range": intraday["session_range"],
            "vol_gex_velocity": intraday["vol_gex_velocity"],
            "vol_gex_direction": intraday["vol_gex_direction"],
            "regime_transitions": intraday["regime_transitions"],
            "vwap_breaks": intraday["vwap_breaks"],
            "gamma_wall_above": intraday["gamma_wall_above"],
            "gamma_wall_below": intraday["gamma_wall_below"],
            "time_of_day": intraday["time_of_day"],
            "news_sentiment": news_sentiment,
            "news_score": news_score,
            "catalyst_type": catalyst_type,
            "catalyst_strength": catalyst_strength,
            "macro_override": macro_override,
            "session_type": "MARKET",
            "futures_direction": state.get("gap_direction", ""),
            "overnight_vix_move": "",
            "overnight_news_flag": "",
            "gap_direction": state.get("gap_direction", ""),
            "gap_size": state.get("gap_size", ""),
            "gap_type": state.get("gap_type", "UNKNOWN"),
            "gap_conviction": state.get("gap_conviction", 0),
            "vol_gex_velocity_alert": "YES" if state.get("vol_gex_velocity_alert_sent") else "NO",
            "outcome_direction": "",
            "outcome_points": "",
            "signal_correct": "",
            "max_move_up": "",
            "max_move_down": "",
            "claude_verdict": claude_verdict,
            "claude_confidence": claude_confidence,
            "claude_reasoning": claude_reasoning[:500] if claude_reasoning else "",
            "combined_score": combined_score,
            # Candle fields — populated only on 6:35am row via state,
            # blank on all other rows. Never breaks existing data.
            "open_candle_type":       state.get("open_candle_type") or "",
            "open_candle_body_pct":   state.get("open_candle_body_pct") or "",
            "open_candle_upper_wick": state.get("open_candle_upper_wick") or "",
            "open_candle_lower_wick": state.get("open_candle_lower_wick") or "",
            "open_candle_vol_ratio":  state.get("open_candle_vol_ratio") or "",
            "open_candle_vwap_pos":   state.get("open_candle_vwap_pos") or "",
            "open_candle_confluence": state.get("open_candle_confluence") or "",
            "notes": ""
        }
        with open(LOG_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_HEADERS).writerow(row)

        # Update dedup tracker
        state["last_logged_row"] = {
            "time":             current_minute,
            "gex_state":        gex_state,
            "regime":           regime,
            "price":            str(round(price, 2)),
            "conviction_score": str(conv),
        }

        print(f"📝 Logged: {current_minute} | {gex_state} | Score:{conv} | "
              f"News:{news_sentiment} | {catalyst_type} | Override:{macro_override}")
        git_commit_log(reason="reading")
    except Exception as e:
        print(f"log_reading error: {e}")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_gex_state(oi_gex, vol_gex):
    if oi_gex == 0: return "UNKNOWN"
    same = (oi_gex < 0 and vol_gex < 0) or (oi_gex > 0 and vol_gex > 0)
    if not same: return "COUNTER"
    r = abs(vol_gex) / abs(oi_gex)
    if r < 1.2: return "NEUTRAL"
    if r < 1.5: return "WATCH"
    return f"DIRECTIONAL_{'BEARISH' if oi_gex < 0 else 'BULLISH'}"

def format_gex(v):
    a = abs(v); s = "-" if v < 0 else ""
    return (f"{s}{a/1000:.1f}B" if a >= 1000 else
            f"{s}{a:.1f}M"     if a >= 1    else
            f"{s}{a*1000:.0f}K")

def fmt(v):
    a = abs(v); s = "-" if v < 0 else "+"
    return (f"{s}{a/1000:.1f}B" if a >= 1000 else
            f"{s}{a:.1f}M"     if a >= 1    else
            f"{s}{a*1000:.0f}K")

async def _send(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)

def alert(text):
    asyncio.run(_send(text))

def fetch_true_session_data():
    """Fetch real open/high/low/close/prev_close for today from yfinance."""
    try:
        spy = yf.download("SPY", period="5d", interval="1m", progress=False)
        if spy.empty: return None
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy.index = spy.index.tz_convert("America/Los_Angeles")
        today = now_pdt().strftime("%Y-%m-%d")
        today_bars = spy[spy.index.strftime("%Y-%m-%d") == today]
        market_bars = today_bars.between_time("06:30", "13:00")
        if market_bars.empty: return None
        # Get previous session close
        prev_bars = spy[spy.index.strftime("%Y-%m-%d") < today]
        prev_close = None
        if not prev_bars.empty:
            prev_session = prev_bars.between_time("06:30", "13:00")
            if not prev_session.empty:
                prev_close = round(float(prev_session["Close"].iloc[-1]), 2)
        return {
            "open":  round(float(market_bars["Open"].iloc[0]), 2),
            "high":  round(float(market_bars["High"].max()), 2),
            "low":   round(float(market_bars["Low"].min()), 2),
            "close": round(float(market_bars["Close"].iloc[-1]), 2),
            "prev_close": prev_close,
        }
    except Exception as e:
        print(f"fetch_true_session_data error: {e}")
        return None

def get_vwap():
    try:
        spy = yf.download("SPY", period="1d", interval="5m", progress=False)
        if spy.empty: return None, None, None, None, None
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy["vwap"] = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
        cp  = float(spy["Close"].iloc[-1])
        cv  = float(spy["vwap"].iloc[-1])
        pp  = float(spy["Close"].iloc[-2]) if len(spy) > 1 else cp
        pv  = float(spy["vwap"].iloc[-2])  if len(spy) > 1 else cv
        vol = float(spy["Volume"].iloc[-1])
        return cp, cv, pp, pv, vol
    except Exception:
        return None, None, None, None, None

# ─────────────────────────────────────────────
# EOD AUTO-FILL
# ─────────────────────────────────────────────
def eod_autofill(close_price):
    try:
        today_str = now_pdt().strftime("%Y-%m-%d")
        if not os.path.exists(LOG_FILE): return
        with open(LOG_FILE, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows: return
        today_rows = [r for r in rows
                      if r["date"] == today_str and r.get("session_type","MARKET") == "MARKET"]
        # Use real session data seeded at startup, fall back to logged data
        open_price  = state.get("open_price")
        if open_price is None and today_rows:
            open_price = float(today_rows[0]["price"])
        if open_price is None:
            sd = fetch_true_session_data()
            if sd: open_price = sd["open"]
        if open_price is None:
            open_price = close_price
        true_close  = state.get("true_session_close") or close_price
        point_move  = round(true_close - open_price, 2)
        direction   = ("CHOP" if abs(point_move) < 1.0 else
                       "UP"   if point_move > 0 else "DOWN")
        morning_signal = None
        for r in today_rows:
            gs = r.get("gex_state", "")
            if "DIRECTIONAL" in gs: morning_signal = gs; break
            elif gs in ["NEUTRAL","WATCH","COUNTER"]: morning_signal = gs; break
        if morning_signal and "DIRECTIONAL" in morning_signal:
            correct = ("YES"     if ("BEARISH" in morning_signal and direction == "DOWN") or
                                    ("BULLISH" in morning_signal and direction == "UP")
                       else "PARTIAL" if direction == "CHOP" else "NO")
        elif morning_signal in ["NEUTRAL","WATCH","COUNTER"]:
            correct = "YES" if direction == "CHOP" else "PARTIAL"
        elif len(today_rows) <= 2:
            correct = "PARTIAL"
        else:
            correct = ""
        s_high = state.get("session_high") or true_close
        s_low  = state.get("session_low")  or true_close
        updated = 0
        for r in rows:
            if r["date"] == today_str and r["outcome_direction"] == "":
                r["outcome_direction"] = direction
                r["outcome_points"]    = point_move
                r["signal_correct"]    = correct
                r["max_move_up"]       = round(s_high - open_price, 2) if s_high else ""
                r["max_move_down"]     = round(open_price - s_low, 2)  if s_low  else ""
                updated += 1
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        state["overnight_vix_close"]  = state["vix_history"][-1] if state["vix_history"] else None
        state["overnight_gex_snapshot"]= {"oi": state.get("previous_oi_gex"),
                                           "vol": state.get("previous_vol_gex")}
        state["overnight_vol_gex_close"] = state.get("previous_vol_gex")
        git_commit_log(reason="eod")
        now_str = now_pdt().strftime("%H:%M")
        sig_emoji = "✅" if correct == "YES" else "❌" if correct == "NO" else "⚠️"
        max_up    = round(s_high - open_price, 2)
        max_down  = round(open_price - s_low, 2)
        mid_day_deploy = len(today_rows) <= 2
        eod_context = (
            f"Date: {today_str}\n"
            f"Open: ${round(open_price,2)} | Close: ${round(true_close,2)}\n"
            f"Move: {'+' if point_move>0 else ''}{point_move} pts ({direction})\n"
            f"Max up: +{max_up} | Max down: -{max_down}\n"
            f"Morning signal: {morning_signal or 'None'} | Correct: {correct}\n"
            f"Rows logged: {len(today_rows)}"
            + (" (⚠️ mid-day deploy)" if mid_day_deploy else "") +
            f"\n\n{load_historical_context(days=30)}"
        )
        written = write_alert_with_claude(
            alert_type="eod_summary",
            price=true_close, gex_state=state.get("previous_gex_state",""),
            regime=state.get("previous_regime",""),
            vol_gex=state.get("previous_vol_gex",0) or 0,
            oi_gex=state.get("previous_oi_gex",0) or 0,
            ratio=0, vix=0, vvix=0, conviction=0, combined=0,
            unwind_score=0, vanna_target=None, charm_target=None,
            news_sentiment=state.get("last_news_sentiment","NEUTRAL"),
            catalyst_type=state.get("last_catalyst_type","NONE"),
            macro_override=state.get("last_macro_override","NO"),
            flow_dir="", previous_regime="", previous_gex_state="",
            claude_verdict="", extra_context=eod_context
        )
        if written:
            alert(f"📊 END OF DAY — SPY\n{'─'*35}\n{today_str} | {now_str} PDT\n"
                  f"${round(open_price,2)} → ${round(true_close,2)} "
                  f"({'+' if point_move>0 else ''}{point_move} pts) {sig_emoji}"
                  + (f"\n⚠️ Mid-day deploy — {len(today_rows)} rows logged" if mid_day_deploy else "") +
                  f"\n\n{written}\n\n"
                  f"💾 {updated} rows saved | 💬 /notes | 🌙 Overnight monitoring active")
        else:
            alert(f"📊 END OF DAY — SPY\n{'─'*35}\n{today_str} | {now_str} PDT\n\n"
                  f"Open: ${round(open_price,2)} → Close: ${round(true_close,2)}\n"
                  f"Move: {'+' if point_move>0 else ''}{point_move} pts | {direction}\n"
                  f"Max Up: +{max_up} | Max Down: -{max_down}\n"
                  f"Signal: {morning_signal or 'None'} → {sig_emoji} {correct}\n\n"
                  f"📝 {updated} rows | 💾 GitHub | 💬 /notes\n🌙 Overnight monitoring active")
        print(f"EOD complete — {updated} rows | {direction} {point_move}pts | {correct}")
    except Exception as e:
        print(f"EOD autofill error: {e}")

# ─────────────────────────────────────────────
# VWAP CROSS
# ─────────────────────────────────────────────
def check_vwap():
    if not is_market_open(): return
    gex_s = state["previous_gex_state"] or ""
    if "DIRECTIONAL" not in gex_s:
        state["vwap_alert_sent"] = False
        return
    r = get_vwap()
    if r[0] is None: return
    cp, cv, pp, pv, _ = r
    now_str = now_pdt().strftime("%H:%M")
    bearish = "BEARISH" in gex_s
    bullish = "BULLISH" in gex_s
    if bearish and not state["vwap_alert_sent"] and pp >= pv and cp < cv:
        alert(f"🔽 VWAP CROSS — BEARISH ENTRY\n"
              f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
              f"Regime: {state['regime']} | {now_str} PDT\n\n"
              f"⚠️ Confirm candle close below VWAP before entering.")
        state["vwap_alert_sent"] = True
    elif bullish and not state["vwap_alert_sent"] and pp <= pv and cp > cv:
        alert(f"🔼 VWAP CROSS — BULLISH ENTRY\n"
              f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
              f"Regime: {state['regime']} | {now_str} PDT\n\n"
              f"⚠️ Confirm candle close above VWAP before entering.")
        state["vwap_alert_sent"] = True
    if (bearish and cp > cv) or (bullish and cp < cv):
        state["vwap_alert_sent"] = False

# ─────────────────────────────────────────────
# HEARTBEAT
# ─────────────────────────────────────────────
def check_heartbeat():
    if not is_market_open(): return
    try:
        now_epoch = time.time()
        if now_epoch - state.get("last_heartbeat", 0) < 3600: return
        r = get_vwap()
        if r[0] is None: return
        cp, cv, _, _, _ = r
        now_str  = now_pdt().strftime("%H:%M")
        vwap_d   = round(cp - cv, 2)
        vwap_s   = "above" if vwap_d > 0 else "below"
        alert(f"💓 BOT ALIVE — SPY\n{'─'*30}\n"
              f"{now_str} PDT | ${round(cp,2)}\n\n"
              f"Regime: {state.get('regime','UNKNOWN')}\n"
              f"GEX: {state.get('previous_gex_state','UNKNOWN')}\n"
              f"Score: {state.get('last_conviction_score',0)}/100\n"
              f"VWAP: ${round(cv,2)} (${abs(vwap_d):.2f} {vwap_s})\n\n"
              f"Bot running ✅")
        state["last_heartbeat"] = now_epoch
    except Exception as e:
        print(f"Heartbeat error: {e}")

# ─────────────────────────────────────────────
# DOJI TRANSITION DETECTOR
# ─────────────────────────────────────────────
def check_doji_transition():
    if not is_market_open(): return
    try:
        r = get_vwap()
        if r[0] is None: return
        cp, cv, _, _, _ = r
        if abs(cp - cv) > 0.75:
            state["doji_transition_sent"] = False; return
        history = state.get("vol_gex_history", [])
        if len(history) < 3: return
        rec = history[-3:]
        roc_r = abs(rec[-1]) - abs(rec[-2])
        roc_o = abs(rec[-2]) - abs(rec[-3])
        if not (roc_o < 0 and roc_r > roc_o): return
        if state.get("regime","") in ["HEDGE_UNWIND_CONFIRMED","BULLISH_MOMENTUM"]: return
        if state.get("doji_transition_sent"): return
        now_str = now_pdt().strftime("%H:%M")
        if rec[-1] < 0 and roc_r > 0:
            direction = "BEARISH → BULLISH"
            action = "Watch for Vol GEX flip positive → Calls on VWAP break above"
        elif rec[-1] > 0 and roc_r < 0:
            direction = "BULLISH → BEARISH"
            action = "Watch for Vol GEX flip negative → Puts on VWAP break below"
        else:
            direction = "CONSOLIDATING"
            action = "No clear direction — wait for Vol GEX commitment"
        alert(f"🔄 DOJI TRANSITION FORMING — SPY\n{'─'*35}\n"
              f"{now_str} PDT | ${round(cp,2)} | VWAP ${round(cv,2)}\n\n"
              f"Transition: {direction}\n"
              f"Vol GEX decelerating — momentum losing steam\n\n"
              f"Action: {action}\n\n"
              f"⚠️ NOT a trade signal yet — wait for Vol GEX to confirm")
        state["doji_transition_sent"] = True
    except Exception as e:
        print(f"Doji transition error: {e}")

# ─────────────────────────────────────────────
# GAMMA WALL / TP ALERT
# ─────────────────────────────────────────────
def check_gamma_wall_approach():
    if not is_market_open(): return
    try:
        r = get_vwap()
        if r[0] is None: return
        cp, _, _, _, _ = r
        vt = state.get("current_vanna_target")
        if not vt: return
        if abs(cp - state.get("last_wall_alert_price", 0)) < 2.0: return
        dist = vt - cp
        if abs(dist) > 1.5: return
        now_str = now_pdt().strftime("%H:%M")
        ct = state.get("current_charm_target")
        if dist > 0:
            alert(f"🎯 VANNA TARGET APPROACHING — SPY\n{'─'*35}\n"
                  f"{now_str} PDT | ${round(cp,2)} | Target: ${vt} (${abs(dist):.2f} away)\n\n"
                  f"💰 IF HOLDING CALLS — THIS IS YOUR EXIT ZONE\n"
                  f"Sell between ${round(vt-0.5,0)}-${vt}\n"
                  f"Charm reverses above here. Don't hold past target.\n\n"
                  f"After target: watch Vol GEX for put re-entry.")
        else:
            alert(f"🎯 SUPPORT ZONE APPROACHING — SPY\n{'─'*35}\n"
                  f"{now_str} PDT | ${round(cp,2)} | Support: ${vt} (${abs(dist):.2f} away)\n\n"
                  f"💰 IF HOLDING PUTS — Consider partial profit here\n"
                  f"Vanna support may cause bounce. Sell half, keep half.\n\n"
                  f"If wall breaks → hold remaining puts to next level.")
        state["last_wall_alert_price"] = cp
    except Exception as e:
        print(f"Gamma wall error: {e}")

# ─────────────────────────────────────────────
# CONSOLIDATION JOB
# ─────────────────────────────────────────────
def check_consolidation_job():
    if not is_market_open(): return
    pdt = now_pdt()
    mins = (pdt.hour - 6) * 60 + pdt.minute - 30
    if mins > 45 or mins < 5 or state["consolidation_alert_sent"]: return
    try:
        r = get_vwap()
        if r[0] is None: return
        cp, cv, _, _, vol = r
        if state["open_price"] is None:  state["open_price"] = cp
        if state["open_volume"] is None: state["open_volume"] = vol
        vix_d = yf.Ticker("^VIX").history(period="1d", interval="5m")
        iv = float(vix_d["Close"].iloc[-1]) if not vix_d.empty else None
        if state["open_iv"] is None and iv: state["open_iv"] = iv
        vt, vs, ct, cs, conflict = fetch_vanna_charm()
        is_cons, score, signals = run_consolidation_check(cp, iv, vol, vt, ct, conflict)
        if is_cons:
            sigs = "\n".join(signals)
            alert(f"🚨 CONSOLIDATION TRAP WARNING — SPY\n{'─'*35}\n"
                  f"{pdt.strftime('%H:%M')} PDT | ${cp:.2f} | Score: {score}/100\n\n"
                  f"⚠️ SIGNALS\n{sigs}\n\n"
                  f"Vanna + charm forces canceling = premium decays fast.\n\n"
                  f"🚫 DO NOT TRADE THIS OPEN\n"
                  f"Wait for: >1% move + volume OR TICK ±600 sustained 15min\n"
                  f"Re-assess after 10:00am PDT.")
            state["consolidation_alert_sent"] = True
            state["consolidation_gex_state"] = state.get("previous_gex_state", "")
    except Exception as e:
        print(f"Consolidation job error: {e}")

# ─────────────────────────────────────────────
# MAIN JOB
# ─────────────────────────────────────────────
def run_job():
    pdt = now_pdt()
    now_str = pdt.strftime("%H:%M")
    h, m = pdt.hour, pdt.minute
    print(f"\n{'='*60}\nJob: {now_str} PDT\n{'='*60}")
    try:
        oi_gex, vol_gex, price = fetch_gex()
        if oi_gex is None: print("No GEX data"); return
        if vol_gex == 0:
            print(f"Pre-market. OI: {round(oi_gex/1e9,2)}B | ${price}")
            # Pre-market: run futures check and log gap
            if h == 6 and m < 30:
                fut_dir, fut_chg, fut_price = fetch_futures_direction()
                if abs(fut_chg) >= 0.5:
                    emoji = "🟢" if fut_dir == "UP" else "🔴" if fut_dir == "DOWN" else "⚪"
                    alert(f"{emoji} PRE-MARKET FUTURES — SPY\n"
                          f"{'─'*30}\n"
                          f"{now_str} PDT\n\n"
                          f"ES Futures: {fut_dir} {fut_chg:+.2f}%"
                          + (f" (${fut_price})" if fut_price else "") +
                          f"\n\nGap at open: {'UP' if fut_chg > 0 else 'DOWN'} likely\n"
                          f"65% of gaps fill intraday — watch for reversal")
            return
        # Update Vol GEX history
        state["vol_gex_history"].append(vol_gex)
        if len(state["vol_gex_history"]) > 10:
            state["vol_gex_history"].pop(0)
        ratio   = abs(vol_gex) / abs(oi_gex) if oi_gex != 0 else 0
        gex_state = get_gex_state(oi_gex, vol_gex)
        oi_b    = round(oi_gex / 1e9, 2)
        vol_b   = round(vol_gex / 1e9, 2)
        oi_m    = oi_gex / (price * 6.31) / 1e6
        vol_m   = vol_gex / (price * 6.31) / 1e6
        oi_fmt  = format_gex(oi_m)
        vol_fmt = format_gex(vol_m)
        ratio_r = round(ratio, 2)
        # All intelligence modules
        vix_spot, vvix_val, vix_term, term_sig, vix_sig, vvix_sig = fetch_vix_data()
        vt, vs, ct, cs, conflict = fetch_vanna_charm()
        vc_text, vanna_window = get_vanna_charm_read(vt, vs, ct, cs, price, conflict)
        state["current_vanna_target"] = vt
        state["current_charm_target"] = ct
        # Update SPY liquidity zone — populates cache_levels for /levels command
        update_spy_liquidity_zone(price, vt, ct)
        unwind_det, unwind_score, unwind_sigs, flow_dir = fetch_hedge_unwind_signals()
        regime, reg_conf = detect_regime(oi_gex, vol_gex, state["vol_gex_history"])
        reg_signal = get_regime_signal(regime, reg_conf, oi_b, vol_b)
        cal_flags, cal_bonus, days_opex, cycle_phase = get_calendar_flags()
        tick_sig, tick_approx, inv_bias, open_drive = fetch_tick_and_inventory()
        conv, grade, rec, checklist = score_conviction(
            vix_spot, vvix_val, vix_term, vol_gex,
            state["previous_vol_gex"], regime, unwind_score,
            cal_bonus, vanna_window, conflict, ratio,
            tick_approx, inv_bias, open_drive
        )
        print(f"${price} | {gex_state} | {regime} | Score:{conv} | Phase:{cycle_phase}")
        # Vol GEX velocity check (dedicated alert)
        check_vol_gex_velocity(vol_gex)

        # ── NEWS FETCH — must run before gap classification AND AI verify ──
        # Gap classification needs news_sentiment, catalyst_type, macro_override.
        # AI verification needs the same. Fetch once here, reuse everywhere.
        news_sent, news_score, cat_type, cat_strength, mac_ovr = fetch_news_sentiment()
        state["last_news_sentiment"] = news_sent
        state["last_catalyst_type"]  = cat_type
        state["last_macro_override"] = mac_ovr

        # ── GAP CLASSIFICATION — fires once at 6:30am open ──────────
        # Snapshot Vol GEX at open for gap classification signal 1
        if h == 6 and 29 <= m <= 35 and state.get("open_vol_gex_snapshot") is None:
            state["open_vol_gex_snapshot"] = vol_gex
            print(f"📸 Vol GEX open snapshot: {round(vol_gex/1e9,2)}B")

        if (h == 6 and 29 <= m <= 45 and
                state.get("gap_direction", "NONE") != "NONE" and
                not state.get("gap_type_sent")):
            vix_chg = None
            if state.get("overnight_vix_close") and len(state.get("vix_history", [])) > 0:
                vix_chg = round(state["vix_history"][-1] - state["overnight_vix_close"], 2)
            g_type, g_conv, g_detail = classify_gap(
                vol_gex=vol_gex,
                vix_change=vix_chg,
                news_sentiment=news_sent,
                macro_override=mac_ovr,
                catalyst_type=cat_type,
                catalyst_strength=cat_strength,
                futures_chg=state.get("gap_size", 0) * (1 if state.get("gap_direction") == "UP" else -1),
            )
            gap_alert_text = build_gap_alert(
                g_type, g_conv, g_detail,
                vol_gex, vix_chg, news_sent, cat_type
            )
            alert(gap_alert_text)
            state["gap_type_sent"] = True
            print(f"🔲 Gap classified: {g_type} ({g_conv}%)")
        # ────────────────────────────────────────────────────────────
        # Gap fill check
        if is_market_open():
            gap_fill, gap_msg = check_gap_fill(price)
            if gap_fill and not state.get("gap_fill_alert_sent"):
                alert(gap_msg)
                state["gap_fill_alert_sent"] = True
        # AI verification
        should_verify = (
            conv >= 50 or
            regime in ["BEARISH_HEDGE_BUILD","HEDGE_UNWIND_CONFIRMED",
                       "HEDGE_UNWIND_EARLY","BULLISH_MOMENTUM"] or
            unwind_score >= 40 or gex_state != state["previous_gex_state"] or
            regime != state["previous_regime"]
        )
        claude_verdict = "SKIPPED"
        claude_conf    = 0
        claude_reason  = "Low conviction — verification skipped"
        combined_score = conv
        if should_verify and anthropic_client:
            signal_desc = (f"{gex_state} regime={regime} "
                           f"conviction={conv}/100 unwind={unwind_score}/100 "
                           f"phase={cycle_phase}")
            claude_verdict, claude_conf, claude_reason, combined_score = \
                verify_signal_with_claude(
                    signal_type=signal_desc, price=price,
                    gex_state=gex_state, regime=regime,
                    vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                    vix=vix_spot, vvix=vvix_val,
                    news_sentiment=news_sent, catalyst_type=cat_type,
                    macro_override=mac_ovr, conviction_score=conv,
                    unwind_score=unwind_score, vanna_target=vt, charm_target=ct,
                    cycle_phase=cycle_phase
                )
        v_emoji  = ("✅" if claude_verdict == "CONFIRM" else
                    "⚠️" if claude_verdict == "CHALLENGE" else
                    "➖" if claude_verdict == "NEUTRAL" else "🔇")
        ai_line  = (f"\n🤖 AI: {v_emoji} {claude_verdict} ({claude_conf}%)\n"
                    f"→ {claude_reason[:300]}\nCombined: {combined_score}/100"
                    if claude_verdict not in ["SKIPPED","UNAVAILABLE"] else "")
        # Log reading (news fetched again inside but state already updated above)
        log_reading(
            price=price, oi_gex=oi_gex, vol_gex=vol_gex,
            oi_m=oi_m, vol_m=vol_m, ratio=ratio,
            gex_state=gex_state, regime=regime,
            conv=conv, grade=grade,
            vix_spot=vix_spot, vvix_val=vvix_val, vix_term=vix_term,
            tick_approx=tick_approx, inv_bias=inv_bias,
            unwind_score=unwind_score, open_drive=open_drive,
            vt=vt, ct=ct, cal_flags=cal_flags, days_opex=days_opex,
            cycle_phase=cycle_phase,
            claude_verdict=claude_verdict, claude_confidence=claude_conf,
            claude_reasoning=claude_reason, combined_score=combined_score
        )
        # GEX state change alert
        if gex_state != state["previous_gex_state"]:
            emojis = {"NEUTRAL":"⚪","WATCH":"⚠️",
                      "DIRECTIONAL_BEARISH":"🔴","DIRECTIONAL_BULLISH":"🟢","COUNTER":"🔄"}
            alert(f"{emojis.get(gex_state,'❓')} SPY GEX SIGNAL\n\n"
                  f"State: {gex_state}\n"
                  f"OI: {oi_fmt} ({oi_b}B) | VOL: {vol_fmt} ({vol_b}B)\n"
                  f"Ratio: {ratio_r}x | ${price} | {now_str} PDT\n\n"
                  f"📊 REGIME: {regime} ({reg_conf}%)\n"
                  f"📅 OPEX Phase: {cycle_phase}\n"
                  f"🎯 CONVICTION: {conv}/100 — {grade}\n→ {rec}"
                  f"{ai_line}")
        # Regime change alert
        if regime != state["previous_regime"] and regime != "INSUFFICIENT_DATA":
            prev = state["previous_regime"] or "None"
            key_transitions = [
                ("BEARISH_HEDGE_BUILD","TRANSITION_ZONE"),
                ("BEARISH_HEDGE_BUILD","HEDGE_UNWIND_EARLY"),
                ("BEARISH_HEDGE_BUILD","HEDGE_UNWIND_CONFIRMED"),
                ("TRANSITION_ZONE","HEDGE_UNWIND_EARLY"),
                ("HEDGE_UNWIND_EARLY","HEDGE_UNWIND_CONFIRMED"),
            ]
            is_key = any(prev == a and regime == b for a, b in key_transitions)
            written = write_alert_with_claude(
                alert_type="regime_transition",
                price=price, gex_state=gex_state, regime=regime,
                vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                vix=vix_spot, vvix=vvix_val,
                conviction=conv, combined=combined_score,
                unwind_score=unwind_score, vanna_target=vt, charm_target=ct,
                news_sentiment=news_sent, catalyst_type=cat_type,
                macro_override=mac_ovr, flow_dir=flow_dir,
                previous_regime=prev, previous_gex_state=state.get("previous_gex_state",""),
                claude_verdict=claude_verdict
            )
            r_emoji = "🚨" if is_key else "🔄"
            if written:
                alert(f"{r_emoji} REGIME TRANSITION — SPY\n{'─'*35}\n"
                      f"{prev} → {regime} | {now_str} PDT | Phase:{cycle_phase}\n\n{written}")
            else:
                cl_text = "\n".join(checklist)
                alert(f"{r_emoji} REGIME TRANSITION — SPY\n{'─'*35}\n"
                      f"NEW: {regime} | WAS: {prev}\n{now_str} PDT | ${price}\n\n"
                      f"{reg_signal}\n\n📈 VANNA/CHARM\n{vc_text}\n\n"
                      f"🎯 {conv}/100 — {grade}\n→ {rec}\n\n"
                      f"📋 CHECKLIST\n{cl_text}{ai_line}")
            state["regime_transitions_today"] = state.get("regime_transitions_today", 0) + 1
        # Hedge unwind standalone
        elif (unwind_det and not state["hedge_unwind_alert_sent"]
              and unwind_score >= 40
              and regime == state["previous_regime"]):
            uw_text = "\n".join(unwind_sigs)
            written = write_alert_with_claude(
                alert_type="hedge_unwind",
                price=price, gex_state=gex_state, regime=regime,
                vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                vix=vix_spot, vvix=vvix_val,
                conviction=conv, combined=combined_score,
                unwind_score=unwind_score, vanna_target=vt, charm_target=ct,
                news_sentiment=news_sent, catalyst_type=cat_type,
                macro_override=mac_ovr, flow_dir=flow_dir,
                previous_regime=state.get("previous_regime",""),
                previous_gex_state=state.get("previous_gex_state",""),
                claude_verdict=claude_verdict,
                extra_context=f"Flow signals:\n{uw_text}"
            )
            if written:
                alert(f"🚀 HEDGE UNWIND — SPY\n{'─'*35}\n"
                      f"{now_str} PDT | ${price} | Score: {unwind_score}/100\n\n{written}")
            else:
                alert(f"🚀 HEDGE UNWIND — SPY\n{'─'*35}\n"
                      f"Score: {unwind_score}/100 | {now_str} PDT | ${price}\n\n"
                      f"📊 FLOW\n{uw_text}\n\n"
                      f"Institutions closing puts = MMs buying shares back.\n"
                      f"📈 VANNA TARGET: ${vt}\n\n"
                      f"🎯 {conv}/100 — {grade}\n→ {rec}{ai_line}")
            state["hedge_unwind_alert_sent"] = True
            state["last_unwind_alert_time"] = time.time()
        if unwind_score < 20:
            state["hedge_unwind_alert_sent"] = False
        # Conviction spike
        if ("DIRECTIONAL" in gex_state and state["previous_ratio"]
                and ratio - state["previous_ratio"] > 0.3):
            dir_ = "BEARISH" if oi_gex < 0 else "BULLISH"
            alert(f"📈 CONVICTION INCREASING — SPY\n\n"
                  f"Direction: {dir_} | Ratio: {ratio_r}x (was {round(state['previous_ratio'],2)}x)\n"
                  f"OI: {oi_fmt} | VOL: {vol_fmt} | ${price} | {now_str} PDT\n\n"
                  f"VIX: {vix_sig} | VVIX: {vvix_sig}\n"
                  f"Phase: {cycle_phase}\n"
                  f"🎯 {conv}/100 — {grade}\n→ {rec}")
        # Morning brief
        if (h == 6 and m >= 25) or (h == 7 and m <= 15):
            conv_changed = (state["last_conviction_score"] is not None and
                            abs(conv - state["last_conviction_score"]) >= 15)
            if not state["velocity_score_sent"] or conv_changed:
                cal_text  = "\n".join(cal_flags) if cal_flags else "No special events"
                hist_ctx  = load_historical_context(days=30)
                update_tag= "🔄 UPDATED — " if conv_changed else ""

                # ── Multi-ticker comparison (Option C: only when SPY is weak) ──
                spy_grade_clean = grade.split()[0] if grade else "F"
                qqq_context = ""
                if spy_grade_clean in SPY_WEAK_GRADES:
                    # Quick scan of all secondary tickers for morning brief context
                    best_alt_conv  = 0
                    best_alt_lines = []
                    grade_rank_map = {"A+": 5, "B+": 4, "C": 3, "D": 2, "F": 1}
                    spy_rank_v     = grade_rank_map.get(spy_grade_clean, 1)
                    for alt_t in ["QQQ", "GOOGL", "NVDA", "TSLA"]:
                        try:
                            ac, ag, ar, ap, ad, _, _ = fetch_ticker_signal(alt_t)
                            agc = ag.split()[0] if ag else "F"
                            best_alt_lines.append(
                                f"  {alt_t}: {agc} ({ac}/100) | {ar}"
                            )
                            if ac > best_alt_conv:
                                best_alt_conv   = ac
                                best_alt_ticker = alt_t
                                best_alt_grade  = agc
                        except Exception:
                            pass
                    if best_alt_conv > conv:
                        qqq_context = (
                            f"\n\nALTERNATIVES (SPY is weak today):\n"
                            + "\n".join(best_alt_lines) +
                            f"\n→ Best pick: {best_alt_ticker} ({best_alt_grade})"
                        )
                        state["multi_ticker_signal_sent"] = True
                        state["qqq_signal_sent"]          = True

                # ── PDT status ──
                pdt_note = check_pdt_status(conv, grade)

                written = write_alert_with_claude(
                    alert_type="morning_report",
                    price=price, gex_state=gex_state, regime=regime,
                    vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                    vix=vix_spot, vvix=vvix_val,
                    conviction=conv, combined=combined_score,
                    unwind_score=unwind_score, vanna_target=vt, charm_target=ct,
                    news_sentiment=news_sent, catalyst_type=cat_type,
                    macro_override=mac_ovr, flow_dir=flow_dir,
                    previous_regime=state.get("previous_regime",""),
                    previous_gex_state=state.get("previous_gex_state",""),
                    claude_verdict=claude_verdict,
                    extra_context=(f"Calendar: {cal_text}\nVIX: {vix_sig}\n"
                                   f"VVIX: {vvix_sig}\nOPEX Phase: {cycle_phase}\n\n"
                                   f"{hist_ctx}{qqq_context}")
                )
                trades_left = MAX_DAY_TRADES - state.get("day_trades_used", 0)
                pdt_line = f"PDT: {state.get('day_trades_used',0)}/{MAX_DAY_TRADES} trades used ({trades_left} left)"
                if written:
                    alert(f"🌅 {update_tag}MORNING BRIEF — SPY\n{'─'*35}\n"
                          f"{now_str} PDT | ${price} | Score: {combined_score}/100\n"
                          f"{pdt_line}\n\n{written}{pdt_note}")
                else:
                    cl_text = "\n".join(checklist)
                    alert(f"🌅 {update_tag}PRE-MARKET REPORT — SPY\n{'─'*35}\n"
                          f"{now_str} PDT | ${price}\n{pdt_line}\n\n"
                          f"🎯 {conv}/100 — {grade}\n→ {rec}\n\n"
                          f"📅 Phase: {cycle_phase}\n"
                          f"📋 CHECKLIST\n{cl_text}\n\n"
                          f"VIX: {vix_sig}\nVVIX: {vvix_sig}"
                          f"{qqq_context}{pdt_note}")
                state["velocity_score_sent"] = True

                # ── Fire standalone QQQ comparison alert if SPY weak ──
                if spy_grade_clean in SPY_WEAK_GRADES and not state.get("qqq_signal_sent"):
                    run_qqq_check(grade, conv, regime, now_str)
        # Alert flag resets
        if state["consolidation_alert_sent"]:
            if gex_state != state.get("consolidation_gex_state"):
                state["consolidation_alert_sent"] = False
        now_epoch = time.time()
        if (state["hedge_unwind_alert_sent"] and unwind_score >= 60
                and now_epoch - state.get("last_unwind_alert_time", 0) > 2700):
            state["hedge_unwind_alert_sent"] = False
        # Mid-session update
        if (is_market_open() and 6 <= h <= 12
                and now_epoch - state.get("last_summary_time", 0) > 5400):
            alert(f"📊 MID-SESSION UPDATE — SPY\n{'─'*35}\n"
                  f"{now_str} PDT | ${price}\n\n"
                  f"GEX: {gex_state} | Regime: {regime}\n"
                  f"Ratio: {ratio_r}x | OI: {oi_fmt} | VOL: {vol_fmt}\n"
                  f"Phase: {cycle_phase}\n\n"
                  f"VIX: {vix_sig} | VVIX: {vvix_sig}\n"
                  f"Flow: {flow_dir} | Unwind: {unwind_score}/100\n\n"
                  f"🎯 {conv}/100 — {grade}\n→ {rec}")
            state["last_summary_time"] = now_epoch
        # EOD — fires exactly once at 1pm PDT
        if h >= 13 and not state.get("eod_fired_today"):
            eod_autofill(price)
            state["eod_fired_today"] = True
            # Reset daily intraday state
            for k in ["velocity_score_sent","consolidation_alert_sent",
                      "hedge_unwind_alert_sent","open_drive_detected",
                      "doji_transition_sent","vol_gex_velocity_alert_sent",
                      "gap_fill_alert_sent","gap_type_sent",
                      "open_candle_analyzed","qqq_signal_sent",
                      "multi_ticker_signal_sent","day_trades_warning_sent"]:
                state[k] = False
            for k in ["last_unwind_alert_time","last_summary_time",
                      "last_heartbeat","last_wall_alert_price","gap_size",
                      "day_trades_used"]:
                state[k] = 0
            for k in ["consolidation_gex_state","open_price","open_iv","open_volume",
                      "session_vwap","gap_direction","open_vol_gex_snapshot",
                      "open_candle_type","open_candle_confluence",
                      "open_candle_body_pct","open_candle_upper_wick",
                      "open_candle_lower_wick","open_candle_vol_ratio",
                      "open_candle_vwap_pos",
                      "spy_liq_zone","qqq_liq_zone","tsla_liq_zone",
                      "nvda_liq_zone","googl_liq_zone"]:
                state[k] = None
            for k in ["open_time_prices","tick_history"]:
                state[k] = []
            for k in ["session_high","session_low"]:
                state[k] = None
            for k in ["regime_transitions_today","vwap_breaks_today"]:
                state[k] = 0
            state["last_logged_row"] = {}
            state["gap_type"]        = "UNKNOWN"
            state["gap_conviction"]  = 0
            state["cache_status"]    = ""
            state["cache_levels"]    = ""
            state["cache_all_levels"]= ""
            state["cache_last_updated"] = 0
        # Update state
        state["previous_gex_state"] = gex_state
        state["previous_ratio"]     = ratio
        state["previous_vol_gex"]   = vol_gex
        state["previous_oi_gex"]    = oi_gex
        state["previous_regime"]    = regime
        state["regime"]             = regime
        state["last_conviction_score"] = conv
        # Rebuild fast command caches — called last so all state is final
        rebuild_command_cache(price, gex_state, regime, conv, grade,
                              vix_sig, vvix_sig, flow_dir, unwind_score,
                              vt, ct, cycle_phase)
        build_all_levels_cache()
    except Exception as e:
        print(f"run_job error: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# TELEGRAM COMMANDS
# /status — bot health check
# /notes [text] — add context to today's log
# /overnight — manual overnight snapshot
# ─────────────────────────────────────────────
async def handle_telegram_updates():
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        # FIX: use offset to prevent re-processing old updates (stops spam loop)
        last_id = state.get("telegram_last_update_id", 0)
        offset  = last_id + 1 if last_id > 0 else None
        updates = await bot.get_updates(timeout=5, offset=offset)
        today_str = now_pdt().strftime("%Y-%m-%d")
        for update in updates:
            if update.update_id > state.get("telegram_last_update_id", 0):
                state["telegram_last_update_id"] = update.update_id
            if not update.message: continue
            text = (update.message.text or "").strip()
            # ── /status — instant from cache ───────────────────────────
            if text == "/status":
                msg = state.get("cache_status") or (
                    f"Bot starting up — status ready after 6:00am PDT first reading.\n"
                    f"APIs: UW:{'✅' if UW_TOKEN else '❌'} "
                    f"Claude:{'✅' if anthropic_client else '❌'} "
                    f"GitHub:{'✅' if GITHUB_TOKEN else '❌'}"
                )
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                continue

            # ── /levels — SPY zones, instant from cache ────────────────────
            if text == "/levels":
                msg = state.get("cache_levels") or (
                    "📍 SPY levels not cached yet.\n"
                    "Available after 6:30am PDT first reading."
                )
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                continue

            # ── /levels all — all tickers, instant from cache ─────────────
            if text == "/levels all":
                msg = state.get("cache_all_levels") or (
                    "📍 All levels not cached yet.\n"
                    "Available after secondary tickers scan (SPY D/F day)."
                )
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                continue

            # ── /pdt — instant from state ──────────────────────────────────
            if text == "/pdt":
                used = state.get("day_trades_used", 0)
                left = MAX_DAY_TRADES - used
                now_s = now_pdt().strftime("%H:%M")
                status_line = (
                    "🚫 LIMIT — 1DTE only" if left <= 0
                    else "⚠️ LAST TRADE — A+/B+ only" if left == 1
                    else "✅ Trades available"
                )
                await bot.send_message(chat_id=CHAT_ID, text=(
                    f"📊 PDT — {now_s} PDT\n\n"
                    f"Used: {used}/{MAX_DAY_TRADES}  Left: {left}\n"
                    f"{status_line}\n\n"
                    f"/trade to log a day trade."
                ))
                continue

            # ── /trade — log a day trade, instant ─────────────────────────
            if text == "/trade":
                used  = state.get("day_trades_used", 0) + 1
                state["day_trades_used"] = used
                left  = MAX_DAY_TRADES - used
                msg   = f"✅ Trade logged: {used}/{MAX_DAY_TRADES} — {left} left\n"
                msg  += ("🚫 PDT limit. 1DTE entries only." if left <= 0
                         else "⚠️ One trade left — best signal only." if left == 1
                         else f"{left} remaining.")
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                continue

            # ── /tickers /qqq — instant from state ────────────────────────
            if text in ("/qqq", "/tickers"):
                spy_c = state.get("last_conviction_score", 0)
                if spy_c >= 80: spy_g = "A+"
                elif spy_c >= 65: spy_g = "B+"
                elif spy_c >= 50: spy_g = "C"
                elif spy_c >= 35: spy_g = "D"
                else: spy_g = "F"
                lines = [f"🔵 SPY  {spy_g} ({spy_c}/100) | {state.get('regime','—')}"]
                for t in ["QQQ","TSLA","NVDA","GOOGL"]:
                    sc = state.get(f"{t.lower()}_last_score", 0)
                    gr = (state.get(f"{t.lower()}_last_grade") or "—")
                    re = state.get(f"{t.lower()}_last_regime") or "—"
                    gc = gr.split()[0] if gr not in ("—","") else "—"
                    lines.append(f"{'⚫' if sc==0 else '⚪'} {t:5s} {gc} ({sc}/100) | {re}")
                best_t  = max(["QQQ","TSLA","NVDA","GOOGL"],
                              key=lambda t: state.get(f"{t.lower()}_last_score", 0))
                best_s  = state.get(f"{best_t.lower()}_last_score", 0)
                note    = (f"\n🏆 Best alt: {best_t} ({best_s}/100)" if best_s > spy_c
                           else "\n✅ SPY leads today")
                if all(state.get(f"{t.lower()}_last_score",0)==0 for t in ["QQQ","TSLA","NVDA","GOOGL"]):
                    note += "\n(Secondaries run when SPY is D/F — use /tickers then)"
                now_s = now_pdt().strftime("%H:%M")
                await bot.send_message(chat_id=CHAT_ID, text=(
                    f"📊 TICKERS — {now_s} PDT\n\n" + "\n".join(lines) + note
                ))
                continue

            # ── /overnight — still live (must be fresh data) ──────────────
            if text == "/overnight":
                now_str = now_pdt().strftime("%H:%M")
                await bot.send_message(chat_id=CHAT_ID,
                    text=f"🌙 Fetching overnight snapshot at {now_str} PDT...")
                overnight = fetch_overnight_data()
                written   = write_overnight_alert_with_claude(overnight)
                fut_dir   = overnight.get("futures_direction","FLAT")
                fut_chg   = overnight.get("futures_change_pct",0) or 0
                vix_curr  = overnight.get("vix_current","N/A")
                cat       = overnight.get("catalyst_type","NONE")
                msg = (f"🌙 OVERNIGHT — SPY\n{'─'*30}\n{now_str} PDT\n\n"
                       f"Futures: {fut_dir} {fut_chg:+.2f}%\n"
                       f"VIX: {vix_curr} | Catalyst: {cat}\n\n"
                       + (written or "Full update at next check."))
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                log_overnight_reading(overnight)
                continue

            # ── /notes — writes to CSV ─────────────────────────────────────
            if text.startswith("/notes"):
                note = text[6:].strip()
                if not note:
                    await bot.send_message(chat_id=CHAT_ID,
                        text="Usage: /notes [text]\nExample: /notes Iran news drove gap fade")
                    continue
                if not os.path.exists(LOG_FILE):
                    await bot.send_message(chat_id=CHAT_ID,
                        text="No log yet — available after first market reading.")
                    continue
                with open(LOG_FILE, "r", newline="") as f:
                    rows = list(csv.DictReader(f))
                updated = 0
                for row in rows:
                    if row["date"] == today_str:
                        row["notes"] = note
                        updated += 1
                with open(LOG_FILE, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
                    writer.writeheader()
                    writer.writerows(rows)
                await bot.send_message(chat_id=CHAT_ID, text=(
                    f"✅ Note saved to {updated} rows: {note}"))
                git_commit_log(reason="notes")
                continue

    except Exception as e:
        print(f"Telegram command error: {e}")

def check_telegram_commands():
    asyncio.run(handle_telegram_updates())

# ─────────────────────────────────────────────
# MODULE: OPEN CANDLE ANALYSIS
# Runs at 6:35am after first 5-min candle closes.
# Reads body size, wick ratio, volume vs average,
# VWAP position, and checks confluence with the
# pre-market GEX thesis.
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# MODULE: QQQ SECONDARY SIGNAL
# Only runs when SPY conviction is D or F.
# Uses same GEX + regime logic as SPY.
# VXN (Nasdaq volatility) instead of VIX.
# Generic functions that work for any ticker.
# SPY continues to use its own dedicated modules.
# ─────────────────────────────────────────────

# Generic functions that work for any ticker.
# SPY continues to use its own dedicated modules.
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# MODULE: LIQUIDITY ZONE ENGINE
# ─────────────────────────────────────────────
# Computes the price range a ticker is expected
# to travel and the key levels for contract selection.
#
# TARGET     — where price is being pulled (profit exit)
# SUPPORT    — where buyers defend (stop zone for puts)
# RESISTANCE — where sellers wait (stop zone for calls)
# RANGE      — expected low to high for today
# ─────────────────────────────────────────────

def compute_liquidity_zones(ticker, price, vanna_target=None,
                             charm_target=None, vwap=None,
                             prev_close=None, session_high=None,
                             session_low=None):
    """
    Returns a dict with all key levels for the ticker.
    Uses pivot points, ATR, vanna/charm, VWAP, and
    round-number magnetism to define the expected range.
    """
    zones = {
        "ticker": ticker, "price": round(price, 2),
        "target": None, "target2": None,
        "support": None, "resistance": None,
        "run_low": None, "run_high": None,
        "atr": None,
        "contract_guide": "", "levels_text": "",
    }
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        pivot = pp_r1 = pp_r2 = pp_s1 = pp_s2 = atr = None

        if not df.empty and len(df) >= 2:
            prev   = df.iloc[-2]
            ph, pl, pc = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
            pivot  = round((ph + pl + pc) / 3, 2)
            pp_r1  = round(2 * pivot - pl, 2)
            pp_r2  = round(pivot + (ph - pl), 2)
            pp_s1  = round(2 * pivot - ph, 2)
            pp_s2  = round(pivot - (ph - pl), 2)
            highs  = df["High"].values.flatten().astype(float)
            lows   = df["Low"].values.flatten().astype(float)
            closes = df["Close"].values.flatten().astype(float)
            trs = [max(highs[i]-lows[i],
                       abs(highs[i]-closes[i-1]),
                       abs(lows[i]-closes[i-1]))
                   for i in range(1, len(highs))]
            atr = round(float(np.mean(trs[-5:])), 2) if trs else None
            zones["atr"] = atr

        # Round-number levels ($5 increments)
        r5_above = round(np.ceil(price / 5) * 5, 2)
        r5_below = round(np.floor(price / 5) * 5, 2)
        if r5_above == price: r5_above += 5
        if r5_below == price: r5_below -= 5

        # Assign primary target
        if vanna_target:
            zones["target"] = vanna_target
            zones["target2"] = charm_target if charm_target and charm_target != vanna_target \
                                else (r5_above if vanna_target > price else r5_below)
        else:
            zones["target"]  = pp_r1 or r5_above
            zones["target2"] = pp_s1 or r5_below

        # Resistance and support
        zones["resistance"] = pp_r1 or r5_above
        zones["support"]    = pp_s1 or r5_below

        # VWAP refines support/resistance
        if vwap:
            if price > vwap:
                zones["support"] = max(round(vwap, 2), zones["support"] or round(vwap, 2))
            else:
                zones["resistance"] = min(round(vwap, 2), zones["resistance"] or round(vwap, 2))

        # Expected daily range
        if atr:
            zones["run_low"]  = round(price - atr * 0.8, 2)
            zones["run_high"] = round(price + atr * 0.8, 2)
        else:
            zones["run_low"]  = pp_s2 or round(r5_below - 5, 2)
            zones["run_high"] = pp_r2 or round(r5_above + 5, 2)

        if prev_close:
            zones["run_low"] = min(zones["run_low"], round(prev_close * 0.99, 2))

        # Contract selection guide
        t  = zones["target"]
        t2 = zones["target2"]
        s  = zones["support"]
        r  = zones["resistance"]
        rl = zones["run_low"]
        rh = zones["run_high"]
        regime = state.get("regime", "UNKNOWN")

        if regime == "BEARISH_HEDGE_BUILD":
            zones["contract_guide"] = (
                f"PUTS:\n"
                f"  Buy:    ATM or ${round(price-1,0):.0f}P\n"
                f"  Target: ${t:.2f}" + (f" then ${t2:.2f}" if t2 else "") + "\n"
                f"  Stop:   price closes back above ${r:.2f}\n"
                f"  Range:  ${rl:.2f} – ${rh:.2f} today"
            )
        elif regime in ("HEDGE_UNWIND_CONFIRMED", "BULLISH_MOMENTUM", "HEDGE_UNWIND_EARLY"):
            zones["contract_guide"] = (
                f"CALLS:\n"
                f"  Buy:    ATM or ${round(price+1,0):.0f}C\n"
                f"  Target: ${t:.2f}" + (f" then ${t2:.2f}" if t2 else "") + "\n"
                f"  Stop:   price closes below ${s:.2f}\n"
                f"  Range:  ${rl:.2f} – ${rh:.2f} today"
            )
        else:
            zones["contract_guide"] = (
                f"No regime — wait for confirmation\n"
                f"  Support:    ${s:.2f}\n"
                f"  Resistance: ${r:.2f}\n"
                f"  Range:      ${rl:.2f} – ${rh:.2f} today"
            )

        # Formatted text block
        van_line   = f"  Vanna pull:  ${vanna_target:.2f}\n" if vanna_target else ""
        charm_line = f"  Charm wall:  ${charm_target:.2f}\n" if charm_target else ""
        vwap_line  = f"  VWAP:        ${round(vwap,2)}\n"   if vwap       else ""
        atr_line   = f"  ATR (5d):    ${atr:.2f}\n"         if atr        else ""

        zones["levels_text"] = (
            f"{ticker} @ ${price:.2f}\n"
            f"{'─'*28}\n"
            f"  Range:       ${rl:.2f} – ${rh:.2f}\n"
            f"  Resistance:  ${r:.2f}\n"
            f"  Target:      ${t:.2f}" +
            (f" → ${t2:.2f}" if t2 else "") + "\n"
            f"  Support:     ${s:.2f}\n"
            + vwap_line + van_line + charm_line + atr_line +
            f"\n{zones['contract_guide']}"
        )

    except Exception as e:
        print(f"Liq zone error {ticker}: {e}")
        zones["levels_text"] = f"{ticker} @ ${price:.2f} — levels unavailable"

    return zones


def update_spy_liquidity_zone(price, vt, ct):
    """Updates SPY zone + rebuilds /levels cache. Called from run_job."""
    try:
        zone = compute_liquidity_zones(
            "SPY", price,
            vanna_target=vt, charm_target=ct,
            vwap=state.get("session_vwap"),
            prev_close=state.get("prev_session_close"),
            session_high=state.get("session_high"),
            session_low=state.get("session_low"),
        )
        state["spy_liq_zone"] = zone
        now_str = now_pdt().strftime("%H:%M")
        state["cache_levels"] = (
            f"📍 SPY LEVELS — {now_str} PDT\n\n" + zone["levels_text"]
        )
        state["cache_last_updated"] = time.time()
        print(f"💧 SPY zone: ${zone.get('run_low')}–${zone.get('run_high')}"
              f" | target ${zone.get('target')}")
    except Exception as e:
        print(f"SPY zone update error: {e}")


def update_ticker_liquidity_zone(ticker, price):
    """Compute and cache zone for a secondary ticker. Called from fetch_ticker_signal."""
    try:
        zone = compute_liquidity_zones(ticker, price,
                                        vwap=None, prev_close=None)
        state[f"{ticker.lower()}_liq_zone"] = zone
    except Exception as e:
        print(f"{ticker} zone error: {e}")


def build_all_levels_cache():
    """Rebuilds /levels all cache from already-cached zones. No API calls."""
    try:
        now_str  = now_pdt().strftime("%H:%M")
        parts    = [f"📍 ALL LEVELS — {now_str} PDT\n"]
        for t in ["SPY", "QQQ", "TSLA", "NVDA", "GOOGL"]:
            key  = f"{t.lower()}_liq_zone"
            zone = state.get(key)
            if zone and zone.get("levels_text"):
                parts.append(zone["levels_text"])
        state["cache_all_levels"] = "\n\n".join(parts)
    except Exception as e:
        print(f"All levels cache error: {e}")


# ─────────────────────────────────────────────
# COMMAND CACHE BUILDER
# Pre-builds /status response string after each
# run_job. Commands read from cache — <100ms.
# ─────────────────────────────────────────────

def rebuild_command_cache(price, gex_state, regime, conv, grade,
                          vix_sig, vvix_sig, flow_dir, unwind_score,
                          vt, ct, cycle_phase):
    """Rebuilds fast-response cache strings. Called at end of run_job."""
    try:
        now_str    = now_pdt().strftime("%H:%M")
        today_str  = now_pdt().strftime("%Y-%m-%d")
        trades_left= MAX_DAY_TRADES - state.get("day_trades_used", 0)
        vt_str = f"${vt:.2f}" if vt else "—"
        ct_str = f"${ct:.2f}" if ct else "—"

        def gs(ticker):
            s = state.get(f"{ticker.lower()}_last_score", 0)
            g = state.get(f"{ticker.lower()}_last_grade") or "—"
            r = state.get(f"{ticker.lower()}_last_regime") or "—"
            gc = g.split()[0] if g and g != "—" else "—"
            return f"{gc} ({s}/100) | {r}"

        state["cache_status"] = (
            f"📊 STATUS — {now_str} PDT\n"
            f"{'─'*28}\n\n"
            f"SPY  ${round(price,2)} | {gex_state}\n"
            f"     {grade.split()[0]} ({conv}/100) | {regime}\n"
            f"     Vanna:{vt_str} | Charm:{ct_str}\n"
            f"     Unwind:{unwind_score}/100 | {cycle_phase}\n\n"
            f"QQQ  {gs('qqq')}\n"
            f"TSLA {gs('tsla')}\n"
            f"NVDA {gs('nvda')}\n"
            f"GOOGL {gs('googl')}\n\n"
            f"VIX:  {vix_sig}\n"
            f"VVIX: {vvix_sig}\n"
            f"Flow: {flow_dir}\n\n"
            f"PDT: {state.get('day_trades_used',0)}/{MAX_DAY_TRADES} "
            f"({trades_left} left)\n"
            f"APIs: UW:{'✅' if UW_TOKEN else '❌'} "
            f"Claude:{'✅' if anthropic_client else '❌'} "
            f"GitHub:{'✅' if GITHUB_TOKEN else '❌'}"
        )
        state["cache_last_updated"] = time.time()
    except Exception as e:
        print(f"Cache rebuild error: {e}")


def fetch_ticker_gex(ticker):
    """Fetch GEX for any UW-supported ticker."""
    try:
        url = f"https://api.unusualwhales.com/api/stock/{ticker}/spot-exposures"
        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()["data"]
        if not data:
            return None, None, None
        latest  = data[-1]
        oi_gex  = float(latest["gamma_per_one_percent_move_oi"])
        vol_gex = float(latest["gamma_per_one_percent_move_vol"])
        price   = float(latest["price"])
        return oi_gex, vol_gex, price
    except Exception as e:
        print(f"{ticker} GEX error: {e}")
        return None, None, None


def fetch_ticker_iv(ticker, iv_ticker_symbol):
    """
    Fetch implied volatility for any ticker.

    QQQ  -> VXN  (Nasdaq VIX equivalent)
    TSLA/NVDA/GOOGL -> IV rank from UW API
      IV rank 0-100: where 50 = median historical IV
      Above 50 = elevated, above 80 = extreme

    Returns: iv_spot, iv_term, iv_sig
    """
    try:
        if iv_ticker_symbol:
            iv_h   = yf.Ticker(iv_ticker_symbol).history(period="5d", interval="1d")
            vix3m  = yf.Ticker("^VIX3M").history(period="2d", interval="1d")
            iv_spot = float(iv_h["Close"].iloc[-1])  if not iv_h.empty  else None
            v3m_val = float(vix3m["Close"].iloc[-1]) if not vix3m.empty else None
            if iv_spot and v3m_val:
                if iv_spot > v3m_val * 1.15:
                    iv_term = "BACKWARDATION"
                elif iv_spot < v3m_val * 0.95:
                    iv_term = "CONTANGO"
                else:
                    iv_term = "FLAT"
            else:
                iv_term = "UNKNOWN"
            if iv_spot:
                if iv_spot >= 35:
                    iv_sig = f"EXTREME ({round(iv_spot,1)})"
                elif iv_spot >= 25:
                    iv_sig = f"ELEVATED ({round(iv_spot,1)})"
                elif iv_spot >= 18:
                    iv_sig = f"MODERATE ({round(iv_spot,1)})"
                else:
                    iv_sig = f"LOW ({round(iv_spot,1)})"
            else:
                iv_sig = "Unavailable"
            return iv_spot, iv_term, iv_sig

        # IV rank from UW API (single stocks)
        r = requests.get(
            f"https://api.unusualwhales.com/api/stock/{ticker}/iv-rank",
            headers={"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"},
            timeout=8
        )
        if r.status_code == 200:
            data = r.json().get("data", {})
            iv_rank = float(data.get("iv_rank", 50) or 50) if data else 50.0
            if iv_rank >= 80:
                iv_term = "BACKWARDATION"
                iv_sig  = f"IV RANK {round(iv_rank)} — Extreme"
            elif iv_rank >= 60:
                iv_term = "FLAT"
                iv_sig  = f"IV RANK {round(iv_rank)} — Elevated"
            elif iv_rank >= 40:
                iv_term = "FLAT"
                iv_sig  = f"IV RANK {round(iv_rank)} — Normal"
            else:
                iv_term = "CONTANGO"
                iv_sig  = f"IV RANK {round(iv_rank)} — Low"
            return iv_rank, iv_term, iv_sig

        return None, "UNKNOWN", "Unavailable"
    except Exception as e:
        print(f"{ticker} IV error: {e}")
        return None, "UNKNOWN", "Unavailable"


def fetch_ticker_signal(ticker):
    """
    Generic signal scorer for any secondary ticker.
    Returns: score, grade, regime, price, direction, iv_sig, summary

    Uses identical logic to SPY:
    - GEX fetch from UW API
    - Regime detection (same algorithm)
    - Conviction scorer (same weights)
    - IV rank or index replaces VIX
    - VVIX still used market-wide
    - Same unwind detector
    """
    try:
        cfg = SECONDARY_TICKERS.get(ticker, {})
        iv_ticker_sym = cfg.get("iv_ticker")
        vol_hist_key  = cfg.get("vol_history_key",
                                f"{ticker.lower()}_vol_gex_history")

        oi_gex, vol_gex, price = fetch_ticker_gex(ticker)
        if oi_gex is None or vol_gex == 0:
            return 0, "F", "INSUFFICIENT_DATA", 0, "NEUTRAL", "N/A",                    f"{ticker} data unavailable"

        hist = state.setdefault(vol_hist_key, [])
        hist.append(vol_gex)
        if len(hist) > 10:
            hist.pop(0)

        ratio = abs(vol_gex) / abs(oi_gex) if oi_gex != 0 else 0
        regime, _ = detect_regime(oi_gex, vol_gex, hist)
        iv_spot, iv_term, iv_sig = fetch_ticker_iv(ticker, iv_ticker_sym)

        vvix_h   = yf.Ticker("^VVIX").history(period="2d", interval="1d")
        vvix_val = float(vvix_h["Close"].iloc[-1]) if not vvix_h.empty else None

        _, unwind_score, _, _ = fetch_hedge_unwind_signals_for(ticker)

        pdt_t = now_pdt()
        vanna_window = ((pdt_t.hour - 6) * 60 + pdt_t.minute - 30) < 270
        _, cal_bonus, _, _ = get_calendar_flags()

        tick_approx, inv_bias = 0, "NEUTRAL"
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not df.empty and len(df) >= 10:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                closes = df["Close"].iloc[-10:].values.flatten().astype(float)
                opens  = df["Open"].iloc[-10:].values.flatten().astype(float)
                up   = sum(1 for c, o in zip(closes, opens) if c > o)
                down = sum(1 for c, o in zip(closes, opens) if c < o)
                tick_approx = (up - down) * 100
                cp = float(df["Close"].iloc[-1])
                vw = float(
                    (df["Close"] * df["Volume"]).cumsum().iloc[-1]
                    / df["Volume"].cumsum().iloc[-1]
                )
                inv_bias = ("BULL ZONE" if cp > vw * 1.002
                            else "BEAR ZONE" if cp < vw * 0.998
                            else "NEUTRAL")
        except Exception:
            pass

        conv, grade, rec, _ = score_conviction(
            vix_spot=iv_spot, vvix_val=vvix_val, vix_term=iv_term,
            vol_gex=vol_gex, prev_vol_gex=None, regime=regime,
            unwind_score=unwind_score, cal_bonus=cal_bonus,
            vanna_window=vanna_window, conflict=False, ratio=ratio,
            tick_approx=tick_approx, inv_bias=inv_bias, open_drive=False
        )

        if regime == "BEARISH_HEDGE_BUILD":
            direction = "BEARISH"
        elif regime in ("HEDGE_UNWIND_CONFIRMED", "BULLISH_MOMENTUM"):
            direction = "BULLISH"
        elif regime == "HEDGE_UNWIND_EARLY":
            direction = "EARLY BULLISH"
        else:
            direction = "NEUTRAL"

        state[f"{ticker.lower()}_last_score"]  = conv
        state[f"{ticker.lower()}_last_regime"] = regime
        state[f"{ticker.lower()}_last_grade"]  = grade
        # Update this ticker's liquidity zone for /levels all command
        update_ticker_liquidity_zone(ticker, price)

        summary = (f"{ticker} ${round(price,2)} | {grade.split()[0]} ({conv}/100)"
                   f" | {regime} | {direction}")
        print(f"📊 {ticker}: {summary}")
        return conv, grade, regime, price, direction, iv_sig, summary

    except Exception as e:
        print(f"{ticker} signal error: {e}")
        return 0, "F", "INSUFFICIENT_DATA", 0, "NEUTRAL", "N/A",                f"{ticker} error: {e}"


def fetch_hedge_unwind_signals_for(ticker):
    """Same unwind detector as SPY — works for any ticker."""
    try:
        url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-contracts"
        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        response = requests.get(url, headers=headers,
                                params={"limit": 100, "order": "desc"}, timeout=10)
        data = response.json().get("data", [])
        if not data:
            return False, 0, [], "NEUTRAL"
        signals, score = [], 0
        for contract in data:
            try:
                ctype  = str(contract.get("type", "")).upper()
                volume = float(contract.get("volume", 0) or 0)
                oi     = float(contract.get("open_interest", 0) or 0)
                strike = float(contract.get("strike", 0) or 0)
                exec_  = str(contract.get("execution_estimate", "")).upper()
                ratio  = volume / oi if oi > 0 else 0
                is_put = "PUT" in ctype
                is_call= "CALL" in ctype
                if is_put and ratio >= 50 and volume >= 2000:
                    score += 25
                    signals.append(f"PUT HEDGE CLOSING ${strike:.0f}P {round(ratio)}x")
                elif is_put and ratio >= 10 and volume >= 500:
                    score += 10
                    signals.append(f"PUT CLOSING ${strike:.0f}P {round(ratio)}x")
                if is_put and "DESCENDING" in exec_ and volume >= 500:
                    score += 15
                    signals.append(f"DESCENDING FILL ${strike:.0f}P")
                if is_call and "SWEEP" in exec_ and volume >= 500:
                    score += 8
                    signals.append(f"CALL SWEEP ${strike:.0f}C")
            except Exception:
                continue
        score = min(score, 100)
        if score >= 40: return True, score, signals[:6], "BULLISH"
        if score >= 20: return True, score, signals[:6], "LEANING BULLISH"
        return False, score, signals[:6], "NEUTRAL"
    except Exception as e:
        print(f"Unwind {ticker} error: {e}")
        return False, 0, [], "UNAVAILABLE"


def run_multi_ticker_check(spy_grade, spy_conv, spy_regime, now_str):
    """
    Runs all 4 secondary tickers when SPY is D or F.
    Ranks by conviction score. Fires ONE alert with
    the full ranking and the recommended trade.
    """
    if state.get("multi_ticker_signal_sent"):
        return

    grade_rank = {"A+": 5, "B+": 4, "C": 3, "D": 2, "F": 1}
    spy_rank   = grade_rank.get(
        spy_grade.split()[0] if spy_grade else "F", 1)

    results = []
    for ticker in ["QQQ", "GOOGL", "NVDA", "TSLA"]:
        try:
            conv, grade, regime, price, direction, iv_sig, summary =                 fetch_ticker_signal(ticker)
            results.append({
                "ticker": ticker, "conv": conv, "grade": grade,
                "gc":     grade.split()[0] if grade else "F",
                "regime": regime, "price": price,
                "direction": direction, "iv_sig": iv_sig,
            })
        except Exception as e:
            print(f"Multi-check {ticker}: {e}")

    if not results:
        return

    results.sort(key=lambda x: x["conv"], reverse=True)
    best      = results[0]
    best_rank = grade_rank.get(best["gc"], 1)

    if best_rank <= spy_rank and best_rank < 3:
        print(f"No secondary ticker better than SPY ({spy_grade}) — no alert")
        return

    de = {"BEARISH": "🔴", "BULLISH": "🟢",
          "EARLY BULLISH": "🟡", "NEUTRAL": "⚪"}
    am = {"BEARISH": "PUTS — wait for open candle confirmation",
          "BULLISH": "CALLS — wait for open candle confirmation",
          "EARLY BULLISH": "Prepare calls — not confirmed yet",
          "NEUTRAL": "Wait — no clear direction"}

    trades_left = MAX_DAY_TRADES - state.get("day_trades_used", 0)
    pdt_note = ""
    if trades_left <= 0:
        pdt_note = "\n\n🚫 PDT LIMIT — 1DTE entries only today"
    elif trades_left == 1:
        pdt_note = f"\n\n⚠️ Last day trade — use on {best['ticker']} only"

    rows = []
    for i, r in enumerate(results, 1):
        rows.append(
            f"  {i}. {de.get(r['direction'],'⚪')} {r['ticker']:5s}"
            f" {r['gc']:3s} ({r['conv']:3d}/100) | {r['regime']}"
        )

    alert(
        f"📊 TICKER RANKINGS — SPY {spy_grade.split()[0]} today\n"
        f"{'─'*35}\n"
        f"{now_str} PDT\n\n"
        f"🔵 SPY: {spy_grade.split()[0]} ({spy_conv}/100) — skip\n\n"
        f"ALTERNATIVES:\n" + "\n".join(rows) +
        f"\n\n{'─'*20}\n"
        f"🏆 BEST TRADE: {best['ticker']}\n"
        f"${round(best['price'],2)} | {best['gc']} ({best['conv']}/100)\n"
        f"Regime: {best['regime']}\n"
        f"Direction: {de.get(best['direction'],'')} {best['direction']}\n"
        f"→ {am.get(best['direction'],'Wait')}\n"
        f"IV: {best['iv_sig']}" + pdt_note
    )

    state["multi_ticker_signal_sent"] = True
    state["qqq_signal_sent"]          = True
    print(f"Multi-ticker alert fired: best={best['ticker']} {best['gc']} ({best['conv']})")


def run_qqq_check(spy_grade, spy_conv, spy_regime, now_str):
    """Backward-compat wrapper — now runs all tickers."""
    run_multi_ticker_check(spy_grade, spy_conv, spy_regime, now_str)

# ─────────────────────────────────────────────
# MODULE: PDT TRADE COUNTER
# Tracks day trades used this week.
# Warns when approaching limit.
# /pdt command shows current count.
# /trade command increments counter.
# ─────────────────────────────────────────────

def check_pdt_status(conv, grade):
    """
    Called after every signal alert.
    Warns if approaching PDT limit.
    Returns a PDT note string to append to alerts.
    """
    used   = state.get("day_trades_used", 0)
    left   = MAX_DAY_TRADES - used
    grade_clean = grade.split()[0] if grade else "?"

    if left <= 0:
        return (
            f"\n\n🚫 PDT LIMIT REACHED ({used}/{MAX_DAY_TRADES})\n"
            f"No more day trades available today.\n"
            f"1DTE entry only — held overnight won't count."
        )
    elif left == 1:
        if not state.get("day_trades_warning_sent"):
            state["day_trades_warning_sent"] = True
            return (
                f"\n\n⚠️ LAST DAY TRADE ({used}/{MAX_DAY_TRADES} used)\n"
                f"Use it wisely — only {grade_clean} grade or better.\n"
                f"Or enter 1DTE and hold overnight to skip PDT count."
            )
    return ""


def analyze_open_candle():
    if not is_market_open():
        return
    pdt = now_pdt()
    h, m = pdt.hour, pdt.minute
    # Only fires 6:34-6:42am — after first candle closes, before second job
    if not (h == 6 and 34 <= m <= 42):
        return
    if state.get("open_candle_analyzed"):
        return
    try:
        spy = yf.download("SPY", period="1d", interval="5m", progress=False)
        if spy.empty or len(spy) < 3:
            print("Open candle: not enough bars yet")
            return
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        # First candle = most recent completed 5-min bar
        first  = spy.iloc[-1]
        o_     = float(first["Open"])
        c_     = float(first["Close"])
        h_     = float(first["High"])
        l_     = float(first["Low"])
        vol_   = float(first["Volume"])

        # Average volume: prior 10 bars (pre-open context)
        avg_vol = float(spy["Volume"].iloc[:-1].tail(10).mean())

        # Candle geometry
        full_range  = h_ - l_
        if full_range < 0.01:
            print("Open candle: zero-range candle, skipping")
            return
        body        = abs(c_ - o_)
        upper_wick  = h_ - max(o_, c_)
        lower_wick  = min(o_, c_) - l_
        body_pct    = round((body / full_range) * 100)
        upper_pct   = round((upper_wick / full_range) * 100)
        lower_pct   = round((lower_wick / full_range) * 100)
        is_bear     = c_ < o_
        is_bull     = c_ > o_
        vol_ratio   = round(vol_ / avg_vol, 2) if avg_vol > 0 else 1.0
        vol_surge   = vol_ratio >= 1.5

        vwap      = state.get("session_vwap")
        gap_type  = state.get("gap_type", "UNKNOWN")
        regime    = state.get("regime", "UNKNOWN")
        conv      = state.get("last_conviction_score", 0)
        now_str   = pdt.strftime("%H:%M")

        # ── CANDLE TYPE ──────────────────────────
        if body_pct >= 60 and vol_surge:
            if is_bear:
                ctype  = "STRONG_BEAR"
                cemoji = "🔴"
                cread  = "Large red body + volume surge — institutions selling aggressively"
                action = "PUTS — full size. This is the candle that produces 400-700% days."
            else:
                ctype  = "STRONG_BULL"
                cemoji = "🟢"
                cread  = "Large green body + volume surge — institutions buying aggressively"
                action = "CALLS — full size. Strong institutional conviction on open."

        elif upper_pct >= 40:
            ctype  = "REJECTION_WICK_TOP"
            cemoji = "⬇️"
            cread  = (f"Long upper wick ({upper_pct}% of range) — "
                      f"gap up immediately rejected by sellers")
            action = "PUTS on next red candle. Rejection wicks at open = cleanest put entry."

        elif lower_pct >= 40:
            ctype  = "REJECTION_WICK_BOTTOM"
            cemoji = "⬆️"
            cread  = (f"Long lower wick ({lower_pct}% of range) — "
                      f"gap down immediately rejected by buyers")
            action = "CALLS on next green candle. Rejection wicks at open = cleanest call entry."

        elif body_pct < 30:
            ctype  = "DOJI"
            cemoji = "⚪"
            cread  = (f"Doji — body only {body_pct}% of range. "
                      f"No conviction either direction.")
            action = ("WAIT. Do not enter. Watch next candle. "
                      "This open will chop 15-30min before committing.")

        elif body_pct >= 40:
            if is_bear:
                ctype  = "MODERATE_BEAR"
                cemoji = "🟠"
                cread  = f"Moderate red candle ({body_pct}% body) — bearish lean"
                action = ("PUTS half size on VWAP rejection. "
                          "Wait for second candle to confirm before adding.")
            else:
                ctype  = "MODERATE_BULL"
                cemoji = "🟡"
                cread  = f"Moderate green candle ({body_pct}% body) — bullish lean"
                action = ("CALLS half size on VWAP hold. "
                          "Wait for second candle to confirm before adding.")
        else:
            ctype  = "UNCLEAR"
            cemoji = "❓"
            cread  = "Mixed candle — no clean read"
            action = "WAIT for 3-5 candles before entering."

        # ── VWAP POSITION ────────────────────────
        if vwap:
            if c_ > vwap * 1.001:
                vwap_pos  = "ABOVE"
                vwap_note = f"${round(c_ - vwap, 2)} above VWAP"
            elif c_ < vwap * 0.999:
                vwap_pos  = "BELOW"
                vwap_note = f"${round(vwap - c_, 2)} below VWAP"
            else:
                vwap_pos  = "AT"
                vwap_note = "Pinned at VWAP — no direction yet"
        else:
            vwap_pos  = "UNKNOWN"
            vwap_note = "VWAP unavailable"

        # ── CONFLUENCE WITH PRE-MARKET THESIS ────
        pre_bear = (
            regime in ["BEARISH_HEDGE_BUILD", "HEDGE_UNWIND_EARLY"] or
            gap_type in ["FULL_FADE", "FADE_THEN_STATIC"]
        )
        pre_bull = (
            regime in ["HEDGE_UNWIND_CONFIRMED", "BULLISH_MOMENTUM"] or
            gap_type == "DIRECTIONAL"
        )
        bear_candle = ctype in ("STRONG_BEAR", "REJECTION_WICK_TOP", "MODERATE_BEAR")
        bull_candle = ctype in ("STRONG_BULL", "REJECTION_WICK_BOTTOM", "MODERATE_BULL")

        if (pre_bear and bear_candle) or (pre_bull and bull_candle):
            confluence = "CONFIRMED"
            conf_emoji = "🚨" if ctype in ("STRONG_BEAR","STRONG_BULL") else "✅"
            conf_note  = ("Pre-market thesis + candle structure ALIGNED.\n"
                         "Two independent signals agree. This is the entry.")
        elif ctype == "DOJI":
            confluence = "WAIT"
            conf_emoji = "⏳"
            conf_note  = "Candle indecision — pre-market thesis unconfirmed. Watch next candle."
        elif (pre_bear and bull_candle) or (pre_bull and bear_candle):
            confluence = "CONFLICT"
            conf_emoji = "⚠️"
            conf_note  = ("Candle CONTRADICTS pre-market thesis.\n"
                         "Do NOT enter. Something shifted at open. Reassess.")
        else:
            confluence = "NEUTRAL"
            conf_emoji = "➖"
            conf_note  = "No strong confluence. Wait for direction to develop."

        # ── VOLUME READ ──────────────────────────
        if vol_ratio >= 2.0:
            vol_tag = "🔥 SURGE"
        elif vol_surge:
            vol_tag = "✅ ELEVATED"
        elif vol_ratio < 0.7:
            vol_tag = "🔇 LIGHT — conviction lacking"
        else:
            vol_tag = "⚪ AVERAGE"

        # ── SAVE TO STATE ────────────────────────
        state["open_candle_type"]       = ctype
        state["open_candle_body_pct"]   = body_pct
        state["open_candle_upper_wick"] = upper_pct
        state["open_candle_lower_wick"] = lower_pct
        state["open_candle_vol_ratio"]  = vol_ratio
        state["open_candle_vwap_pos"]   = vwap_pos
        state["open_candle_confluence"] = confluence
        state["open_candle_analyzed"]   = True

        # ── ALERT ────────────────────────────────
        alert(
            f"{cemoji} OPEN CANDLE — SPY\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | First 5-min candle closed\n\n"
            f"Type: {ctype}\n"
            f"Body: {body_pct}% | "
            f"Upper wick: {upper_pct}% | "
            f"Lower wick: {lower_pct}%\n"
            f"Volume: {vol_ratio}x average — {vol_tag}\n"
            f"VWAP: {vwap_pos} ({vwap_note})\n\n"
            f"{cread}\n\n"
            f"{conf_emoji} CONFLUENCE: {confluence}\n"
            f"{conf_note}\n\n"
            f"{'─'*20}\n"
            f"📌 ACTION: {action}\n\n"
            f"Pre-market context:\n"
            f"Gap: {gap_type} | Regime: {regime} | Score: {conv}/100"
        )

        print(f"🕯️ Open candle: {ctype} | {confluence} | "
              f"Body:{body_pct}% | Vol:{vol_ratio}x | VWAP:{vwap_pos}")

    except Exception as e:
        print(f"Open candle error: {e}")


def midnight_reset():
    pdt = now_pdt()
    if pdt.hour == 0 and pdt.minute < 1:
        state["eod_fired_today"]        = False
        state["overnight_alerts_today"] = 0
        state["overnight_report_sent"]  = False
        state["last_overnight_check"]   = 0
        state["last_logged_row"]        = {}
        state["gap_type_sent"]          = False
        state["gap_type"]               = "UNKNOWN"
        state["gap_conviction"]         = 0
        state["open_vol_gex_snapshot"]  = None
        state["open_candle_analyzed"]   = False
        state["open_candle_type"]       = None
        state["open_candle_confluence"] = None
        state["open_candle_body_pct"]   = None
        state["open_candle_upper_wick"] = None
        state["open_candle_lower_wick"] = None
        state["open_candle_vol_ratio"]  = None
        state["open_candle_vwap_pos"]   = None
        state["qqq_signal_sent"]          = False
        state["multi_ticker_signal_sent"] = False
        state["qqq_last_score"]           = 0
        state["tsla_last_score"]          = 0
        state["nvda_last_score"]          = 0
        state["googl_last_score"]         = 0
        state["qqq_last_regime"]          = None
        state["tsla_last_regime"]         = None
        state["nvda_last_regime"]         = None
        state["googl_last_regime"]        = None
        state["qqq_vol_gex_history"]      = []
        state["tsla_vol_gex_history"]     = []
        state["nvda_vol_gex_history"]     = []
        state["googl_vol_gex_history"]    = []
        state["day_trades_used"]          = 0
        state["day_trades_warning_sent"]  = False
        state["spy_liq_zone"]             = None
        state["qqq_liq_zone"]             = None
        state["tsla_liq_zone"]            = None
        state["nvda_liq_zone"]            = None
        state["googl_liq_zone"]           = None
        state["cache_status"]             = ""
        state["cache_levels"]             = ""
        state["cache_all_levels"]         = ""
        state["cache_last_updated"]       = 0
        print("🌙 Midnight reset — daily flags cleared")

# ─────────────────────────────────────────────
# SCHEDULE — all UTC (PDT = UTC-7)
# ─────────────────────────────────────────────
schedule.every().day.at("13:00").do(run_job)   # 6:00am PDT
schedule.every().day.at("13:30").do(run_job)   # 6:30am PDT
schedule.every().day.at("14:00").do(run_job)   # 7:00am PDT
schedule.every().day.at("14:45").do(run_job)   # 7:45am PDT
schedule.every().day.at("15:30").do(run_job)   # 8:30am PDT
schedule.every().day.at("16:15").do(run_job)   # 9:15am PDT
schedule.every().day.at("17:00").do(run_job)   # 10:00am PDT
schedule.every().day.at("17:45").do(run_job)   # 10:45am PDT
schedule.every().day.at("18:30").do(run_job)   # 11:30am PDT
schedule.every().day.at("19:15").do(run_job)   # 12:15pm PDT
schedule.every().day.at("20:00").do(run_job)   # 1:00pm PDT (EOD)

schedule.every(5).minutes.do(check_vwap)
schedule.every(5).minutes.do(check_consolidation_job)
schedule.every(5).minutes.do(check_telegram_commands)
schedule.every(5).minutes.do(check_doji_transition)
schedule.every(5).minutes.do(check_gamma_wall_approach)
schedule.every(5).minutes.do(analyze_open_candle)
schedule.every(5).minutes.do(midnight_reset)
schedule.every(60).minutes.do(check_heartbeat)
schedule.every(90).minutes.do(run_overnight_check)

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
print("SPY GAMMABOT v5.2 — MULTI-TICKER + PDT TRACKER")
print("=" * 60)
print("Core: SPY GEX, Regime, Vanna/Charm, Unwind, VIX, TICK")
print("AI:   Claude verification + alert writing")
print("NEW:")
print("  27. QQQ Secondary Signal (fires when SPY is D/F)")
print("  28. PDT Trade Counter (3-trade limit tracking)")
print("  29. Open Candle Analysis (6:35am first candle)")
print("  30. Gap Classification Engine (6 gap types)")
print("Commands: /status /notes /overnight /pdt /trade /qqq")
print("=" * 60)
print()
if not GITHUB_TOKEN:
    print("⚠️  GITHUB_TOKEN not set — CSV will be wiped on redeploy!")
    print("   Add GITHUB_TOKEN to Railway vars (github.com/settings/tokens)")
    print()

init_log()

print("Bot started — waiting for scheduled jobs.")
print(f"Next market open: 6:00am PDT")
print()

while True:
    schedule.run_pending()
    time.sleep(30)
