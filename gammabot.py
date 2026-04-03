import os
import csv
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
UW_TOKEN = os.getenv("UW_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TICKER = "SPY"

# Anthropic client — used for AI signal verification + alert writing
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ─────────────────────────────────────────────
# TIMEZONE HELPER — all times PDT (UTC-7)
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
    Returns True ONLY during genuine overnight hours.
    Deliberately excludes the 1pm-6pm PDT window (right after close)
    to prevent the EOD reset from immediately triggering overnight alerts.
    
    Active windows:
    - Weekday evenings: 6pm PDT → 6am PDT next day
    - Weekends: all day Saturday and Sunday
    """
    pdt = now_pdt()
    h = pdt.hour
    weekday = pdt.weekday()  # 0=Mon, 6=Sun
    
    # Weekend = always overnight monitoring
    if weekday >= 5:
        return True
    # Weekday: only 6pm onwards OR before 6am
    # Deliberately skips 1pm-6pm to avoid post-EOD spam
    return h >= 18 or h < 6

# ─────────────────────────────────────────────
# LOGGING SYSTEM v3 — Full ML Dataset
# Everything logged automatically
# Only "notes" column is optional for you
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
    # Intraday ML features
    "vwap_distance", "price_vs_open", "session_range",
    "vol_gex_velocity", "vol_gex_direction",
    "regime_transitions", "vwap_breaks",
    "gamma_wall_above", "gamma_wall_below", "time_of_day",
    # Catalyst ML features
    "news_sentiment", "news_score",
    "catalyst_type", "catalyst_strength", "macro_override",
    # Overnight features — new
    "session_type",        # MARKET / OVERNIGHT / WEEKEND
    "futures_direction",   # UP / DOWN / FLAT (ES futures overnight)
    "overnight_vix_move",  # VIX change since close
    "overnight_news_flag", # MAJOR_EVENT / MINOR / NONE
    # Outcomes — auto at EOD
    "outcome_direction", "outcome_points",
    "signal_correct", "max_move_up", "max_move_down",
    # AI verification
    "claude_verdict", "claude_confidence",
    "claude_reasoning", "combined_score",
    # Optional
    "notes"
]

# Known economic event calendar
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
    # Core gamma bot state
    "previous_gex_state": None,
    "previous_ratio": None,
    "vwap_alert_sent": False,
    # Intelligence state
    "previous_vol_gex": None,
    "previous_oi_gex": None,
    "vol_gex_history": [],
    "regime": None,
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
    # New module state
    "last_heartbeat": 0,
    "doji_transition_sent": False,
    "last_wall_alert_price": 0,
    "current_vanna_target": None,
    "current_charm_target": None,
    "regime_transitions_today": 0,
    "vwap_breaks_today": 0,
    "session_vwap": None,
    "eod_fired_today": False,
    # Persistence tracking
    "last_git_push": 0,
    "github_csv_sha": "",   # SHA cached from GitHub — needed for API updates
    "true_session_close": None,  # Real close from yfinance — set on startup
    # Overnight state — new
    "overnight_report_sent": False,
    "last_overnight_check": 0,
    "overnight_gex_snapshot": None,   # GEX at close for comparison
    "overnight_vix_close": None,      # VIX at close for comparison
    "overnight_alerts_today": 0,      # cap overnight alerts
    "last_news_sentiment": "NEUTRAL",
    "last_catalyst_type": "NONE",
    "last_macro_override": "NO",
    # Telegram deduplication — prevents re-processing old messages
    "telegram_last_update_id": 0,
    # Change detection — skip alerts when nothing meaningful changed
    "last_logged_price": 0,
}


# ─────────────────────────────────────────────
# PERSISTENT STORAGE — GitHub auto-commit
# Survives Railway redeployments
# Requires GITHUB_TOKEN in Railway vars
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PERSISTENT STORAGE SYSTEM v2
#
# Problem solved: Railway wipes the filesystem
# on every redeploy. Without this, every bot
# update loses all CSV data — including months
# of ML training data.
#
# How it works:
#   STARTUP → pull CSV from GitHub (merge)
#   EVERY WRITE → push CSV to GitHub immediately
#   STARTUP RECOVERY → detect mid-day deploy,
#                      restore today's rows
#
# This means:
#   - 10 redeploys in one day = zero data loss
#   - Each deploy continues exactly where last left off
#   - Months of ML data are always safe
#   - Claude always has access to full history
#
# Required Railway variable: GITHUB_TOKEN
# Get at: github.com/settings/tokens (repo scope)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# GITHUB API HELPERS
# Pure HTTP — no git binary required.
# Railway containers don't have git installed,
# so we use GitHub's REST API directly via
# the requests library (already imported).
# ─────────────────────────────────────────────

def _github_headers():
    """Return auth headers for GitHub API calls."""
    token = os.getenv("GITHUB_TOKEN", "")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

def _github_repo():
    return os.getenv("GITHUB_REPO", "coding101010rizz/trading-bots")

def _github_available():
    return bool(os.getenv("GITHUB_TOKEN", ""))


def pull_csv_from_github():
    """
    Called at startup — downloads the CSV from GitHub via REST API
    and merges it with any local data already on disk.

    Uses GET /repos/{owner}/{repo}/contents/{path}
    No git binary needed — pure HTTP request.

    Merge strategy:
    - Rows keyed by (date, time, session_type)
    - GitHub rows = ground truth for historical data
    - Local rows not on GitHub (written this session) = kept
    - Result: no duplicates, no lost rows
    - Version safe: missing columns filled with "" automatically
    """
    if not _github_available():
        print("⚠️ GITHUB_TOKEN not set — running without persistence")
        print("   Data will be lost on next redeploy.")
        print("   Add GITHUB_TOKEN to Railway vars: github.com/settings/tokens")
        return False

    try:
        print("🔄 Pulling CSV from GitHub API...")
        repo = _github_repo()
        url = f"https://api.github.com/repos/{repo}/contents/{LOG_FILE}"

        resp = requests.get(url, headers=_github_headers(), timeout=15)

        if resp.status_code == 404:
            print("📝 No existing CSV on GitHub — starting fresh")
            return False

        if resp.status_code != 200:
            print(f"⚠️ GitHub API error {resp.status_code} — starting fresh")
            return False

        data = resp.json()

        # GitHub returns file content as base64
        import base64
        raw_content = base64.b64decode(data["content"]).decode("utf-8")

        # Store SHA — needed for the update (push) call later
        state["github_csv_sha"] = data.get("sha", "")

        # Parse GitHub rows
        github_rows = []
        lines = raw_content.strip().split("\n")
        if len(lines) > 1:
            reader = csv.DictReader(lines)
            for row in reader:
                # Fill any missing columns (handles version upgrades safely)
                for header in LOG_HEADERS:
                    if header not in row:
                        row[header] = ""
                github_rows.append(row)

        if not github_rows:
            print("📝 GitHub CSV is empty — starting fresh")
            return False

        # Load any rows already written locally this session
        local_rows = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", newline="") as f:
                local_rows = list(csv.DictReader(f))

        # Merge: GitHub is source of truth, keep local rows not yet pushed
        github_keys = {
            (r["date"], r["time"], r.get("session_type", "MARKET"))
            for r in github_rows
        }
        new_local_rows = [
            r for r in local_rows
            if (r["date"], r["time"], r.get("session_type", "MARKET"))
            not in github_keys
        ]
        merged_rows = github_rows + new_local_rows

        # Write merged CSV to disk
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged_rows)

        today_str = now_pdt().strftime("%Y-%m-%d")
        today_count = sum(1 for r in merged_rows if r["date"] == today_str)
        total_count = len(merged_rows)

        print(f"✅ CSV restored: {total_count} total rows "
              f"({today_count} today, {len(new_local_rows)} new local merged)")
        return True

    except Exception as e:
        print(f"CSV pull error (non-critical): {e}")
        print("Continuing with local data only")
        return False


def git_commit_log(reason="scheduled"):
    """
    Pushes the CSV to GitHub via REST API after every write.
    Uses PUT /repos/{owner}/{repo}/contents/{path}

    No git binary needed — pure HTTP.
    Rate limited to once per 60s for 'reading' calls
    to avoid GitHub API limits. EOD always pushes.
    """
    import base64

    # Rate limit for high-frequency reading calls
    now_epoch = time.time()
    last_push = state.get("last_git_push", 0)
    if reason == "reading" and now_epoch - last_push < 60:
        return  # Will push on next write or EOD

    if not _github_available():
        return  # Already warned at startup

    if not os.path.exists(LOG_FILE):
        return

    try:
        repo = _github_repo()
        today_str = now_pdt().strftime("%Y-%m-%d")

        # Read current CSV content
        with open(LOG_FILE, "rb") as f:
            content_bytes = f.read()
        encoded = base64.b64encode(content_bytes).decode("utf-8")

        # Get current SHA from GitHub (needed for update)
        # Use cached SHA if available, otherwise fetch it
        sha = state.get("github_csv_sha", "")
        if not sha:
            url = f"https://api.github.com/repos/{repo}/contents/{LOG_FILE}"
            r = requests.get(url, headers=_github_headers(), timeout=10)
            if r.status_code == 200:
                sha = r.json().get("sha", "")
            # If 404, file doesn't exist yet — first push, sha stays ""

        # PUT request — creates or updates the file
        url = f"https://api.github.com/repos/{repo}/contents/{LOG_FILE}"
        payload = {
            "message": f"Auto-log [{reason}]: SPY GEX {today_str}",
            "content": encoded,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha  # Required for updates, omit for first create

        resp = requests.put(url, headers=_github_headers(),
                            json=payload, timeout=20)

        if resp.status_code in (200, 201):
            # Cache the new SHA for the next update
            new_sha = resp.json().get("content", {}).get("sha", "")
            if new_sha:
                state["github_csv_sha"] = new_sha
            state["last_git_push"] = now_epoch
            print(f"✅ CSV pushed to GitHub [{reason}]: {today_str}")
        else:
            print(f"⚠️ GitHub push failed {resp.status_code}: {resp.text[:100]}")

    except Exception as e:
        print(f"GitHub push error (non-critical, data safe locally): {e}")


def init_log():
    """
    Called at startup — restores CSV from GitHub first,
    then creates headers if still no file exists.
    Also seeds session open/high/low from real market data
    so a mid-day deploy still has accurate numbers.
    """
    print("─" * 60)
    print("STARTUP: Initializing persistent storage...")

    # Step 1: Restore from GitHub
    restored = pull_csv_from_github()

    # Step 2: Create fresh file if nothing exists
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
        print("📝 New CSV created (no prior data on GitHub)")
    else:
        with open(LOG_FILE, "r", newline="") as f:
            existing = list(csv.DictReader(f))
        today_str = now_pdt().strftime("%Y-%m-%d")
        today_rows = [r for r in existing if r.get("date") == today_str]
        print(f"📊 CSV ready: {len(existing)} total rows, "
              f"{len(today_rows)} today")

    # Step 3: Seed session state from real market data.
    # This is the fix for mid-day deploys — the bot fetches the actual
    # 6:30am open, session high/low, and close from yfinance so that
    # eod_autofill has real numbers even if the bot just started at noon.
    pdt = now_pdt()
    h = pdt.hour
    session_data = None

    # Run this if market is open OR if we're past close (to fix EOD data)
    if (6 <= h <= 13) or (h >= 13 and pdt.weekday() < 5):
        print("🔍 Fetching real session data to seed open/high/low...")
        session_data = fetch_true_session_data()

    if session_data:
        # Only set open_price if not already set (don't overwrite live session)
        if state["open_price"] is None:
            state["open_price"] = session_data["open"]
            print(f"   Seeded open_price: ${session_data['open']}")

        # Always update session high/low with real data
        state["session_high"] = session_data["high"]
        state["session_low"]  = session_data["low"]
        print(f"   Seeded session H/L: ${session_data['high']} / "
              f"${session_data['low']}")

        # Store close for EOD autofill to use
        state["true_session_close"] = session_data["close"]
        print(f"   Seeded close: ${session_data['close']}")

    print("─" * 60)


def load_historical_context(days=30):
    """
    Loads the last N days of CSV data as a structured summary
    for Claude to use in morning briefs and signal verification.

    This is what makes the ML context useful — Claude can see
    patterns across weeks, not just today's readings.

    Returns a compact text summary suitable for Claude's prompt.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return "No historical data available yet."

        with open(LOG_FILE, "r", newline="") as f:
            all_rows = list(csv.DictReader(f))

        if not all_rows:
            return "No historical data available yet."

        # Filter to last N days of MARKET session rows
        cutoff = (now_pdt() - timedelta(days=days)).strftime("%Y-%m-%d")
        market_rows = [
            r for r in all_rows
            if r.get("date", "") >= cutoff
            and r.get("session_type", "MARKET") == "MARKET"
            and r.get("signal_correct", "") != ""
        ]

        if not market_rows:
            return f"No completed signal data in last {days} days yet."

        # Aggregate stats
        total = len(market_rows)
        correct = sum(1 for r in market_rows if r.get("signal_correct") == "YES")
        partial = sum(1 for r in market_rows if r.get("signal_correct") == "PARTIAL")
        wrong = sum(1 for r in market_rows if r.get("signal_correct") == "NO")
        win_rate = round((correct / total) * 100, 1) if total > 0 else 0

        # Regime accuracy breakdown
        regime_stats = {}
        for r in market_rows:
            regime = r.get("regime", "UNKNOWN")
            if regime not in regime_stats:
                regime_stats[regime] = {"correct": 0, "total": 0}
            regime_stats[regime]["total"] += 1
            if r.get("signal_correct") == "YES":
                regime_stats[regime]["correct"] += 1

        regime_summary = []
        for regime, stats in sorted(regime_stats.items(),
                                     key=lambda x: -x[1]["total"]):
            if stats["total"] >= 3:
                pct = round((stats["correct"] / stats["total"]) * 100, 0)
                regime_summary.append(
                    f"  {regime}: {pct}% win rate ({stats['total']} signals)"
                )

        # Average move on correct signals
        moves = []
        for r in market_rows:
            try:
                pts = float(r.get("outcome_points", 0) or 0)
                if pts != 0:
                    moves.append(abs(pts))
            except Exception:
                pass
        avg_move = round(sum(moves) / len(moves), 2) if moves else 0

        # Recent catalyst accuracy
        catalyst_rows = [r for r in market_rows if r.get("catalyst_type") != "NONE"]
        cat_correct = sum(1 for r in catalyst_rows if r.get("signal_correct") == "YES")
        cat_rate = round((cat_correct / len(catalyst_rows)) * 100, 1) if catalyst_rows else 0

        # Macro override accuracy
        override_rows = [r for r in market_rows if r.get("macro_override") == "YES"]
        ovr_correct = sum(1 for r in override_rows if r.get("signal_correct") == "YES")
        ovr_rate = round((ovr_correct / len(override_rows)) * 100, 1) if override_rows else 0

        # Most recent 5 outcomes for recency context
        recent = sorted(market_rows, key=lambda x: (x["date"], x["time"]))[-5:]
        recent_lines = []
        for r in recent:
            recent_lines.append(
                f"  {r['date']} {r['time']}: {r.get('gex_state','?')} | "
                f"{r.get('regime','?')} | Score:{r.get('conviction_score','?')} | "
                f"Outcome:{r.get('outcome_direction','?')} {r.get('outcome_points','?')}pts | "
                f"Correct:{r.get('signal_correct','?')}"
            )

        summary = (
            f"HISTORICAL PERFORMANCE (last {days} days, {total} signals):\n"
            f"  Overall win rate: {win_rate}% "
            f"(Correct:{correct} Partial:{partial} Wrong:{wrong})\n"
            f"  Average move on signals: {avg_move} pts\n"
            f"  With catalyst active: {cat_rate}% win rate "
            f"({len(catalyst_rows)} signals)\n"
            f"  Macro override YES: {ovr_rate}% win rate "
            f"({len(override_rows)} signals)\n"
            f"\nREGIME WIN RATES:\n" + "\n".join(regime_summary) +
            f"\n\nRECENT SIGNALS:\n" + "\n".join(recent_lines)
        )

        print(f"📚 Historical context loaded: {total} signals, "
              f"{win_rate}% win rate")
        return summary

    except Exception as e:
        print(f"Historical context error: {e}")
        return "Historical context unavailable."


# ─────────────────────────────────────────────
# NEWS SENTIMENT — UW API
# ─────────────────────────────────────────────
def fetch_news_sentiment():
    """
    Pulls latest market headlines from UW's news endpoint.
    UW pre-scores sentiment — no keyword matching needed.
    Returns: sentiment, score, catalyst_type, strength, macro_override
    """
    try:
        if not UW_TOKEN:
            return "NEUTRAL", 50, "NONE", 0, "NO"

        headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
        url = "https://api.unusualwhales.com/api/news/headlines"
        params = {"major_only": "true", "limit": 20}
        r = requests.get(url, headers=headers, params=params, timeout=8)

        if r.status_code != 200:
            return "NEUTRAL", 50, "NONE", 0, "NO"

        data = r.json().get("data", [])
        if not data:
            return "NEUTRAL", 50, "NONE", 0, "NO"

        bull_count = bear_count = neutral_count = 0
        geo_count = fed_count = tariff_count = major_count = 0

        geo_words = ["iran", "war", "strait", "hormuz", "military", "attack",
                     "strike", "nato", "conflict", "missile", "ceasefire",
                     "nuclear", "hezbollah", "houthi"]
        fed_words = ["federal reserve", "fed", "powell", "rate decision", "fomc",
                     "interest rate", "monetary policy", "rate hike", "rate cut"]
        tariff_words = ["tariff", "trade war", "import tax", "liberation day",
                        "trade deal", "sanctions", "reciprocal tariff"]

        for item in data[:20]:
            headline = (item.get("headline") or "").lower()
            sentiment_uw = (item.get("sentiment") or "").lower()
            is_major = item.get("is_major", False)

            if is_major:
                major_count += 1

            if sentiment_uw == "positive":
                bull_count += 2 if is_major else 1
            elif sentiment_uw == "negative":
                bear_count += 2 if is_major else 1
            else:
                neutral_count += 1

            for w in geo_words:
                if w in headline:
                    geo_count += 1
                    break
            for w in fed_words:
                if w in headline:
                    fed_count += 1
                    break
            for w in tariff_words:
                if w in headline:
                    tariff_count += 1
                    break

        today = date.today()
        if today in FED_DATES_2026 or fed_count >= 3:
            catalyst_type = "FED"
            catalyst_strength = min(100, fed_count * 15 + 50)
        elif geo_count >= 3:
            catalyst_type = "GEO"
            catalyst_strength = min(100, geo_count * 12 + 40)
        elif tariff_count >= 3:
            catalyst_type = "TARIFF"
            catalyst_strength = min(100, tariff_count * 12 + 40)
        elif today in OPEX_DATES:
            catalyst_type = "OPEX"
            catalyst_strength = 60
        else:
            catalyst_type = "NONE"
            catalyst_strength = 0

        if major_count >= 3:
            catalyst_strength = min(100, catalyst_strength + 15)

        total = bull_count + bear_count
        if total == 0:
            sentiment = "NEUTRAL"
            news_score = 50
        elif bull_count > bear_count * 1.5:
            sentiment = "BULLISH"
            news_score = min(100, int(50 + (bull_count / total) * 50))
        elif bear_count > bull_count * 1.5:
            sentiment = "BEARISH"
            news_score = max(0, int(50 - (bear_count / total) * 50))
        else:
            sentiment = "NEUTRAL"
            news_score = 50

        macro_override = "YES" if (
            catalyst_strength >= 60 or
            (catalyst_type == "GEO" and geo_count >= 4) or
            (catalyst_type == "FED" and catalyst_strength >= 50) or
            major_count >= 5
        ) else "NO"

        print(f"📰 News: {sentiment} | Catalyst: {catalyst_type} "
              f"({catalyst_strength}) | Override: {macro_override} | "
              f"Major: {major_count} | Bull:{bull_count} Bear:{bear_count}")

        return sentiment, news_score, catalyst_type, catalyst_strength, macro_override

    except Exception as e:
        print(f"News sentiment error: {e}")
        return "NEUTRAL", 50, "NONE", 0, "NO"


# ─────────────────────────────────────────────
# OVERNIGHT MONITORING ENGINE
# Tracks what institutions are doing after hours:
# - ES futures direction and magnitude
# - VIX change since close (fear building?)
# - Options flow on the overnight session
# - News catalyst detection
# - GEX snapshot comparison
# ─────────────────────────────────────────────
def fetch_overnight_data():
    """
    Fetches data relevant to overnight institutional positioning.
    Returns dict with futures, VIX change, news, and GEX if available.
    """
    try:
        result = {
            "futures_price": None,
            "futures_change_pct": None,
            "futures_direction": "FLAT",
            "vix_current": None,
            "vix_change": None,
            "vix_direction": "STABLE",
            "news_sentiment": "NEUTRAL",
            "catalyst_type": "NONE",
            "catalyst_strength": 0,
            "macro_override": "NO",
            "overnight_news_flag": "NONE",
            "gex_available": False,
            "oi_gex": None,
            "vol_gex": None,
        }

        # ES futures via yfinance (^GSPC is S&P, ES=F is futures)
        try:
            es = yf.Ticker("ES=F").history(period="2d", interval="5m")
            if not es.empty:
                if isinstance(es.columns, pd.MultiIndex):
                    es.columns = es.columns.get_level_values(0)
                current_price = float(es["Close"].iloc[-1])
                # Compare to yesterday's close (last regular session close)
                # Find close around 4pm ET (1pm PDT)
                prev_close_candidates = es[es.index.hour == 20]  # 1pm PDT = 8pm UTC
                if not prev_close_candidates.empty:
                    prev_close = float(prev_close_candidates["Close"].iloc[-1])
                else:
                    prev_close = float(es["Close"].iloc[0])
                change_pct = ((current_price - prev_close) / prev_close) * 100
                result["futures_price"] = round(current_price, 2)
                result["futures_change_pct"] = round(change_pct, 2)
                if change_pct > 0.3:
                    result["futures_direction"] = "UP"
                elif change_pct < -0.3:
                    result["futures_direction"] = "DOWN"
                else:
                    result["futures_direction"] = "FLAT"
        except Exception as e:
            print(f"Futures fetch error: {e}")

        # VIX overnight
        try:
            vix = yf.Ticker("^VIX").history(period="2d", interval="5m")
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_current = float(vix["Close"].iloc[-1])
                result["vix_current"] = round(vix_current, 2)

                # Compare to close snapshot if we have it
                vix_at_close = state.get("overnight_vix_close")
                if vix_at_close:
                    vix_change = round(vix_current - vix_at_close, 2)
                    result["vix_change"] = vix_change
                    if vix_change > 1.5:
                        result["vix_direction"] = "RISING"
                    elif vix_change < -1.5:
                        result["vix_direction"] = "FALLING"
                    else:
                        result["vix_direction"] = "STABLE"
        except Exception as e:
            print(f"VIX overnight error: {e}")

        # News overnight — same UW endpoint
        news_sentiment, news_score, catalyst_type, catalyst_strength, macro_override = \
            fetch_news_sentiment()
        result["news_sentiment"] = news_sentiment
        result["catalyst_type"] = catalyst_type
        result["catalyst_strength"] = catalyst_strength
        result["macro_override"] = macro_override

        # Overnight news significance
        if macro_override == "YES" and catalyst_strength >= 70:
            result["overnight_news_flag"] = "MAJOR_EVENT"
        elif catalyst_strength >= 40:
            result["overnight_news_flag"] = "MINOR"
        else:
            result["overnight_news_flag"] = "NONE"

        # GEX from UW (sometimes available overnight for next session)
        try:
            url = f"https://api.unusualwhales.com/api/stock/{TICKER}/spot-exposures"
            headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json().get("data", [])
            if data:
                latest = data[-1]
                oi_gex = float(latest.get("gamma_per_one_percent_move_oi", 0))
                vol_gex = float(latest.get("gamma_per_one_percent_move_vol", 0))
                if oi_gex != 0:
                    result["gex_available"] = True
                    result["oi_gex"] = oi_gex
                    result["vol_gex"] = vol_gex
        except Exception as e:
            print(f"Overnight GEX error: {e}")

        return result

    except Exception as e:
        print(f"Overnight data error: {e}")
        return {}


def write_overnight_alert_with_claude(overnight_data, alert_type="overnight_update"):
    """
    Claude writes the overnight alert in plain English.
    Explains what institutions are doing and what it means for tomorrow.
    """
    if not anthropic_client:
        return None

    try:
        pdt = now_pdt()
        now_str = pdt.strftime("%H:%M")
        day_name = pdt.strftime("%A")

        futures_dir = overnight_data.get("futures_direction", "FLAT")
        futures_chg = overnight_data.get("futures_change_pct", 0)
        vix_current = overnight_data.get("vix_current", "N/A")
        vix_change = overnight_data.get("vix_change", 0)
        vix_dir = overnight_data.get("vix_direction", "STABLE")
        news_sentiment = overnight_data.get("news_sentiment", "NEUTRAL")
        catalyst_type = overnight_data.get("catalyst_type", "NONE")
        catalyst_strength = overnight_data.get("catalyst_strength", 0)
        macro_override = overnight_data.get("macro_override", "NO")
        news_flag = overnight_data.get("overnight_news_flag", "NONE")
        gex_available = overnight_data.get("gex_available", False)
        oi_gex = overnight_data.get("oi_gex")
        vol_gex = overnight_data.get("vol_gex")

        gex_context = ""
        if gex_available and oi_gex and vol_gex:
            oi_b = round(oi_gex / 1e9, 2)
            vol_b = round(vol_gex / 1e9, 2)
            gex_context = f"\nOvernight GEX snapshot: OI={oi_b}B, Vol={vol_b}B"

        vix_close = state.get("overnight_vix_close", "unknown")
        gex_close = state.get("overnight_gex_snapshot", "unknown")

        alert_prompts = {
            "overnight_update": f"""You are writing an overnight market update for a SPY 0DTE options trader in California.
Time: {now_str} PDT ({day_name})

Overnight data:
- ES Futures: {futures_dir} {futures_chg:+.2f}%
- VIX: {vix_current} ({vix_dir}, change: {vix_change:+.2f} since close)
- VIX at close: {vix_close}
- News sentiment: {news_sentiment}
- Catalyst: {catalyst_type} (strength: {catalyst_strength}/100)
- Macro override active: {macro_override}
- News significance: {news_flag}
{gex_context}

Write a brief overnight update. Rules:
- Max 8 lines total
- What are institutions doing right now and why
- Is fear building (VIX rising) or fear leaving (VIX falling)?
- What does futures direction tell you about tomorrow's open?
- One sentence on what to watch when market opens
- Plain English — write like a trader, not a textbook
- 1-2 emojis max
- End with: tomorrow's early bias (BULLISH / BEARISH / NEUTRAL)""",

            "major_overnight_event": f"""You are writing an URGENT overnight alert for a SPY 0DTE options trader in California.
Time: {now_str} PDT — MAJOR EVENT DETECTED

Overnight data:
- ES Futures: {futures_dir} {futures_chg:+.2f}%
- VIX: {vix_current} ({vix_dir}, change: {vix_change:+.2f})
- Catalyst: {catalyst_type} (strength: {catalyst_strength}/100)
- News sentiment: {news_sentiment}
- Macro override: {macro_override}
- News significance: {news_flag}
{gex_context}

Write an urgent overnight alert. Rules:
- Max 8 lines
- Line 1: what the major event is and why it matters
- Lines 2-3: what it does to tomorrow's open mechanically
- Lines 4-5: what to do — wait, prepare puts, prepare calls
- Line 6: the risk — could this reverse?
- 2 emojis max
- Urgent but not panicked tone"""
        }

        prompt = alert_prompts.get(alert_type, alert_prompts["overnight_update"])

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()
        print(f"✍️ Claude wrote overnight alert ({len(text)} chars)")
        return text

    except Exception as e:
        print(f"Overnight Claude writer error: {e}")
        return None


def run_overnight_check():
    """
    Overnight monitoring job — runs every 90 minutes from 1pm-6am PDT.
    Tracks:
    - ES futures direction and magnitude
    - VIX change since close
    - Institutional options flow (if available)
    - Major news events
    - Builds context for tomorrow's morning brief

    Alert thresholds:
    - Futures move > 0.5% → send update
    - VIX move > 2pts → send update
    - Major news event detected → send immediately
    - Regular summary every 3 hours during evening
    """
    if not is_overnight_window():
        return

    pdt = now_pdt()
    now_epoch = time.time()
    h = pdt.hour
    now_str = pdt.strftime("%H:%M")

    # Cap overnight alerts — don't spam all night
    if state.get("overnight_alerts_today", 0) >= 3:  # was 6 — max 3 per night
        return

    # Minimum 2 hours between any overnight alert (was 60-90 min)
    last_check = state.get("last_overnight_check", 0)
    time_since_last = now_epoch - last_check
    if time_since_last < 7200:
        return

    try:
        overnight = fetch_overnight_data()
        if not overnight:
            return

        futures_dir = overnight.get("futures_direction", "FLAT")
        futures_chg = overnight.get("futures_change_pct", 0) or 0
        vix_current = overnight.get("vix_current")
        vix_change = overnight.get("vix_change") or 0
        vix_dir = overnight.get("vix_direction", "STABLE")
        news_flag = overnight.get("overnight_news_flag", "NONE")
        catalyst_type = overnight.get("catalyst_type", "NONE")
        catalyst_str = overnight.get("catalyst_strength", 0)
        macro_override = overnight.get("macro_override", "NO")
        news_sentiment = overnight.get("news_sentiment", "NEUTRAL")

        # Stricter thresholds — only genuinely notable events
        significant_futures = abs(futures_chg) >= 1.0   # was 0.5%
        significant_vix = abs(vix_change) >= 3.0         # was 2.0pts
        major_event = news_flag == "MAJOR_EVENT" and catalyst_str >= 70
        # One evening summary only at 7-8pm, never fires again after sent
        regular_evening_update = (19 <= h <= 20 and
                                  not state.get("overnight_report_sent"))

        should_alert = (significant_futures or significant_vix or
                        major_event or regular_evening_update)

        if not should_alert:
            state["last_overnight_check"] = now_epoch
            print(f"Overnight check {now_str}: quiet — "
                  f"futures {futures_chg:+.1f}%, VIX {vix_dir}")
            return

        # Determine alert type
        alert_type = "major_overnight_event" if major_event else "overnight_update"

        # Try Claude-written alert
        written = write_overnight_alert_with_claude(overnight, alert_type)

        # Format futures and VIX display
        futures_emoji = "🟢" if futures_dir == "UP" else "🔴" if futures_dir == "DOWN" else "⚪"
        vix_emoji = "😨" if vix_dir == "RISING" else "😌" if vix_dir == "FALLING" else "😐"
        event_emoji = "🚨" if major_event else "🌙"

        futures_str = (f"{futures_emoji} ES Futures: {futures_dir} "
                       f"{futures_chg:+.2f}%")
        if overnight.get("futures_price"):
            futures_str += f" (${overnight['futures_price']})"

        vix_str = (f"{vix_emoji} VIX: {vix_current} ({vix_dir})")
        if vix_change and state.get("overnight_vix_close"):
            vix_str += f" | Change: {vix_change:+.2f} since close"

        catalyst_str_display = (f"📰 {catalyst_type} | {news_sentiment} | "
                                f"Strength: {catalyst_str}/100"
                                if catalyst_type != "NONE"
                                else "📰 No major catalyst")

        if written:
            alert(
                f"{event_emoji} OVERNIGHT UPDATE — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} PDT\n\n"
                f"{futures_str}\n"
                f"{vix_str}\n"
                f"{catalyst_str_display}\n\n"
                f"{written}"
            )
        else:
            # Template fallback
            # Determine tomorrow's bias from data
            if futures_dir == "UP" and news_sentiment != "BEARISH":
                tomorrow_bias = "BULLISH LEAN"
                bias_note = "Gap up open likely. Watch for hedge unwind or fade."
            elif futures_dir == "DOWN" or (news_sentiment == "BEARISH" and catalyst_str >= 50):
                tomorrow_bias = "BEARISH LEAN"
                bias_note = "Gap down possible. Institutions rebuilding put protection."
            else:
                tomorrow_bias = "NEUTRAL"
                bias_note = "No clear overnight edge. Wait for 6:30am GEX signal."

            vix_note = ""
            if vix_dir == "RISING":
                vix_note = "VIX rising = fear building. Premium expensive at open."
            elif vix_dir == "FALLING":
                vix_note = "VIX falling = IV crush active. Vanna effects stronger."

            alert(
                f"{event_emoji} OVERNIGHT UPDATE — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} PDT\n\n"
                f"{futures_str}\n"
                f"{vix_str}\n"
                f"{catalyst_str_display}\n\n"
                f"📊 WHAT THIS MEANS\n"
                f"{vix_note}\n\n"
                f"🌅 TOMORROW'S BIAS: {tomorrow_bias}\n"
                f"{bias_note}\n\n"
                f"⚠️ Confirm with 6:30am bot signal before trading."
            )

        state["last_overnight_check"] = now_epoch
        state["overnight_alerts_today"] = state.get("overnight_alerts_today", 0) + 1
        if regular_evening_update:
            state["overnight_report_sent"] = True

        print(f"🌙 Overnight alert sent at {now_str} PDT | "
              f"Futures:{futures_chg:+.1f}% | VIX:{vix_dir} | {news_flag}")

        # Log overnight reading to CSV
        log_overnight_reading(overnight)

    except Exception as e:
        print(f"Overnight check error: {e}")
        import traceback
        traceback.print_exc()


def log_overnight_reading(overnight_data):
    """
    Logs an overnight snapshot to the CSV.
    Marked as session_type=OVERNIGHT so ML can
    distinguish daytime vs overnight readings.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return

        pdt = now_pdt()
        today_str = pdt.strftime("%Y-%m-%d")
        now_str = pdt.strftime("%H:%M")

        futures_price = overnight_data.get("futures_price") or 0
        futures_dir = overnight_data.get("futures_direction", "FLAT")
        vix_current = overnight_data.get("vix_current") or 0
        vix_change = overnight_data.get("vix_change") or 0
        news_sentiment = overnight_data.get("news_sentiment", "NEUTRAL")
        news_score = 50
        catalyst_type = overnight_data.get("catalyst_type", "NONE")
        catalyst_strength = overnight_data.get("catalyst_strength", 0)
        macro_override = overnight_data.get("macro_override", "NO")
        news_flag = overnight_data.get("overnight_news_flag", "NONE")

        # GEX if available overnight
        oi_gex = overnight_data.get("oi_gex") or 0
        vol_gex = overnight_data.get("vol_gex") or 0
        oi_b = round(oi_gex / 1e9, 4) if oi_gex else ""
        vol_b = round(vol_gex / 1e9, 4) if vol_gex else ""

        row = {
            "date": today_str,
            "time": now_str,
            "price": futures_price,
            "oi_gex_raw": oi_b,
            "vol_gex_raw": vol_b,
            "oi_gex_m": "", "vol_gex_m": "",
            "ratio": "", "gex_state": "OVERNIGHT",
            "regime": "OVERNIGHT",
            "conviction_score": "", "grade": "",
            "vix": vix_current, "vvix": "", "vix_term": "",
            "tick_approx": "", "inventory_bias": "",
            "unwind_score": "", "open_drive": "",
            "vanna_target": "", "charm_target": "",
            "calendar_flags": "", "days_to_opex": "",
            "vwap_distance": "", "price_vs_open": "",
            "session_range": "", "vol_gex_velocity": "",
            "vol_gex_direction": "", "regime_transitions": "",
            "vwap_breaks": "", "gamma_wall_above": "",
            "gamma_wall_below": "", "time_of_day": "OVERNIGHT",
            "news_sentiment": news_sentiment,
            "news_score": news_score,
            "catalyst_type": catalyst_type,
            "catalyst_strength": catalyst_strength,
            "macro_override": macro_override,
            # Overnight-specific
            "session_type": "OVERNIGHT",
            "futures_direction": futures_dir,
            "overnight_vix_move": vix_change,
            "overnight_news_flag": news_flag,
            # Outcomes blank
            "outcome_direction": "", "outcome_points": "",
            "signal_correct": "", "max_move_up": "", "max_move_down": "",
            # AI blank for overnight
            "claude_verdict": "", "claude_confidence": "",
            "claude_reasoning": "", "combined_score": "",
            "notes": ""
        }

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writerow(row)

        print(f"📝 Overnight logged: {now_str} | {futures_dir} | VIX {vix_current}")

        # Push immediately — overnight data also persisted
        git_commit_log(reason="overnight")

    except Exception as e:
        print(f"Overnight log error: {e}")


# ─────────────────────────────────────────────
# INTRADAY FEATURES
# ─────────────────────────────────────────────
def get_intraday_features(price, vol_gex):
    try:
        vwap = state.get("session_vwap")
        vwap_distance = round(price - vwap, 2) if vwap else 0

        open_price = state.get("open_price")
        price_vs_open = round(price - open_price, 2) if open_price else 0

        session_high = state.get("session_high") or price
        session_low = state.get("session_low") or price
        session_high = max(session_high, price)
        session_low = min(session_low, price)
        state["session_high"] = session_high
        state["session_low"] = session_low
        session_range = round(session_high - session_low, 2)

        history = state.get("vol_gex_history", [])
        if len(history) >= 2:
            velocity = round(vol_gex - history[-2], 4)
            vol_gex_direction = (
                "ACCELERATING" if abs(vol_gex) > abs(history[-2])
                else "DECELERATING"
            )
        else:
            velocity = 0
            vol_gex_direction = "STABLE"

        h = now_pdt().hour
        time_of_day = "EARLY" if h < 8 else "MID" if h < 11 else "LATE"

        return {
            "vwap_distance": vwap_distance,
            "price_vs_open": price_vs_open,
            "session_range": session_range,
            "vol_gex_velocity": velocity,
            "vol_gex_direction": vol_gex_direction,
            "regime_transitions": state.get("regime_transitions_today", 0),
            "vwap_breaks": state.get("vwap_breaks_today", 0),
            "gamma_wall_above": state.get("gamma_wall_above", ""),
            "gamma_wall_below": state.get("gamma_wall_below", ""),
            "time_of_day": time_of_day,
        }
    except Exception as e:
        print(f"Intraday features error: {e}")
        return {k: "" for k in [
            "vwap_distance", "price_vs_open", "session_range",
            "vol_gex_velocity", "vol_gex_direction", "regime_transitions",
            "vwap_breaks", "gamma_wall_above", "gamma_wall_below", "time_of_day"
        ]}


# ─────────────────────────────────────────────
# LOG READING — daytime
# ─────────────────────────────────────────────
def log_reading(price, oi_gex, vol_gex, oi_m, vol_m, ratio, gex_state,
                regime, conv, grade, vix_spot, vvix_val, vix_term,
                tick_approx, inventory_bias, unwind_score, open_drive,
                vanna_target, charm_target, cal_flags, days_opex,
                claude_verdict="", claude_confidence=0,
                claude_reasoning="", combined_score=0):
    try:
        now = now_pdt()
        cal_summary = (
            "QUARTER_END" if "QUARTER END TODAY" in str(cal_flags) else
            "OPEX_DAY" if "OPEX DAY" in str(cal_flags) else
            f"OPEX_IN_{days_opex}D" if days_opex else "NORMAL"
        )

        intraday = get_intraday_features(price, vol_gex)
        news_sentiment, news_score, catalyst_type, catalyst_strength, macro_override = \
            fetch_news_sentiment()

        state["last_news_sentiment"] = news_sentiment
        state["last_catalyst_type"] = catalyst_type
        state["last_macro_override"] = macro_override

        clean_grade = (grade.replace("🔥", "").replace("✅", "")
                       .replace("⚠️", "").replace("🔴", "")
                       .replace("❌", "").strip())

        row = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
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
            "inventory_bias": inventory_bias,
            "unwind_score": unwind_score,
            "open_drive": "YES" if open_drive else "NO",
            "vanna_target": vanna_target or "",
            "charm_target": charm_target or "",
            "calendar_flags": cal_summary,
            "days_to_opex": days_opex or "",
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
            "futures_direction": "",
            "overnight_vix_move": "",
            "overnight_news_flag": "",
            "outcome_direction": "",
            "outcome_points": "",
            "signal_correct": "",
            "max_move_up": "",
            "max_move_down": "",
            "claude_verdict": claude_verdict,
            "claude_confidence": claude_confidence,
            "claude_reasoning": claude_reasoning[:500] if claude_reasoning else "",
            "combined_score": combined_score,
            "notes": ""
        }

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writerow(row)

        print(f"📝 Logged: {row['time']} | {gex_state} | Score:{conv} | "
              f"News:{news_sentiment} | Catalyst:{catalyst_type}")

        # Push to GitHub immediately — data safe even if bot crashes next second
        git_commit_log(reason="reading")

    except Exception as e:
        print(f"log_reading error: {e}")


# ─────────────────────────────────────────────
# AI VERIFICATION LAYER
# ─────────────────────────────────────────────
def verify_signal_with_claude(signal_type, price, gex_state, regime,
                               vol_gex, oi_gex, ratio, vix, vvix,
                               news_sentiment, catalyst_type,
                               macro_override, conviction_score,
                               unwind_score, vanna_target, charm_target):
    if not anthropic_client:
        return "UNAVAILABLE", 0, "No API key configured", conviction_score

    try:
        vol_b = round(vol_gex / 1e9, 2)
        oi_b = round(oi_gex / 1e9, 2)

        prompt = f"""You are an expert options flow analyst reviewing a SPY 0DTE trading signal.
Analyze this market data and verify whether the bot signal is sound.

CURRENT MARKET DATA:
- SPY Price: ${price}
- GEX State: {gex_state}
- Regime: {regime}
- Vol GEX: {vol_b}B | OI GEX: {oi_b}B | Ratio: {ratio:.2f}x
- VIX: {vix} | VVIX: {vvix}
- News: {news_sentiment} | Catalyst: {catalyst_type} | Override: {macro_override}
- Unwind Score: {unwind_score}/100
- Vanna Target: ${vanna_target or 'None'} | Charm: ${charm_target or 'None'}
- Bot Conviction: {conviction_score}/100

SIGNAL: {signal_type}

Rules:
1. If macro_override YES + catalyst GEO/FED → reduce structure signal confidence
2. Vol GEX + OI GEX same sign with ratio > 1.5x → directionally confirmed
3. VVIX > 100 → velocity conditions support premium expansion
4. Unwind score > 40 → mechanical bullish pressure regardless of structure
5. Contradictions between news and GEX → higher risk

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

        import json
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        verdict = result.get("verdict", "NEUTRAL")
        confidence = int(result.get("confidence", 50))
        reasoning = result.get("reasoning", "")
        risk_factor = result.get("risk_factor", "")

        if verdict == "CONFIRM":
            combined = min(100, int((conviction_score * 0.6) + (confidence * 0.4) * 1.15))
        elif verdict == "CHALLENGE":
            combined = max(0, int((conviction_score * 0.4) + ((100 - confidence) * 0.2) * 0.7))
        else:
            combined = conviction_score

        full_reasoning = f"{reasoning} Risk: {risk_factor}"
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
        oi_b = round(oi_gex / 1e9, 2)
        now_str = now_pdt().strftime("%H:%M")

        prompts = {
            "morning_report": f"""You are writing a pre-market briefing for a SPY 0DTE options trader in California.
Time: {now_str} PDT | SPY: ${price}

Market data:
- Regime: {regime} (was {previous_regime})
- GEX State: {gex_state} | Vol: {vol_b}B | OI: {oi_b}B | Ratio: {ratio:.2f}x
- VIX: {vix} | VVIX: {vvix} | Conviction: {conviction}/100 | AI: {combined}/100
- News: {news_sentiment} | Catalyst: {catalyst_type} | Override: {macro_override}
- Unwind: {unwind_score}/100 | Vanna: ${vanna_target or 'none'} | Charm: ${charm_target or 'none'}
{extra_context}

Write a morning briefing. Rules:
- Max 10 lines total, plain English
- If historical data is provided above, reference it — e.g. "this regime has been
  correct X% of the time" or "macro override days have been unreliable recently"
- What to expect today and why
- One clear actionable recommendation
- Name the biggest risk
- End with: size and when to enter
- Write like an experienced trader to a friend
- No bullet points, no headers, 1-2 emojis max""",

            "regime_transition": f"""You are writing a trading alert for a SPY 0DTE options trader.
Time: {now_str} PDT | SPY: ${price}

What changed: {previous_regime} → {regime}
GEX: {gex_state} | Vol: {vol_b}B | OI: {oi_b}B | Ratio: {ratio:.2f}x
VIX: {vix} | VVIX: {vvix} | Conviction: {conviction}/100 | AI: {combined}/100
News: {news_sentiment} | Catalyst: {catalyst_type} | Override: {macro_override}
Vanna: ${vanna_target or 'none'} | AI verdict: {claude_verdict}
{extra_context}

Write a regime transition alert. Rules:
- Max 8 lines, no bullet points, no headers
- Line 1: what changed and why it matters now
- Lines 2-3: mechanical explanation in plain English
- Lines 4-5: exactly what to do — calls/puts, entry, target, stop
- Line 6: the one thing that kills this trade
- 1-2 emojis max""",

            "hedge_unwind": f"""You are writing a hedge unwind alert for a SPY 0DTE options trader.
Time: {now_str} PDT | SPY: ${price}

Unwind score: {unwind_score}/100 | Regime: {regime}
Vol GEX: {vol_b}B | Vanna target: ${vanna_target or 'none'}
VIX: {vix} | VVIX: {vvix} | Conviction: {conviction}/100 | AI: {combined}/100
News: {news_sentiment} | Catalyst: {catalyst_type} | AI verdict: {claude_verdict}
{extra_context}

Write a hedge unwind alert. Rules:
- Max 7 lines
- Explain why price is rising in plain English
  (institutions closing puts forces MMs to buy shares)
- Specific call target price
- Exit zone — where to sell
- One thing that kills this trade
- No jargon, 1-2 emojis max""",

            "eod_summary": f"""You are writing an end of day summary for a SPY 0DTE options trader.
{extra_context}

Write a brief EOD summary. Rules:
- Max 8 lines
- What happened today in plain English
- Was the morning signal right or wrong and why
- What to watch for tomorrow morning
- Any overnight risks
- Honest — if signal was wrong, say so
- 1-2 emojis max
- Last line: tomorrow's one-sentence bias"""
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


# ─────────────────────────────────────────────
# MODULE 1: CALENDAR / QUARTER SYSTEM
# ─────────────────────────────────────────────
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

    if days_to_opex is not None:
        if days_to_opex == 0:
            flags.append("⚡ OPEX DAY — Maximum gamma decay")
            score_bonus += 20
        elif days_to_opex <= 2:
            flags.append(f"⚡ OPEX IN {days_to_opex} DAYS — Gamma acceleration zone")
            score_bonus += 15
        elif days_to_opex <= 5:
            flags.append(f"📅 OPEX IN {days_to_opex} DAYS — Elevated gamma activity")
            score_bonus += 8
        else:
            flags.append(f"📅 OPEX IN {days_to_opex} DAYS — Standard conditions")

    return flags, score_bonus, days_to_opex


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
        oi_gex = float(latest["gamma_per_one_percent_move_oi"])
        vol_gex = float(latest["gamma_per_one_percent_move_vol"])
        price = float(latest["price"])
        return oi_gex, vol_gex, price
    except Exception as e:
        print(f"GEX fetch error: {e}")
        return None, None, None


# ─────────────────────────────────────────────
# MODULE 3: REGIME DETECTION ENGINE
# ─────────────────────────────────────────────
def detect_regime(oi_gex, vol_gex, vol_gex_history):
    if len(vol_gex_history) < 3:
        return "INSUFFICIENT_DATA", 50

    recent = vol_gex_history[-3:]
    roc = recent[-1] - recent[0]

    vol_positive = vol_gex > 0
    oi_positive = oi_gex > 0

    if vol_positive and oi_positive:
        return "BULLISH_MOMENTUM", 95
    if vol_positive and not oi_positive:
        return "HEDGE_UNWIND_CONFIRMED", 88
    if not vol_positive:
        mid_roc = recent[-1] - recent[-2]
        if roc > 0 and mid_roc > 0:
            return "HEDGE_UNWIND_EARLY", 72
        elif roc > 0 and mid_roc <= 0:
            return "TRANSITION_ZONE", 58
        else:
            roc_pct = abs(roc / recent[0]) * 100 if recent[0] != 0 else 0
            confidence = min(85, 60 + roc_pct)
            return "BEARISH_HEDGE_BUILD", int(confidence)
    return "NEUTRAL", 50


def get_regime_signal(regime, confidence, oi_b, vol_b):
    explanations = {
        "BULLISH_MOMENTUM": (
            "🟢 BULLISH MOMENTUM\n"
            "Both OI and Vol GEX positive.\n"
            "→ Calls OK but don't expect 400%+ today."
        ),
        "HEDGE_UNWIND_CONFIRMED": (
            "🚀 HEDGE UNWIND CONFIRMED\n"
            "Vol GEX flipped positive — OI still negative.\n"
            "Put SELLING is the fuel. Price rises mechanically.\n"
            "→ CALLS strongly favored. 400-700% setup."
        ),
        "HEDGE_UNWIND_EARLY": (
            "🔄 EARLY HEDGE UNWIND ← CAUGHT EARLY\n"
            "Vol GEX still negative but improving rapidly.\n"
            "→ Prepare call entry. Watch for Vol GEX flip."
        ),
        "TRANSITION_ZONE": (
            "⚠️ TRANSITION ZONE\n"
            "Vol GEX decelerating but not reversing yet.\n"
            "→ No new entries. Wait for confirmation."
        ),
        "BEARISH_HEDGE_BUILD": (
            "🔴 BEARISH HEDGE BUILD\n"
            "Vol GEX accelerating negative.\n"
            "Institutions buying put protection.\n"
            "→ PUTS favored. Calls at serious risk."
        ),
        "NEUTRAL": "⚪ NEUTRAL\n→ Stay out. Wait for regime to establish.",
        "INSUFFICIENT_DATA": "📊 COLLECTING DATA\n→ Check back in 30-45 minutes.",
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

        unwind_signals = []
        unwind_score = 0

        for contract in data:
            try:
                ctype = str(contract.get("type", "")).upper()
                volume = float(contract.get("volume", 0) or 0)
                oi = float(contract.get("open_interest", 0) or 0)
                strike = float(contract.get("strike", 0) or 0)
                execution = str(contract.get("execution_estimate", "")).upper()

                vol_oi_ratio = volume / oi if oi > 0 else 0
                is_put = "PUT" in ctype
                is_call = "CALL" in ctype
                is_sweep = "SWEEP" in execution
                is_descending = "DESCENDING" in execution

                if is_put and vol_oi_ratio >= 50 and volume >= 10000:
                    unwind_score += 25
                    unwind_signals.append(
                        f"🚀 PUT HEDGE CLOSING: ${strike:.0f}P "
                        f"Vol/OI: {round(vol_oi_ratio)}x"
                    )
                elif is_put and vol_oi_ratio >= 10 and volume >= 5000:
                    unwind_score += 10
                    unwind_signals.append(
                        f"⚠️ PUT CLOSING: ${strike:.0f}P "
                        f"Vol/OI: {round(vol_oi_ratio)}x"
                    )
                if is_put and is_descending and volume >= 5000:
                    unwind_score += 15
                    unwind_signals.append(
                        f"🔽 DESCENDING FILL PUT: ${strike:.0f}P"
                    )
                if is_call and is_sweep and volume >= 5000:
                    unwind_score += 8
                    unwind_signals.append(
                        f"📈 CALL SWEEP: ${strike:.0f}C {int(volume/1000)}K contracts"
                    )
            except Exception:
                continue

        unwind_score = min(unwind_score, 100)

        if unwind_score >= 40:
            return True, unwind_score, unwind_signals[:6], "BULLISH — Hedge unwind active"
        elif unwind_score >= 20:
            return True, unwind_score, unwind_signals[:6], "LEANING BULLISH — Early unwind signs"
        return False, unwind_score, unwind_signals[:6], "NEUTRAL"

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

        params = {"greek": "vanna", "expiry": "0dte"}
        v_data = requests.get(url, headers=headers, params=params,
                               timeout=10).json().get("data", [])
        params["greek"] = "charm"
        c_data = requests.get(url, headers=headers, params=params,
                               timeout=10).json().get("data", [])

        vanna_target = vanna_strength = charm_target = charm_strength = None

        if v_data:
            best = max(v_data, key=lambda x: float(x.get("vanna", 0) or 0))
            vanna_target = float(best.get("strike", 0))
            vanna_strength = float(best.get("vanna", 0) or 0)

        if c_data:
            worst = min(c_data, key=lambda x: float(x.get("charm", 0) or 0))
            charm_target = float(worst.get("strike", 0))
            charm_strength = float(worst.get("charm", 0) or 0)

        conflict = (vanna_target and charm_target and
                    abs(vanna_target - charm_target) <= 1.0)

        return vanna_target, vanna_strength, charm_target, charm_strength, conflict

    except Exception as e:
        print(f"Vanna/Charm error: {e}")
        return None, 0, None, 0, False


def get_vanna_charm_read(vanna_target, vanna_strength, charm_target,
                          charm_strength, price, conflict):
    pdt = now_pdt()
    minutes_since_open = (pdt.hour - 6) * 60 + pdt.minute - 30
    vanna_window_open = minutes_since_open < 270
    mins_left = max(0, 270 - minutes_since_open)
    lines = []

    if vanna_target and price:
        dist = vanna_target - price
        direction = "above" if dist > 0 else "below"
        pull = "STRONG" if abs(dist) <= 2 else "MODERATE" if abs(dist) <= 5 else "WEAK"
        lines.append(
            f"Vanna magnet: ${vanna_target} "
            f"(${abs(dist):.2f} {direction} spot) — {pull} pull"
        )
        lines.append(f"Vanna strength: {round(vanna_strength/1e6,1)}M")

    if charm_target:
        lines.append(
            f"Charm headwind: ${charm_target} — "
            f"Decay force: {round(charm_strength/1e6,1)}M"
        )

    if conflict:
        lines.append("⚠️ CONFLICT: Vanna + Charm stacked — CONSOLIDATION TRAP risk")

    if vanna_window_open:
        lines.append(f"⏰ Vanna window: ~{mins_left} min left")
    else:
        lines.append("🕐 Charm now dominant — vanna tailwind expired")

    return "\n".join(lines), vanna_window_open


# ─────────────────────────────────────────────
# MODULE 6: CONSOLIDATION DETECTOR
# ─────────────────────────────────────────────
def run_consolidation_check(current_price, current_iv, current_volume,
                             vanna_target, charm_target, conflict):
    pdt = now_pdt()
    mins = (pdt.hour - 6) * 60 + pdt.minute - 30
    if mins > 45 or mins < 5:
        return False, 0, []

    state["open_time_prices"].append(current_price)
    score = 0
    signals = []

    if vanna_target:
        prox = abs(current_price - vanna_target) / vanna_target * 100
        if prox <= 0.5:
            score += 30
            signals.append(
                f"⚠️ Price within 0.5% of vanna ${vanna_target} — magnetic stall"
            )

    if state["open_iv"] and current_iv:
        iv_chg = abs(current_iv - state["open_iv"]) / state["open_iv"] * 100
        if iv_chg < 2.0:
            score += 25
            signals.append(f"⚠️ IV only {iv_chg:.1f}% from open — no conviction")
        else:
            signals.append(f"✅ IV moving {iv_chg:.1f}% — conviction present")

    if state["open_volume"] and current_volume:
        vol_ratio = current_volume / state["open_volume"]
        if vol_ratio < 0.7:
            score += 20
            signals.append(f"⚠️ Volume only {round(vol_ratio*100)}% of open")

    if conflict:
        score += 15
        signals.append("⚠️ Vanna + charm stacked at same strike")

    if len(state["open_time_prices"]) >= 4:
        prices = state["open_time_prices"]
        changes = sum(
            1 for i in range(1, len(prices)-1)
            if (prices[i]-prices[i-1]) * (prices[i+1]-prices[i]) < 0
        )
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
        vix_hist = yf.Ticker("^VIX").history(period="5d", interval="1d")
        vvix_hist = yf.Ticker("^VVIX").history(period="5d", interval="1d")
        vix3m_hist = yf.Ticker("^VIX3M").history(period="2d", interval="1d")

        vix_spot = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else None
        vvix_val = float(vvix_hist["Close"].iloc[-1]) if not vvix_hist.empty else None
        vix3m_val = float(vix3m_hist["Close"].iloc[-1]) if not vix3m_hist.empty else None

        if vix_spot:
            state["vix_history"].append(vix_spot)
            if len(state["vix_history"]) > 5:
                state["vix_history"].pop(0)

        if vix_spot and vix3m_val:
            if vix_spot > vix3m_val * 1.02:
                vix_term = "BACKWARDATION"
                term_sig = "⚡ BACKWARDATION — Fear spike. Explosive moves."
            elif vix_spot < vix3m_val * 0.98:
                vix_term = "CONTANGO"
                term_sig = "😴 CONTANGO — Calm market. High chop risk."
            else:
                vix_term = "FLAT"
                term_sig = "⚠️ FLAT — Neutral term structure."
        else:
            vix_term = "UNKNOWN"
            term_sig = "Term structure unavailable"

        vix_momentum = ""
        if len(state["vix_history"]) >= 3:
            vix_roc = state["vix_history"][-1] - state["vix_history"][0]
            if vix_roc > 1.5:
                vix_momentum = " ↑ RISING"
            elif vix_roc < -1.5:
                vix_momentum = " ↓ FALLING — IV crush, vanna fuel"
            else:
                vix_momentum = " → STABLE"

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
                vvix_sig = f"⚠️ BORDERLINE ({round(vvix_val,1)}) — Needs confirmation"
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

        closes = spy["Close"].iloc[-10:].values.flatten().astype(float)
        opens = spy["Open"].iloc[-10:].values.flatten().astype(float)
        volumes = spy["Volume"].iloc[-10:].values.flatten().astype(float)

        up_bars = sum(1 for c, o in zip(closes, opens) if c > o)
        down_bars = sum(1 for c, o in zip(closes, opens) if c < o)
        tick_approx = (up_bars - down_bars) * 100

        state["tick_history"].append(tick_approx)
        if len(state["tick_history"]) > 6:
            state["tick_history"].pop(0)

        sustained_bull = (len(state["tick_history"]) >= 3 and
                          all(t > 300 for t in state["tick_history"][-3:]))
        sustained_bear = (len(state["tick_history"]) >= 3 and
                          all(t < -300 for t in state["tick_history"][-3:]))

        session_high = float(spy["High"].values.max())
        session_low = float(spy["Low"].values.min())
        state["session_high"] = session_high
        state["session_low"] = session_low

        avg_vol = float(np.mean(volumes[:-3])) if len(volumes) > 3 else 0
        recent_vol = float(np.mean(volumes[-3:]))
        volume_surge = recent_vol > avg_vol * 1.5

        pdt = now_pdt()
        mins_since_open = (pdt.hour - 6) * 60 + pdt.minute - 30
        open_drive = False
        if mins_since_open <= 30 and volume_surge and (sustained_bull or sustained_bear):
            open_drive = True
            state["open_drive_detected"] = True

        try:
            spy["vwap"] = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
            current_price = float(spy["Close"].iloc[-1])
            current_vwap = float(spy["vwap"].iloc[-1])
            state["session_vwap"] = current_vwap

            if current_price > current_vwap * 1.002:
                inventory_bias = "BULL ZONE"
                state["inventory_bias"] = "BULL"
            elif current_price < current_vwap * 0.998:
                inventory_bias = "BEAR ZONE"
                state["inventory_bias"] = "BEAR"
            else:
                inventory_bias = "NEUTRAL (100%)"
                state["inventory_bias"] = "NEUTRAL"
        except Exception:
            inventory_bias = "NEUTRAL"

        if tick_approx >= 600 or sustained_bull:
            tick_signal = (
                f"📈 STRONG BUYING (TICK ~+{tick_approx}) "
                f"{'— Open drive!' if open_drive else '— Accumulation'}"
            )
        elif tick_approx >= 200:
            tick_signal = f"🟡 MILD BUYING (TICK ~+{tick_approx})"
        elif tick_approx <= -600 or sustained_bear:
            tick_signal = (
                f"📉 STRONG SELLING (TICK ~{tick_approx}) "
                f"{'— Open drive DOWN!' if open_drive else '— Distribution'}"
            )
        elif tick_approx <= -200:
            tick_signal = f"🟡 MILD SELLING (TICK ~{tick_approx})"
        else:
            tick_signal = (
                f"⚪ NEUTRAL (TICK ~{tick_approx}) — no conviction\n"
                f"   Inventory: {inventory_bias}\n"
                f"   → MONITOR — wait for confirm"
            )

        return tick_signal, tick_approx, inventory_bias, open_drive

    except Exception as e:
        print(f"TICK/Inventory error: {e}")
        return "UNAVAILABLE", 0, "NEUTRAL", False


# ─────────────────────────────────────────────
# MODULE 9: CONVICTION SCORER
# ─────────────────────────────────────────────
def score_conviction(vix_spot, vvix_val, vix_term, vol_gex, prev_vol_gex,
                      regime, unwind_score, cal_bonus, vanna_window_open,
                      conflict, ratio, tick_approx, inventory_bias,
                      open_drive):
    score = 0
    checklist = []

    if vvix_val:
        if vvix_val >= 100:
            score += 25
            checklist.append(f"✅ VVIX {round(vvix_val,1)} ≥ 100 — Velocity day (+25)")
        elif vvix_val >= 90:
            score += 18
            checklist.append(f"✅ VVIX {round(vvix_val,1)} — Active (+18)")
        elif vvix_val >= 85:
            score += 10
            checklist.append(f"⚠️ VVIX {round(vvix_val,1)} — Borderline (+10)")
        else:
            checklist.append(f"❌ VVIX {round(vvix_val,1)} < 85 — Chop risk (+0)")

    if vix_term == "BACKWARDATION":
        score += 15
        checklist.append("✅ VIX Backwardation (+15)")
    elif vix_term == "FLAT":
        score += 7
        checklist.append("⚠️ VIX Flat (+7)")
    else:
        checklist.append("❌ VIX Contango — calm (+0)")

    if len(state["vix_history"]) >= 3:
        vix_roc = state["vix_history"][-1] - state["vix_history"][0]
        if abs(vix_roc) > 1.5:
            score += 5
            checklist.append(f"✅ VIX momentum (+5)")

    regime_pts = {
        "HEDGE_UNWIND_CONFIRMED": 20, "BULLISH_MOMENTUM": 18,
        "BEARISH_HEDGE_BUILD": 16, "HEDGE_UNWIND_EARLY": 14,
        "TRANSITION_ZONE": 8, "NEUTRAL": 0, "INSUFFICIENT_DATA": 0,
    }
    rpts = regime_pts.get(regime, 0)
    score += rpts
    emoji = "✅" if rpts >= 14 else "⚠️" if rpts >= 8 else "❌"
    checklist.append(f"{emoji} Regime: {regime} (+{rpts})")

    if unwind_score >= 40:
        score += 15
        checklist.append(f"✅ Hedge unwind confirmed ({unwind_score}/100) (+15)")
    elif unwind_score >= 20:
        score += 8
        checklist.append(f"⚠️ Early unwind signs (+8)")
    else:
        checklist.append("❌ No unwind detected (+0)")

    cal_pts = min(cal_bonus, 15)
    score += cal_pts
    if cal_pts >= 5:
        checklist.append(f"✅ Calendar: +{cal_pts}")

    if vanna_window_open:
        score += 10
        checklist.append("✅ Vanna window open (+10)")
    else:
        checklist.append("❌ Vanna window closed (+0)")

    if abs(tick_approx) >= 600:
        score += 10
        checklist.append(f"✅ TICK strong (+10)")
    elif abs(tick_approx) >= 300:
        score += 5
        checklist.append("⚠️ TICK moderate (+5)")
    else:
        checklist.append(f"❌ TICK neutral ({tick_approx}) (+0)")

    if inventory_bias in ["BULL ZONE", "BEAR ZONE"]:
        score += 5
        checklist.append(f"✅ Inventory: {inventory_bias} (+5)")
    else:
        checklist.append("❌ Inventory: NEUTRAL (+0)")

    if open_drive:
        score += 10
        checklist.append("🚀 OPEN DRIVE DETECTED (+10)")

    if conflict:
        score -= 20
        checklist.append("🚨 Vanna/charm conflict (-20)")

    score = max(0, min(100, score))

    if score >= 80:
        grade, rec = "A+ 🔥", "FULL SIZE. 400-700% day setup confirmed."
    elif score >= 65:
        grade, rec = "B+ ✅", "NORMAL SIZE. 200-400% realistic."
    elif score >= 50:
        grade, rec = "C ⚠️", "HALF SIZE ONLY. Wait for open confirmation."
    elif score >= 35:
        grade, rec = "D 🔴", "MINIMAL or sit out. High chop risk."
    else:
        grade, rec = "F ❌", "DO NOT TRADE. Theta will destroy premium."

    return score, grade, rec, checklist


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_gex_state(oi_gex, vol_gex):
    if oi_gex == 0:
        return "UNKNOWN"
    same_sign = (oi_gex < 0 and vol_gex < 0) or (oi_gex > 0 and vol_gex > 0)
    if not same_sign:
        return "COUNTER"
    ratio = abs(vol_gex) / abs(oi_gex)
    if ratio < 1.2:
        return "NEUTRAL"
    elif ratio < 1.5:
        return "WATCH"
    direction = "BEARISH" if oi_gex < 0 else "BULLISH"
    return f"DIRECTIONAL_{direction}"


def format_gex(value_m):
    abs_val = abs(value_m)
    sign = "-" if value_m < 0 else ""
    if abs_val >= 1000:
        return f"{sign}{abs_val/1000:.1f}B"
    elif abs_val >= 1:
        return f"{sign}{abs_val:.1f}M"
    return f"{sign}{abs_val*1000:.0f}K"


def fmt(value_m):
    a = abs(value_m)
    s = "-" if value_m < 0 else "+"
    if a >= 1000:
        return f"{s}{a/1000:.1f}B"
    elif a >= 1:
        return f"{s}{a:.1f}M"
    return f"{s}{a*1000:.0f}K"


async def _send(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)


def alert(text):
    asyncio.run(_send(text))


def fetch_true_session_data():
    """
    Fetches the actual open, high, low, close for today's session
    from yfinance using 1-minute bars.

    Called at startup when the bot deploys mid-day or after close.
    This gives eod_autofill the real numbers regardless of when
    the bot started — so a 12pm deploy still gets the 6:30am open.

    Returns: dict with open, high, low, close, or None on failure.
    """
    try:
        spy = yf.download("SPY", period="1d", interval="1m", progress=False)
        if spy.empty:
            return None

        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        # Convert index to PDT for filtering
        spy.index = spy.index.tz_convert("America/Los_Angeles")

        # Filter to regular market hours only: 6:30am–1:00pm PDT
        market_bars = spy.between_time("06:30", "13:00")
        if market_bars.empty:
            return None

        true_open  = float(market_bars["Open"].iloc[0])
        true_high  = float(market_bars["High"].max())
        true_low   = float(market_bars["Low"].min())
        true_close = float(market_bars["Close"].iloc[-1])

        print(f"📈 True session data: O={true_open} H={true_high} "
              f"L={true_low} C={true_close}")

        return {
            "open":  round(true_open,  2),
            "high":  round(true_high,  2),
            "low":   round(true_low,   2),
            "close": round(true_close, 2),
        }

    except Exception as e:
        print(f"fetch_true_session_data error: {e}")
        return None


def get_vwap():
    try:
        spy = yf.download("SPY", period="1d", interval="5m", progress=False)
        if spy.empty:
            return None, None, None, None, None
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy["vwap"] = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
        cp = float(spy["Close"].iloc[-1])
        cv = float(spy["vwap"].iloc[-1])
        pp = float(spy["Close"].iloc[-2]) if len(spy) > 1 else cp
        pv = float(spy["vwap"].iloc[-2]) if len(spy) > 1 else cv
        vol = float(spy["Volume"].iloc[-1])
        return cp, cv, pp, pv, vol
    except Exception:
        return None, None, None, None, None


# ─────────────────────────────────────────────
# VWAP CROSS ALERT
# ─────────────────────────────────────────────
def check_vwap():
    if not is_market_open():
        return
    gex_s = state["previous_gex_state"] or ""
    if "DIRECTIONAL" not in gex_s:
        state["vwap_alert_sent"] = False
        return

    r = get_vwap()
    if r[0] is None:
        return
    cp, cv, pp, pv, _ = r
    now_str = now_pdt().strftime("%H:%M")
    bearish = "BEARISH" in gex_s
    bullish = "BULLISH" in gex_s

    if bearish and not state["vwap_alert_sent"]:
        if pp >= pv and cp < cv:
            alert(
                f"🔽 VWAP CROSS — BEARISH ENTRY\n"
                f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
                f"Regime: {state['regime']} | {now_str} PDT\n\n"
                f"⚠️ Confirm candle close below VWAP before entering."
            )
            state["vwap_alert_sent"] = True
    elif bullish and not state["vwap_alert_sent"]:
        if pp <= pv and cp > cv:
            alert(
                f"🔼 VWAP CROSS — BULLISH ENTRY\n"
                f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
                f"Regime: {state['regime']} | {now_str} PDT\n\n"
                f"⚠️ Confirm candle close above VWAP before entering."
            )
            state["vwap_alert_sent"] = True

    if (bearish and cp > cv) or (bullish and cp < cv):
        state["vwap_alert_sent"] = False


# ─────────────────────────────────────────────
# HEARTBEAT WATCHDOG
# ─────────────────────────────────────────────
def check_heartbeat():
    if not is_market_open():
        return
    try:
        now_epoch = time.time()
        if now_epoch - state.get("last_heartbeat", 0) < 3600:
            return

        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, _, _, _ = r
        now_str = now_pdt().strftime("%H:%M")
        vwap_dist = round(cp - cv, 2)
        vwap_side = "above" if vwap_dist > 0 else "below"

        alert(
            f"💓 BOT ALIVE — SPY\n"
            f"{'─'*30}\n"
            f"{now_str} PDT | ${round(cp,2)}\n\n"
            f"Regime: {state.get('regime', 'UNKNOWN')}\n"
            f"GEX: {state.get('previous_gex_state', 'UNKNOWN')}\n"
            f"Score: {state.get('last_conviction_score', 0)}/100\n"
            f"VWAP: ${round(cv,2)} "
            f"(${abs(vwap_dist):.2f} {vwap_side})\n\n"
            f"Bot running ✅"
        )
        state["last_heartbeat"] = now_epoch

    except Exception as e:
        print(f"Heartbeat error: {e}")


# ─────────────────────────────────────────────
# DOJI TRANSITION DETECTOR
# ─────────────────────────────────────────────
def check_doji_transition():
    if not is_market_open():
        return
    try:
        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, _, _, _ = r

        if abs(cp - cv) > 0.75:
            state["doji_transition_sent"] = False
            return

        history = state.get("vol_gex_history", [])
        if len(history) < 3:
            return

        recent = history[-3:]
        roc_recent = abs(recent[-1]) - abs(recent[-2])
        roc_older = abs(recent[-2]) - abs(recent[-3])
        decelerating = roc_older < 0 and roc_recent > roc_older

        if not decelerating:
            return

        regime = state.get("regime", "")
        if regime in ["HEDGE_UNWIND_CONFIRMED", "BULLISH_MOMENTUM"]:
            return
        if state.get("doji_transition_sent"):
            return

        now_str = now_pdt().strftime("%H:%M")
        vol_gex_current = recent[-1]
        vol_gex_prev = recent[-2]

        if vol_gex_current < 0 and roc_recent > 0:
            direction = "BEARISH → BULLISH"
            action = "Watch for Vol GEX flip positive → Calls on VWAP break above"
        elif vol_gex_current > 0 and roc_recent < 0:
            direction = "BULLISH → BEARISH"
            action = "Watch for Vol GEX flip negative → Puts on VWAP break below"
        else:
            direction = "CONSOLIDATING"
            action = "No clear direction yet → Wait for commitment"

        alert(
            f"🔄 DOJI TRANSITION FORMING — SPY\n"
            f"{'─'*35}\n"
            f"{now_str} PDT | Price: ${round(cp,2)}\n"
            f"VWAP: ${round(cv,2)} (${abs(cp-cv):.2f} away)\n\n"
            f"Transition: {direction}\n"
            f"Vol GEX decelerating — momentum losing steam\n\n"
            f"Action: {action}\n\n"
            f"⚠️ NOT A TRADE SIGNAL YET — wait for Vol GEX to confirm"
        )
        state["doji_transition_sent"] = True

    except Exception as e:
        print(f"Doji transition error: {e}")


# ─────────────────────────────────────────────
# GAMMA WALL APPROACH / TP ALERT
# ─────────────────────────────────────────────
def check_gamma_wall_approach():
    if not is_market_open():
        return
    try:
        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, _, _, _ = r

        vanna_target = state.get("current_vanna_target")
        if not vanna_target:
            return

        now_str = now_pdt().strftime("%H:%M")
        last_wall_alert = state.get("last_wall_alert_price", 0)
        if abs(cp - last_wall_alert) < 2.0:
            return

        dist_to_vanna = vanna_target - cp
        abs_dist = abs(dist_to_vanna)

        if abs_dist > 1.5:
            return

        approaching_from = "below" if dist_to_vanna > 0 else "above"

        if approaching_from == "below":
            alert(
                f"🎯 VANNA TARGET APPROACHING — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} PDT | Price: ${round(cp,2)}\n"
                f"Vanna target: ${vanna_target} (${abs_dist:.2f} away)\n\n"
                f"💰 IF HOLDING CALLS — THIS IS YOUR EXIT ZONE\n"
                f"Sell between ${round(vanna_target-0.5,0)}-${vanna_target}\n"
                f"Charm reverses above this level. Don't hold past it.\n\n"
                f"After target hit: watch Vol GEX for put re-entry signal."
            )
        else:
            alert(
                f"🎯 SUPPORT ZONE APPROACHING — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} PDT | Price: ${round(cp,2)}\n"
                f"Vanna support: ${vanna_target} (${abs_dist:.2f} away)\n\n"
                f"💰 IF HOLDING PUTS — Consider partial profit here\n"
                f"Vanna support may cause bounce. Sell half, keep half.\n\n"
                f"If wall breaks: hold remaining puts toward next level."
            )

        state["last_wall_alert_price"] = cp

    except Exception as e:
        print(f"Gamma wall error: {e}")


# ─────────────────────────────────────────────
# CONSOLIDATION JOB
# ─────────────────────────────────────────────
def check_consolidation_job():
    if not is_market_open():
        return

    pdt = now_pdt()
    mins = (pdt.hour - 6) * 60 + pdt.minute - 30
    if mins > 45 or mins < 5 or state["consolidation_alert_sent"]:
        return

    try:
        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, _, _, vol = r

        if state["open_price"] is None:
            state["open_price"] = cp
        if state["open_volume"] is None:
            state["open_volume"] = vol

        vix_data = yf.Ticker("^VIX").history(period="1d", interval="5m")
        iv = float(vix_data["Close"].iloc[-1]) if not vix_data.empty else None
        if state["open_iv"] is None and iv:
            state["open_iv"] = iv

        vt, vs, ct, cs, conflict = fetch_vanna_charm()
        is_cons, score, signals = run_consolidation_check(cp, iv, vol, vt, ct, conflict)
        now_str = pdt.strftime("%H:%M")

        if is_cons:
            sigs = "\n".join(signals)
            alert(
                f"🚨 CONSOLIDATION TRAP WARNING — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} | ${cp:.2f} | Score: {score}/100\n\n"
                f"⚠️ SIGNALS\n{sigs}\n\n"
                f"Vanna + charm forces canceling = premium decays fast.\n\n"
                f"🚫 DO NOT TRADE THIS OPEN\n"
                f"Wait for >1% move with volume OR TICK ±600 sustained.\n"
                f"Re-assess after 10:00am PDT."
            )
            state["consolidation_alert_sent"] = True
            state["consolidation_gex_state"] = state.get("previous_gex_state", "")

    except Exception as e:
        print(f"Consolidation job error: {e}")


# ─────────────────────────────────────────────
# END OF DAY AUTO-FILL
# ─────────────────────────────────────────────
def eod_autofill(close_price):
    try:
        today_str = now_pdt().strftime("%Y-%m-%d")

        if not os.path.exists(LOG_FILE):
            return

        rows = []
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        today_rows = [r for r in rows
                      if r["date"] == today_str and r.get("session_type") == "MARKET"]

        # ── Determine true open price ──────────────────────────────────────
        # Priority order:
        # 1. Real session data fetched from yfinance at startup (most accurate)
        # 2. First logged row's price (if bot was running all day)
        # 3. close_price as fallback (last resort — will show 0pt move)
        true_open = state.get("open_price")   # seeded from yfinance in init_log

        if true_open is None and today_rows:
            # Bot was running and logged rows — use first row
            true_open = float(today_rows[0]["price"])
            print(f"EOD: using first logged price as open: ${true_open}")

        if true_open is None:
            # Mid-day deploy with no logged rows and no yfinance data
            # Try one more time to get real session data
            print("EOD: no open price available, fetching from yfinance...")
            session_data = fetch_true_session_data()
            if session_data:
                true_open = session_data["open"]
                state["session_high"] = session_data["high"]
                state["session_low"]  = session_data["low"]
            else:
                true_open = close_price  # absolute last resort

        # ── Use real close from yfinance if available ──────────────────────
        # If bot deployed after close, close_price is whatever GEX shows now.
        # The true_session_close from init_log is more accurate.
        true_close = state.get("true_session_close") or close_price

        open_price = true_open
        point_move = round(true_close - open_price, 2)

        if abs(point_move) < 1.0:
            direction = "CHOP"
        elif point_move > 0:
            direction = "UP"
        else:
            direction = "DOWN"

        # ── Detect mid-day deploy (few or no logged rows) ─────────────────
        mid_day_deploy = len(today_rows) <= 2
        if mid_day_deploy:
            print(f"⚠️ EOD: mid-day deploy detected ({len(today_rows)} rows logged). "
                  f"Using yfinance data for open/high/low.")

        morning_signal = None
        for r in today_rows:
            if "DIRECTIONAL" in r.get("gex_state", ""):
                morning_signal = r["gex_state"]
                break
            elif r.get("gex_state") in ["NEUTRAL", "WATCH", "COUNTER"]:
                morning_signal = r["gex_state"]
                break

        if morning_signal and "DIRECTIONAL" in morning_signal:
            if "BEARISH" in morning_signal and direction == "DOWN":
                correct = "YES"
            elif "BULLISH" in morning_signal and direction == "UP":
                correct = "YES"
            elif direction == "CHOP":
                correct = "PARTIAL"
            else:
                correct = "NO"
        elif morning_signal in ["NEUTRAL", "WATCH", "COUNTER"]:
            correct = "YES" if direction == "CHOP" else "PARTIAL"
        elif mid_day_deploy:
            correct = "PARTIAL"   # Can't fairly judge a signal we didn't see
        else:
            correct = ""

        # Use real session high/low (seeded from yfinance in init_log)
        session_high = state.get("session_high") or true_close
        session_low  = state.get("session_low")  or true_close

        updated = 0
        for r in rows:
            if r["date"] == today_str and r["outcome_direction"] == "":
                r["outcome_direction"] = direction
                r["outcome_points"] = point_move
                r["signal_correct"] = correct
                r["max_move_up"] = round(session_high - open_price, 2) if session_high else ""
                r["max_move_down"] = round(open_price - session_low, 2) if session_low else ""
                updated += 1

        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

        # Save VIX and GEX snapshots for overnight comparison
        state["overnight_vix_close"] = state["vix_history"][-1] if state["vix_history"] else None
        state["overnight_gex_snapshot"] = {
            "oi": state.get("previous_oi_gex"),
            "vol": state.get("previous_vol_gex")
        }
        # NOTE: Do NOT reset overnight_alerts_today or last_overnight_check here.
        # Those must persist so the 2hr guard works after EOD.
        # They reset naturally at midnight via midnight_reset().

        # Commit to GitHub — EOD forces a push regardless of rate limit
        git_commit_log(reason="eod")

        now_str = now_pdt().strftime("%H:%M")
        signal_emoji = "✅" if correct == "YES" else "❌" if correct == "NO" else "⚠️"
        max_up = round(session_high - open_price, 2)
        max_down = round(open_price - session_low, 2)

        eod_context = (
            f"Date: {today_str}\n"
            f"Open: ${round(open_price,2)} | Close: ${round(true_close,2)}\n"
            f"Move: {'+' if point_move > 0 else ''}{point_move} pts ({direction})\n"
            f"Max up: +{max_up} | Max down: -{max_down}\n"
            f"Morning signal: {morning_signal or 'None'}\n"
            f"Signal correct: {correct}\n"
            f"Rows logged by bot today: {len(today_rows)}"
            + (" (⚠️ mid-day deploy — session data from yfinance, not live bot)"
               if mid_day_deploy else "") +
            f"\nOvernight monitoring will continue until 6am tomorrow.\n\n"
            f"{load_historical_context(days=30)}"
        )

        written = write_alert_with_claude(
            alert_type="eod_summary",
            price=close_price,
            gex_state=state.get("previous_gex_state", ""),
            regime=state.get("previous_regime", ""),
            vol_gex=state.get("previous_vol_gex", 0) or 0,
            oi_gex=state.get("previous_oi_gex", 0) or 0,
            ratio=0, vix=0, vvix=0,
            conviction=0, combined=0, unwind_score=0,
            vanna_target=None, charm_target=None,
            news_sentiment=state.get("last_news_sentiment", "NEUTRAL"),
            catalyst_type=state.get("last_catalyst_type", "NONE"),
            macro_override=state.get("last_macro_override", "NO"),
            flow_dir="", previous_regime="", previous_gex_state="",
            claude_verdict="",
            extra_context=eod_context
        )

        if written:
            alert(
                f"📊 END OF DAY — SPY\n"
                f"{'─'*35}\n"
                f"{today_str} | {now_str} PDT\n"
                f"${round(open_price,2)} → ${round(true_close,2)} "
                f"({'+' if point_move > 0 else ''}{point_move} pts) "
                f"{signal_emoji}\n"
                + (f"⚠️ Bot deployed mid-day — session data from yfinance\n"
                   if mid_day_deploy else "") +
                f"\n{written}\n\n"
                f"💾 {updated} rows saved\n"
                f"🌙 Overnight monitoring active"
            )
        else:
            alert(
                f"📊 END OF DAY — SPY\n"
                f"{'─'*35}\n"
                f"{today_str} | {now_str} PDT\n\n"
                f"Open: ${round(open_price,2)} → Close: ${round(true_close,2)}\n"
                f"Move: {'+' if point_move > 0 else ''}{point_move} pts ({direction})\n"
                f"Max Up: +{max_up} | Max Down: -{max_down}\n"
                + (f"⚠️ Mid-day deploy — {len(today_rows)} rows logged\n"
                   f"   (open/high/low from yfinance, not live bot)\n"
                   if mid_day_deploy else "") +
                f"\nSignal: {morning_signal or 'None'}\n"
                f"Correct: {signal_emoji} {correct}\n\n"
                f"📝 {updated} rows logged\n"
                f"💾 CSV saved | 💬 /notes to add context\n"
                f"🌙 Overnight monitoring active"
            )

        print(f"EOD complete — {updated} rows | {direction} {point_move}pts")

    except Exception as e:
        print(f"EOD autofill error: {e}")


# ─────────────────────────────────────────────
# MAIN JOB — daytime
# ─────────────────────────────────────────────
def run_job():
    pdt = now_pdt()
    now_str = pdt.strftime("%H:%M")
    h = pdt.hour
    m = pdt.minute
    weekday = pdt.weekday()  # 0=Mon, 6=Sun

    # ── Intelligent activity guard ─────────────────────────────
    # Don't kill the bot on a blunt time check — instead check
    # whether there's anything worth running for.
    #
    # ALWAYS run if:
    #   1. Market is open (Mon-Fri 6:00am-1:00pm PDT)
    #   2. Pre-market window (Mon-Fri 5:30am-6:00am PDT)
    #   3. EOD has not fired yet on a trading day (catch late data)
    #
    # SKIP if ALL of these are true:
    #   - Not a trading day OR market is closed
    #   - EOD already fired (or it's a weekend)
    #   - Futures are quiet (< 1% move) — no overnight catalyst
    #   - VIX is stable (no fear spike)
    #   - No major news catalyst active
    # ───────────────────────────────────────────────────────────

    is_trading_day = weekday < 5
    market_open = is_trading_day and (6 <= h < 13 or (h == 13 and m == 0))
    pre_market = is_trading_day and (h == 5 and m >= 30)
    eod_fired = state.get("eod_fired_today", False)

    # Always run during market hours or pre-market
    if market_open or pre_market:
        pass  # proceed normally

    # After hours / weekend — check if anything is actually happening
    else:
        # Check futures and VIX for overnight activity
        futures_active = False
        vix_active = False
        news_active = False

        try:
            es = yf.Ticker("ES=F").history(period="1d", interval="5m")
            if not es.empty:
                if isinstance(es.columns, pd.MultiIndex):
                    es.columns = es.columns.get_level_values(0)
                current = float(es["Close"].iloc[-1])
                prev = float(es["Close"].iloc[0])
                chg = abs((current - prev) / prev * 100)
                futures_active = chg >= 1.0  # 1%+ futures move
        except Exception:
            pass

        try:
            vix_hist = state.get("vix_history", [])
            if len(vix_hist) >= 2:
                vix_active = abs(vix_hist[-1] - vix_hist[0]) >= 2.0
        except Exception:
            pass

        # Check last known catalyst from state
        last_catalyst = state.get("last_catalyst_type", "NONE")
        last_macro = state.get("last_macro_override", "NO")
        news_active = last_macro == "YES" or last_catalyst in ["FED", "GEO"]

        # If nothing notable is happening, skip
        if not futures_active and not vix_active and not news_active:
            reason = []
            if not is_trading_day:
                reason.append(f"weekend ({pdt.strftime('%A')})")
            elif eod_fired:
                reason.append("EOD fired, market closed")
            else:
                reason.append(f"after hours ({now_str} PDT)")
            reason.append("futures quiet, VIX stable, no catalyst")
            print(f"run_job skipped — {' | '.join(reason)}")
            return

        # Something notable IS happening after hours — run but note it
        print(f"\n{'='*60}\nJob (after-hours activity detected): {now_str} PDT")
        print(f"Futures active: {futures_active} | VIX active: {vix_active} | News: {news_active}")
        print(f"{'='*60}")
        # Don't fire EOD again or send duplicate alerts
        # Just log the data silently for ML purposes
        # (EOD guard below handles this)

    print(f"\n{'='*60}\nJob: {now_str} PDT\n{'='*60}")

    try:
        oi_gex, vol_gex, price = fetch_gex()
        if oi_gex is None:
            print("No GEX data")
            return
        if vol_gex == 0:
            print(f"Pre-market. OI: {round(oi_gex/1e9,2)}B | Price: ${price}")
            return

        state["vol_gex_history"].append(vol_gex)
        if len(state["vol_gex_history"]) > 10:
            state["vol_gex_history"].pop(0)

        ratio = abs(vol_gex) / abs(oi_gex) if oi_gex != 0 else 0
        gex_state = get_gex_state(oi_gex, vol_gex)
        oi_b = round(oi_gex / 1e9, 2)
        vol_b = round(vol_gex / 1e9, 2)
        oi_m = oi_gex / (price * 6.31) / 1e6
        vol_m = vol_gex / (price * 6.31) / 1e6
        oi_fmt = format_gex(oi_m)
        vol_fmt = format_gex(vol_m)
        ratio_r = round(ratio, 2)

        # ── CHANGE DETECTION GATE ──────────────────────────────────────
        # If nothing meaningful has changed since last reading,
        # skip alerts and AI calls — just log silently and return.
        # This prevents redundant messages on flat/choppy mornings.
        #
        # A reading is considered "changed" if ANY of these are true:
        #   1. GEX state changed (e.g. COUNTER → DIRECTIONAL)
        #   2. Price moved more than $1.50 since last reading
        #   3. Vol GEX changed more than 5% (momentum building/dying)
        #   4. Ratio moved more than 0.2x (conviction shifting)
        #   5. It's the first reading of the day (morning brief always fires)
        #   6. EOD is due (h >= 13)
        #   7. It's within the morning brief window (6:25-7:15am PDT)
        prev_gex_state = state.get("previous_gex_state")
        prev_price = state.get("last_logged_price", 0)
        prev_vol = state.get("previous_vol_gex") or vol_gex
        prev_ratio = state.get("previous_ratio") or ratio

        gex_state_changed = gex_state != prev_gex_state
        price_moved = abs(price - prev_price) >= 1.50
        vol_gex_shifted = (abs(vol_gex - prev_vol) / abs(prev_vol) * 100
                           >= 5.0) if prev_vol != 0 else True
        ratio_shifted = abs(ratio - prev_ratio) >= 0.20
        first_reading = prev_gex_state is None
        eod_due = h >= 13 and not state.get("eod_fired_today")
        morning_brief_window = (h == 6 and pdt.minute >= 25) or (h == 7 and pdt.minute <= 15)

        something_changed = (
            gex_state_changed or price_moved or vol_gex_shifted or
            ratio_shifted or first_reading or eod_due or morning_brief_window
        )

        if not something_changed:
            # Log the reading silently for ML data but skip all alerts/AI
            print(f"  → No meaningful change — silent log only "
                  f"(price ${price}, {gex_state}, ratio {ratio_r}x)")
            # Still detect regime for state tracking
            regime, reg_conf = detect_regime(oi_gex, vol_gex,
                                             state["vol_gex_history"])
            vix_spot, vvix_val, vix_term, _, _, _ = fetch_vix_data()
            cal_flags_s, cal_bonus, days_opex = get_calendar_flags()
            tick_signal, tick_approx, inventory_bias, open_drive = \
                fetch_tick_and_inventory()
            vt, vs, ct, cs, conflict = fetch_vanna_charm()
            conv, grade, rec, _ = score_conviction(
                vix_spot, vvix_val, vix_term, vol_gex,
                prev_vol, regime, 0, cal_bonus, False,
                conflict, ratio, tick_approx, inventory_bias, open_drive
            )
            log_reading(
                price=price, oi_gex=oi_gex, vol_gex=vol_gex,
                oi_m=oi_m, vol_m=vol_m, ratio=ratio,
                gex_state=gex_state, regime=regime,
                conv=conv, grade=grade,
                vix_spot=vix_spot, vvix_val=vvix_val, vix_term=vix_term,
                tick_approx=tick_approx, inventory_bias=inventory_bias,
                unwind_score=0, open_drive=open_drive,
                vanna_target=vt, charm_target=ct,
                cal_flags=[], days_opex=days_opex,
                claude_verdict="SKIPPED", claude_confidence=0,
                claude_reasoning="No change detected", combined_score=conv
            )
            state["previous_gex_state"] = gex_state
            state["previous_ratio"] = ratio
            state["previous_vol_gex"] = vol_gex
            state["previous_oi_gex"] = oi_gex
            state["last_logged_price"] = price
            return
        # ── END CHANGE DETECTION ───────────────────────────────────────

        vix_spot, vvix_val, vix_term, term_sig, vix_sig, vvix_sig = fetch_vix_data()
        vt, vs, ct, cs, conflict = fetch_vanna_charm()
        vc_text, vanna_window = get_vanna_charm_read(vt, vs, ct, cs, price, conflict)

        state["current_vanna_target"] = vt
        state["current_charm_target"] = ct

        unwind_det, unwind_score, unwind_sigs, flow_dir = fetch_hedge_unwind_signals()
        regime, reg_conf = detect_regime(oi_gex, vol_gex, state["vol_gex_history"])
        reg_signal = get_regime_signal(regime, reg_conf, oi_b, vol_b)
        cal_flags, cal_bonus, days_opex = get_calendar_flags()
        tick_signal, tick_approx, inventory_bias, open_drive = fetch_tick_and_inventory()

        conv, grade, rec, checklist = score_conviction(
            vix_spot, vvix_val, vix_term, vol_gex,
            state["previous_vol_gex"], regime, unwind_score,
            cal_bonus, vanna_window, conflict, ratio,
            tick_approx, inventory_bias, open_drive
        )

        print(f"${price} | {gex_state} | Regime:{regime} | Score:{conv}")
        print(f"OI:{oi_b}B ({oi_fmt}) | VOL:{vol_b}B ({vol_fmt}) | Ratio:{ratio_r}x")

        # AI verification — only for meaningful signals
        should_verify = (
            conv >= 50 or
            regime in ["BEARISH_HEDGE_BUILD", "HEDGE_UNWIND_CONFIRMED",
                       "HEDGE_UNWIND_EARLY", "BULLISH_MOMENTUM"] or
            unwind_score >= 40 or
            gex_state != state["previous_gex_state"] or
            regime != state["previous_regime"]
        )

        claude_verdict = "SKIPPED"
        claude_confidence = 0
        claude_reasoning = "Low conviction — verification skipped"
        combined_score = conv

        if should_verify and anthropic_client:
            signal_desc = (f"{gex_state} regime={regime} "
                           f"conviction={conv}/100 unwind={unwind_score}/100")
            claude_verdict, claude_confidence, claude_reasoning, combined_score = \
                verify_signal_with_claude(
                    signal_type=signal_desc, price=price,
                    gex_state=gex_state, regime=regime,
                    vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                    vix=vix_spot, vvix=vvix_val,
                    news_sentiment=state.get("last_news_sentiment", "NEUTRAL"),
                    catalyst_type=state.get("last_catalyst_type", "NONE"),
                    macro_override=state.get("last_macro_override", "NO"),
                    conviction_score=conv, unwind_score=unwind_score,
                    vanna_target=vt, charm_target=ct
                )

        verdict_emoji = (
            "✅" if claude_verdict == "CONFIRM" else
            "⚠️" if claude_verdict == "CHALLENGE" else
            "➖" if claude_verdict == "NEUTRAL" else "🔇"
        )
        ai_line = (
            f"\n🤖 AI: {verdict_emoji} {claude_verdict} ({claude_confidence}%)\n"
            f"→ {claude_reasoning[:300]}\n"
            f"Combined: {combined_score}/100"
            if claude_verdict not in ["SKIPPED", "UNAVAILABLE"]
            else ""
        )

        log_reading(
            price=price, oi_gex=oi_gex, vol_gex=vol_gex,
            oi_m=oi_m, vol_m=vol_m, ratio=ratio,
            gex_state=gex_state, regime=regime,
            conv=conv, grade=grade,
            vix_spot=vix_spot, vvix_val=vvix_val, vix_term=vix_term,
            tick_approx=tick_approx, inventory_bias=inventory_bias,
            unwind_score=unwind_score, open_drive=open_drive,
            vanna_target=vt, charm_target=ct,
            cal_flags=cal_flags, days_opex=days_opex,
            claude_verdict=claude_verdict, claude_confidence=claude_confidence,
            claude_reasoning=claude_reasoning, combined_score=combined_score
        )

        # GEX state change alert
        if gex_state != state["previous_gex_state"]:
            emojis = {
                "NEUTRAL": "⚪", "WATCH": "⚠️",
                "DIRECTIONAL_BEARISH": "🔴",
                "DIRECTIONAL_BULLISH": "🟢",
                "COUNTER": "🔄"
            }
            emoji = emojis.get(gex_state, "❓")
            alert(
                f"{emoji} SPY GEX SIGNAL\n\n"
                f"State: {gex_state}\n"
                f"OI Net GEX: {oi_fmt} ({oi_b}B)\n"
                f"VOL Net GEX: {vol_fmt} ({vol_b}B)\n"
                f"Ratio: {ratio_r}x | Spot: ${price} | {now_str} PDT\n\n"
                f"📊 REGIME: {regime} ({reg_conf}%)\n"
                f"🎯 CONVICTION: {conv}/100 — {grade}\n"
                f"→ {rec}"
                f"{ai_line}"
            )

        # Regime change alert
        if regime != state["previous_regime"] and regime != "INSUFFICIENT_DATA":
            prev = state["previous_regime"] or "None"
            key_transitions = [
                ("BEARISH_HEDGE_BUILD", "TRANSITION_ZONE"),
                ("BEARISH_HEDGE_BUILD", "HEDGE_UNWIND_EARLY"),
                ("BEARISH_HEDGE_BUILD", "HEDGE_UNWIND_CONFIRMED"),
                ("TRANSITION_ZONE", "HEDGE_UNWIND_EARLY"),
                ("HEDGE_UNWIND_EARLY", "HEDGE_UNWIND_CONFIRMED"),
            ]
            is_key = any(prev == a and regime == b for a, b in key_transitions)
            r_emoji = "🚨" if is_key else "🔄"

            cl_text = "\n".join(checklist)

            written = write_alert_with_claude(
                alert_type="regime_transition",
                price=price, gex_state=gex_state, regime=regime,
                vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                vix=vix_spot, vvix=vvix_val,
                conviction=conv, combined=combined_score,
                unwind_score=unwind_score, vanna_target=vt, charm_target=ct,
                news_sentiment=state.get("last_news_sentiment", "NEUTRAL"),
                catalyst_type=state.get("last_catalyst_type", "NONE"),
                macro_override=state.get("last_macro_override", "NO"),
                flow_dir=flow_dir, previous_regime=prev,
                previous_gex_state=state.get("previous_gex_state", ""),
                claude_verdict=claude_verdict
            )

            if written:
                alert(
                    f"{r_emoji} REGIME TRANSITION — SPY\n"
                    f"{'─'*35}\n"
                    f"{prev} → {regime} | {now_str} PDT\n\n"
                    f"{written}"
                )
            else:
                alert(
                    f"{r_emoji} REGIME TRANSITION — SPY\n"
                    f"{'─'*35}\n"
                    f"NEW: {regime} | WAS: {prev}\n"
                    f"Confidence: {reg_conf}% | {now_str} PDT | ${price}\n\n"
                    f"{reg_signal}\n\n"
                    f"📈 VANNA/CHARM\n{vc_text}\n\n"
                    f"🎯 CONVICTION: {conv}/100 — {grade}\n→ {rec}\n\n"
                    f"📋 CHECKLIST\n{cl_text}"
                    f"{ai_line}"
                )
            # Track regime transitions for ML
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
                news_sentiment=state.get("last_news_sentiment", "NEUTRAL"),
                catalyst_type=state.get("last_catalyst_type", "NONE"),
                macro_override=state.get("last_macro_override", "NO"),
                flow_dir=flow_dir,
                previous_regime=state.get("previous_regime", ""),
                previous_gex_state=state.get("previous_gex_state", ""),
                claude_verdict=claude_verdict,
                extra_context=f"Flow signals:\n{uw_text}"
            )
            if written:
                alert(
                    f"🚀 HEDGE UNWIND — SPY\n"
                    f"{'─'*35}\n"
                    f"{now_str} PDT | ${price} | Score: {unwind_score}/100\n\n"
                    f"{written}"
                )
            else:
                alert(
                    f"🚀 HEDGE UNWIND DETECTED — SPY\n"
                    f"{'─'*35}\n"
                    f"Score: {unwind_score}/100 | {now_str} PDT | ${price}\n\n"
                    f"📊 FLOW\n{uw_text}\n\n"
                    f"Institutions selling puts = MMs buy shares back.\n"
                    f"📈 VANNA TARGET: ${vt}\n\n"
                    f"🎯 {conv}/100 — {grade}\n→ {rec}"
                    f"{ai_line}"
                )
            state["hedge_unwind_alert_sent"] = True
            state["last_unwind_alert_time"] = time.time()

        if unwind_score < 20:
            state["hedge_unwind_alert_sent"] = False

        # Conviction spike
        if ("DIRECTIONAL" in gex_state and state["previous_ratio"]
                and ratio - state["previous_ratio"] > 0.3):
            direction = "BEARISH" if oi_gex < 0 else "BULLISH"
            alert(
                f"📈 CONVICTION INCREASING — SPY\n\n"
                f"Direction: {direction}\n"
                f"Ratio: {ratio_r}x (was {round(state['previous_ratio'],2)}x)\n"
                f"OI: {oi_fmt} | VOL: {vol_fmt} | ${price} | {now_str}\n\n"
                f"VIX: {vix_sig} | VVIX: {vvix_sig}\n"
                f"🎯 Score: {conv}/100 — {grade}\n→ {rec}"
            )

        # Morning velocity report
        h, m = pdt.hour, pdt.minute
        if (h == 6 and m >= 25) or (h == 7 and m <= 15):
            conviction_changed = (
                state["last_conviction_score"] is not None and
                abs(conv - state["last_conviction_score"]) >= 15
            )
            if not state["velocity_score_sent"] or conviction_changed:
                cal_text = "\n".join(cal_flags) if cal_flags else "No special events"
                update_tag = "🔄 UPDATED — " if conviction_changed else ""

                # Historical performance context — Claude uses this to calibrate
                # its morning brief based on how well the bot has been working
                hist_context = load_historical_context(days=30)

                written = write_alert_with_claude(
                    alert_type="morning_report",
                    price=price, gex_state=gex_state, regime=regime,
                    vol_gex=vol_gex, oi_gex=oi_gex, ratio=ratio,
                    vix=vix_spot, vvix=vvix_val,
                    conviction=conv, combined=combined_score,
                    unwind_score=unwind_score, vanna_target=vt, charm_target=ct,
                    news_sentiment=state.get("last_news_sentiment", "NEUTRAL"),
                    catalyst_type=state.get("last_catalyst_type", "NONE"),
                    macro_override=state.get("last_macro_override", "NO"),
                    flow_dir=flow_dir,
                    previous_regime=state.get("previous_regime", ""),
                    previous_gex_state=state.get("previous_gex_state", ""),
                    claude_verdict=claude_verdict,
                    extra_context=(
                        f"Calendar: {cal_text}\n"
                        f"VIX: {vix_sig}\nVVIX: {vvix_sig}\n\n"
                        f"{hist_context}"
                    )
                )

                if written:
                    alert(
                        f"🌅 {update_tag}MORNING BRIEF — SPY\n"
                        f"{'─'*35}\n"
                        f"{now_str} PDT | ${price} | Score: {combined_score}/100\n\n"
                        f"{written}"
                    )
                else:
                    cl_text = "\n".join(checklist)
                    alert(
                        f"🌅 {update_tag}PRE-MARKET REPORT — SPY\n"
                        f"{'─'*35}\n"
                        f"{now_str} PDT | ${price}\n\n"
                        f"🎯 CONVICTION: {conv}/100 — {grade}\n→ {rec}\n\n"
                        f"📋 CHECKLIST\n{cl_text}\n\n"
                        f"VIX: {vix_sig}\nVVIX: {vvix_sig}"
                    )
                state["velocity_score_sent"] = True

        # Alert flag resets
        if state["consolidation_alert_sent"]:
            if gex_state != state.get("consolidation_gex_state"):
                state["consolidation_alert_sent"] = False

        now_epoch = time.time()
        if (state["hedge_unwind_alert_sent"] and unwind_score >= 60
                and now_epoch - state.get("last_unwind_alert_time", 0) > 2700):
            state["hedge_unwind_alert_sent"] = False

        # Mid-session update every 90 min
        last_summary = state.get("last_summary_time", 0)
        if (is_market_open() and 6 <= h <= 12
                and now_epoch - last_summary > 5400):
            alert(
                f"📊 MID-SESSION UPDATE — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} PDT | ${price}\n\n"
                f"GEX: {gex_state} | Regime: {regime}\n"
                f"Ratio: {ratio_r}x | OI: {oi_fmt} | VOL: {vol_fmt}\n\n"
                f"VIX: {vix_sig} | VVIX: {vvix_sig}\n"
                f"Flow: {flow_dir} | Unwind: {unwind_score}/100\n\n"
                f"🎯 {conv}/100 — {grade}\n→ {rec}"
            )
            state["last_summary_time"] = now_epoch

        # EOD — fires once at 1pm PDT
        if h >= 13 and not state.get("eod_fired_today"):
            eod_autofill(price)
            state["eod_fired_today"] = True
            # Reset daily intraday state
            state["velocity_score_sent"] = False
            state["consolidation_alert_sent"] = False
            state["hedge_unwind_alert_sent"] = False
            state["last_unwind_alert_time"] = 0
            state["last_summary_time"] = 0
            state["consolidation_gex_state"] = None
            state["open_time_prices"] = []
            state["open_price"] = None
            state["open_iv"] = None
            state["open_volume"] = None
            state["open_drive_detected"] = False
            state["tick_history"] = []
            state["session_high"] = None
            state["session_low"] = None
            state["regime_transitions_today"] = 0
            state["vwap_breaks_today"] = 0
            state["session_vwap"] = None
            state["doji_transition_sent"] = False
            state["last_wall_alert_price"] = 0
            state["last_heartbeat"] = 0

        # Update state
        state["previous_gex_state"] = gex_state
        state["previous_ratio"] = ratio
        state["previous_vol_gex"] = vol_gex
        state["previous_oi_gex"] = oi_gex
        state["previous_regime"] = regime
        state["regime"] = regime
        state["last_conviction_score"] = conv

    except Exception as e:
        print(f"run_job error: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────
# TELEGRAM COMMANDS
# /status — bot health check
# /notes [text] — add context to today's log
# /overnight — manual overnight snapshot
# ─────────────────────────────────────────────
async def handle_telegram_updates():
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        # Use offset to only fetch NEW updates — never re-process old ones
        # offset = last_update_id + 1 tells Telegram to discard everything before it
        last_offset = state.get("telegram_last_update_id", 0)
        offset = last_offset + 1 if last_offset > 0 else None

        updates = await bot.get_updates(timeout=5, offset=offset)
        today_str = now_pdt().strftime("%Y-%m-%d")

        for update in updates:
            # Track highest update_id seen — Telegram won't resend these
            if update.update_id > state.get("telegram_last_update_id", 0):
                state["telegram_last_update_id"] = update.update_id

            if not update.message:
                continue
            text = (update.message.text or "").strip()

            # ── /status ──
            if text == "/status":
                rows_today = 0
                overnight_rows = 0
                last_time = "none"
                csv_exists = os.path.exists(LOG_FILE)

                if csv_exists:
                    try:
                        with open(LOG_FILE, "r", newline="") as f:
                            all_rows = list(csv.DictReader(f))
                            today_rows = [r for r in all_rows if r.get("date") == today_str]
                            rows_today = len([r for r in today_rows
                                             if r.get("session_type") == "MARKET"])
                            overnight_rows = len([r for r in today_rows
                                                 if r.get("session_type") == "OVERNIGHT"])
                            if today_rows:
                                last_time = today_rows[-1].get("time", "?")
                    except Exception:
                        pass

                uw_ok = "✅" if UW_TOKEN else "❌"
                tg_ok = "✅" if TELEGRAM_TOKEN else "❌"
                ai_ok = "✅" if anthropic_client else "❌"
                gh_ok = "✅" if os.getenv("GITHUB_TOKEN") else "❌"
                now_str = now_pdt().strftime("%H:%M")

                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        f"📊 BOT STATUS\n"
                        f"{'─'*30}\n"
                        f"Time: {now_str} PDT\n"
                        f"Market open: {'YES' if is_market_open() else 'NO'}\n"
                        f"Overnight window: {'YES' if is_overnight_window() else 'NO'}\n\n"
                        f"📝 Today ({today_str}):\n"
                        f"Market rows: {rows_today}\n"
                        f"Overnight rows: {overnight_rows}\n"
                        f"Last reading: {last_time} PDT\n"
                        f"CSV: {'YES ✅' if csv_exists else 'NO ❌'}\n\n"
                        f"🔄 State:\n"
                        f"Regime: {state.get('regime', 'unknown')}\n"
                        f"Score: {state.get('last_conviction_score', 0)}/100\n\n"
                        f"🔑 API keys:\n"
                        f"UW: {uw_ok} | Telegram: {tg_ok}\n"
                        f"Claude: {ai_ok} | GitHub: {gh_ok}\n\n"
                        f"💡 GitHub ❌? Add GITHUB_TOKEN to Railway vars.\n"
                        f"   Get at: github.com/settings/tokens"
                    )
                )
                continue

            # ── /overnight — manual snapshot ──
            if text == "/overnight":
                now_str = now_pdt().strftime("%H:%M")
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"🌙 Fetching overnight snapshot at {now_str} PDT..."
                )
                overnight = fetch_overnight_data()
                written = write_overnight_alert_with_claude(overnight, "overnight_update")

                futures_chg = overnight.get("futures_change_pct", 0) or 0
                futures_dir = overnight.get("futures_direction", "FLAT")
                vix_current = overnight.get("vix_current", "N/A")
                vix_dir = overnight.get("vix_direction", "STABLE")
                catalyst = overnight.get("catalyst_type", "NONE")

                if written:
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=(
                            f"🌙 OVERNIGHT SNAPSHOT — SPY\n"
                            f"{'─'*35}\n"
                            f"{now_str} PDT\n\n"
                            f"ES Futures: {futures_dir} {futures_chg:+.2f}%\n"
                            f"VIX: {vix_current} ({vix_dir})\n"
                            f"Catalyst: {catalyst}\n\n"
                            f"{written}"
                        )
                    )
                else:
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=(
                            f"🌙 OVERNIGHT SNAPSHOT — SPY\n"
                            f"{'─'*35}\n"
                            f"{now_str} PDT\n\n"
                            f"ES Futures: {futures_dir} {futures_chg:+.2f}%\n"
                            f"VIX: {vix_current} ({vix_dir})\n"
                            f"Catalyst: {catalyst}\n\n"
                            f"Full update in next scheduled check."
                        )
                    )
                log_overnight_reading(overnight)
                continue

            # ── /notes [text] ──
            if text.startswith("/notes"):
                note = text[6:].strip()

                # Handle /notes sent without text
                if not note:
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=(
                            "💬 Usage: /notes [your context]\n\n"
                            "Examples:\n"
                            "/notes Iran ceasefire drove rally\n"
                            "/notes Fed pause rumor bearish\n"
                            "/notes No catalyst today — structure only"
                        )
                    )
                    continue

                if not os.path.exists(LOG_FILE):
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text="⚠️ No log file found yet. Will be created at next market reading."
                    )
                    continue

                rows = []
                with open(LOG_FILE, "r", newline="") as f:
                    rows = list(csv.DictReader(f))

                updated = 0
                for r in rows:
                    if r["date"] == today_str:
                        r["notes"] = note
                        updated += 1

                with open(LOG_FILE, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
                    writer.writeheader()
                    writer.writerows(rows)

                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        f"✅ Note saved to {updated} rows\n"
                        f"Date: {today_str}\n"
                        f"Note: {note}"
                    )
                )

    except Exception as e:
        print(f"Telegram command error: {e}")


def check_telegram_commands():
    asyncio.run(handle_telegram_updates())


# ─────────────────────────────────────────────
# MIDNIGHT RESET
# ─────────────────────────────────────────────
def midnight_reset():
    """Resets daily flags at midnight PDT for the new trading day."""
    pdt = now_pdt()
    if pdt.hour == 0 and pdt.minute < 1:  # Only fires in the 00:00 minute
        state["eod_fired_today"] = False
        # Reset overnight counters for the new day
        state["overnight_alerts_today"] = 0
        state["overnight_report_sent"] = False
        state["last_overnight_check"] = 0
        print("🌙 Midnight reset — daily flags cleared for new trading day")


# ─────────────────────────────────────────────
# SCHEDULE
# All times in UTC (PDT = UTC-7)
# ─────────────────────────────────────────────

# Daytime jobs (market hours)
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

# Every 5 min jobs (market hours only — guarded internally)
schedule.every(5).minutes.do(check_vwap)
schedule.every(5).minutes.do(check_consolidation_job)
schedule.every(5).minutes.do(check_telegram_commands)
schedule.every(5).minutes.do(check_doji_transition)
schedule.every(5).minutes.do(check_gamma_wall_approach)
schedule.every(5).minutes.do(midnight_reset)

# Heartbeat — market hours (guarded internally)
schedule.every(60).minutes.do(check_heartbeat)

# Overnight monitoring — runs every 30 min
# The function itself checks if it's appropriate to alert
# Overnight monitoring — checks every 90 min but only alerts on notable events
# Alert thresholds: futures >1%, VIX change >3pts, major news, or 7-8pm summary
schedule.every(90).minutes.do(run_overnight_check)

print("SPY UNIFIED BOT v5.1 — PERSISTENT ML + OVERNIGHT")
print("=" * 60)
print("Daytime Modules:")
print("  1. GEX Core + State Change Alerts")
print("  2. VWAP Cross Alerts")
print("  3. Conviction Spike Alerts")
print("  4. Regime Detection Engine")
print("  5. Hedge Unwind Detector")
print("  6. Vanna / Charm Engine")
print("  7. Consolidation Trap Detector")
print("  8. VIX / VVIX + Momentum")
print("  9. TICK + Inventory Precision")
print(" 10. Calendar / Quarter System")
print(" 11. Unified Conviction Scorer")
print(" 12. Claude AI Verification + Alert Writer")
print(" 13. Heartbeat Watchdog (60min)")
print(" 14. Doji Transition Detector")
print(" 15. Gamma Wall / TP Zone Alert")
print("-" * 60)
print("Overnight Modules:")
print(" 16. ES Futures Tracker")
print(" 17. VIX Overnight Change Monitor")
print(" 18. Institutional News Catalyst Detector")
print(" 19. Claude Overnight Alert Writer")
print(" 20. Overnight CSV Logger")
print(" 21. /overnight Telegram Command")
print("-" * 60)
print("Data Persistence (NEW):")
print(" 22. Startup CSV pull from GitHub (merge, never overwrite)")
print(" 23. Push to GitHub after EVERY write (not just EOD)")
print(" 24. Historical context injected into Claude morning brief")
print(" 25. load_historical_context() — win rates, regime accuracy")
print(" 26. Redeploy-safe: 10 updates same day = zero data loss")
print("=" * 60)
print("Timezone: All times PDT (UTC-7)")
print("Schedule: Daytime 6:00am-1:00pm | Overnight: continuous")
print("Telegram: /status /notes /overnight")
print("=" * 60)
print()
print("⚠️  GitHub ❌ in /status?")
print("   Add GITHUB_TOKEN to Railway environment variables.")
print("   Get at: github.com/settings/tokens (repo scope only)")
print("   Without this, CSV is wiped on every redeploy.")
print("   IMPORTANT: Until this is set, months of data are at risk.")
print("=" * 60)

init_log()

# Only run job immediately on startup if market is open or pre-market
# Prevents EOD/alerts firing on every afternoon redeploy
print("Bot started — waiting for scheduled jobs.")
print("Next market open: 6:00am PDT")

while True:
    schedule.run_pending()
    time.sleep(30)
