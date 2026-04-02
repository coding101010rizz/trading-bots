import os
import csv
import requests
import schedule
import time
import asyncio
import yfinance as yf
import numpy as np
from datetime import datetime, date, timezone, timedelta
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

# ─────────────────────────────────────────────
# LOGGING SYSTEM v2 — Full ML Dataset
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
    # Intraday ML features — all auto
    "vwap_distance",
    "price_vs_open",
    "session_range",
    "vol_gex_velocity",
    "vol_gex_direction",
    "regime_transitions",
    "vwap_breaks",
    "gamma_wall_above",
    "gamma_wall_below",
    "time_of_day",
    # Catalyst ML features — all auto
    "news_sentiment",
    "news_score",
    "catalyst_type",
    "catalyst_strength",
    "macro_override",
    # Outcomes — all auto at EOD
    "outcome_direction",
    "outcome_points",
    "signal_correct",
    "max_move_up",
    "max_move_down",
    # Only this needs you — optional
    "notes"
]

# Known economic event calendar
FED_DATES_2026 = [
    date(2026, 1, 29), date(2026, 3, 19),
    date(2026, 5, 7),  date(2026, 6, 18),
    date(2026, 7, 30), date(2026, 9, 17),
    date(2026, 11, 5), date(2026, 12, 16),
]

def fetch_news_sentiment():
    """
    Pulls latest market headlines from UW's news endpoint.
    Uses the same UW_TOKEN already in Railway — no new credentials.
    UW pre-scores sentiment so no keyword matching needed.
    Returns: sentiment, score, catalyst_type, strength, macro_override
    """
    try:
        if not UW_TOKEN:
            return "NEUTRAL", 50, "NONE", 0, "NO"

        headers = {
            "Authorization": f"Bearer {UW_TOKEN}",
            "Accept": "application/json"
        }

        # Pull major market news — broad market view
        url = "https://api.unusualwhales.com/api/news/headlines"
        params = {
            "major_only": "true",
            "limit": 20
        }
        r = requests.get(url, headers=headers, params=params, timeout=8)

        if r.status_code != 200:
            print(f"UW news error: {r.status_code}")
            return "NEUTRAL", 50, "NONE", 0, "NO"

        data = r.json().get("data", [])
        if not data:
            return "NEUTRAL", 50, "NONE", 0, "NO"

        # UW already scores sentiment — use it directly
        bull_count = 0
        bear_count = 0
        neutral_count = 0
        geo_count = 0
        fed_count = 0
        tariff_count = 0
        major_count = 0

        geo_words = [
            "iran", "war", "strait", "hormuz", "military",
            "attack", "strike", "nato", "conflict", "missile",
            "ceasefire", "nuclear", "hezbollah", "houthi"
        ]
        fed_words = [
            "federal reserve", "fed", "powell", "rate decision",
            "fomc", "interest rate", "monetary policy", "rate hike",
            "rate cut", "basis points"
        ]
        tariff_words = [
            "tariff", "trade war", "import tax", "liberation day",
            "trade deal", "sanctions", "export controls",
            "reciprocal tariff", "trade deficit"
        ]

        headlines_text = []

        for item in data[:20]:
            headline = (item.get("headline") or "").lower()
            sentiment_uw = (item.get("sentiment") or "").lower()
            is_major = item.get("is_major", False)
            headlines_text.append(headline)

            if is_major:
                major_count += 1

            # Use UW's pre-scored sentiment
            if sentiment_uw == "positive":
                bull_count += 2 if is_major else 1
            elif sentiment_uw == "negative":
                bear_count += 2 if is_major else 1
            else:
                neutral_count += 1

            # Keyword catalyst detection
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

        # Determine catalyst type — priority order
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

        # Boost catalyst strength if multiple major headlines
        if major_count >= 3:
            catalyst_strength = min(100, catalyst_strength + 15)

        # Determine overall sentiment from UW scores
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

        # Macro override — news likely overrides GEX structure
        macro_override = "YES" if (
            catalyst_strength >= 60 or
            (catalyst_type == "GEO" and geo_count >= 4) or
            (catalyst_type == "FED" and catalyst_strength >= 50) or
            (catalyst_type == "TARIFF" and catalyst_strength >= 60) or
            major_count >= 5
        ) else "NO"

        # Log top headlines for console
        print(f"📰 News: {sentiment} | Catalyst: {catalyst_type} "
              f"({catalyst_strength}) | Override: {macro_override} | "
              f"Major: {major_count} | Bull:{bull_count} Bear:{bear_count}")

        return sentiment, news_score, catalyst_type, catalyst_strength, macro_override

    except Exception as e:
        print(f"News sentiment error: {e}")
        return "NEUTRAL", 50, "NONE", 0, "NO"


def get_intraday_features(price, vol_gex):
    """
    Calculates intraday ML features automatically from session state.
    No manual input required.
    """
    try:
        # VWAP distance
        vwap = state.get("session_vwap")
        vwap_distance = round(price - vwap, 2) if vwap else 0

        # Price vs open
        open_price = state.get("open_price")
        price_vs_open = round(price - open_price, 2) if open_price else 0

        # Session range
        session_high = state.get("session_high") or price
        session_low = state.get("session_low") or price
        session_high = max(session_high, price)
        session_low = min(session_low, price)
        state["session_high"] = session_high
        state["session_low"] = session_low
        session_range = round(session_high - session_low, 2)

        # Vol GEX velocity
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

        # Regime transitions today
        regime_transitions = state.get("regime_transitions_today", 0)

        # VWAP breaks today
        vwap_breaks = state.get("vwap_breaks_today", 0)

        # Gamma walls (simple approximation from OI concentration)
        gamma_wall_above = state.get("gamma_wall_above", "")
        gamma_wall_below = state.get("gamma_wall_below", "")

        # Time of day (PDT)
        h = now_pdt().hour
        if h < 8:
            time_of_day = "EARLY"
        elif h < 11:
            time_of_day = "MID"
        else:
            time_of_day = "LATE"

        return {
            "vwap_distance": vwap_distance,
            "price_vs_open": price_vs_open,
            "session_range": session_range,
            "vol_gex_velocity": velocity,
            "vol_gex_direction": vol_gex_direction,
            "regime_transitions": regime_transitions,
            "vwap_breaks": vwap_breaks,
            "gamma_wall_above": gamma_wall_above,
            "gamma_wall_below": gamma_wall_below,
            "time_of_day": time_of_day,
        }
    except Exception as e:
        print(f"Intraday features error: {e}")
        return {k: "" for k in [
            "vwap_distance", "price_vs_open", "session_range",
            "vol_gex_velocity", "vol_gex_direction", "regime_transitions",
            "vwap_breaks", "gamma_wall_above", "gamma_wall_below", "time_of_day"
        ]}


def git_commit_log():
    """
    Auto-commits the CSV to GitHub after every EOD write.
    Requires GITHUB_TOKEN in Railway environment variables.
    Data survives Railway redeploys permanently.

    Setup in Railway:
    Variable name: GITHUB_TOKEN
    Value: your GitHub personal access token
    Get one at: github.com/settings/tokens
    Needs repo scope only
    """
    try:
        import subprocess

        github_token = os.getenv("GITHUB_TOKEN", "")
        github_repo = os.getenv("GITHUB_REPO", "coding101010rizz/trading-bots")

        if not github_token:
            print("⚠️ GITHUB_TOKEN not set — CSV not committed to GitHub")
            print("   Add GITHUB_TOKEN to Railway environment variables")
            print("   Get token at: github.com/settings/tokens")
            return

        today_str = date.today().strftime("%Y-%m-%d")

        # Configure git with token authentication
        remote_url = f"https://{github_token}@github.com/{github_repo}.git"
        subprocess.run(
            ["git", "remote", "set-url", "origin", remote_url],
            check=True, capture_output=True
        )

        # Configure git identity for the commit
        subprocess.run(
            ["git", "config", "user.email", "gammabot@railway.app"],
            check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "GammaBot"],
            check=True, capture_output=True
        )

        # Add and commit the CSV
        subprocess.run(
            ["git", "add", LOG_FILE],
            check=True, capture_output=True
        )

        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "diff", "--staged", "--quiet"],
            capture_output=True
        )
        if result.returncode == 0:
            print("📝 No new CSV data to commit today")
            return

        subprocess.run([
            "git", "commit", "-m",
            f"Auto-log: SPY GEX data {today_str}"
        ], check=True, capture_output=True)

        subprocess.run(
            ["git", "push", "origin", "main"],
            check=True, capture_output=True
        )

        print(f"✅ CSV committed to GitHub: {today_str}")

    except subprocess.CalledProcessError as e:
        print(f"Git commit error: {e}")
        print("CSV data saved locally on Railway")
        print("Check GITHUB_TOKEN has repo write access")
    except Exception as e:
        print(f"Git commit error (non-critical): {e}")


def init_log():
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
        print(f"Log file created: {LOG_FILE}")

def log_reading(price, oi_gex, vol_gex, oi_m, vol_m, ratio, gex_state,
                regime, conv, grade, vix_spot, vvix_val, vix_term,
                tick_approx, inventory_bias, unwind_score, open_drive,
                vanna_target, charm_target, cal_flags, days_opex):
    """
    Fully automated logging — no manual input required.
    Fetches news sentiment, calculates intraday features,
    and writes complete ML-ready row to CSV.
    """
    try:
        now = datetime.now()
        cal_summary = (
            "QUARTER_END" if "QUARTER END TODAY" in str(cal_flags) else
            "OPEX_DAY" if "OPEX DAY" in str(cal_flags) else
            f"OPEX_IN_{days_opex}D" if days_opex else "NORMAL"
        )

        # Fetch all automated features
        intraday = get_intraday_features(price, vol_gex)
        news_sentiment, news_score, catalyst_type, catalyst_strength, macro_override = fetch_news_sentiment()

        clean_grade = grade.replace("🔥","").replace("✅","").replace("⚠️","").replace("🔴","").replace("❌","").strip()

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
            # Intraday features — all auto
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
            # Catalyst features — all auto
            "news_sentiment": news_sentiment,
            "news_score": news_score,
            "catalyst_type": catalyst_type,
            "catalyst_strength": catalyst_strength,
            "macro_override": macro_override,
            # Outcomes — filled by EOD autofill
            "outcome_direction": "",
            "outcome_points": "",
            "signal_correct": "",
            "max_move_up": "",
            "max_move_down": "",
            # Only optional field for you
            "notes": ""
        }

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writerow(row)

        print(f"📝 Logged: {row['time']} | {gex_state} | Score:{conv} | "
              f"News:{news_sentiment} | Catalyst:{catalyst_type} | "
              f"MacroOverride:{macro_override}")

    except Exception as e:
        print(f"log_reading error: {e}")

UW_TOKEN = os.getenv("UW_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
TICKER = "SPY"

# ─────────────────────────────────────────────
# GLOBAL STATE — unified from both bots
# ─────────────────────────────────────────────
state = {
    # Original gamma bot state
    "previous_gex_state": None,
    "previous_ratio": None,
    "vwap_alert_sent": False,

    # Intelligence bot state
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

    # New precision tracking from image data
    "tick_history": [],           # rolling TICK readings
    "vix_history": [],            # rolling VIX for momentum
    "inventory_bias": "NEUTRAL",  # Bull/Bear/Neutral zone
    "open_drive_detected": False, # open drive confirmation
    "session_high": None,
    "session_low": None,

    # Alert timing fixes
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
}

# OPEX dates 2026 — update quarterly
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
            "   → Hedge unwind probability: VERY HIGH\n"
            "   → Funds closing puts to clean Q books\n"
            "   → Expect put selling into close"
        )
        score_bonus += 15
    else:
        for qe in QUARTER_END_DATES:
            delta = (qe - today).days
            if 0 < delta <= 2:
                flags.append(
                    f"🗓️ QUARTER END IN {delta} DAYS\n"
                    f"   → Early hedge unwind likely starting\n"
                    f"   → Bullish bias building"
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
            flags.append(
                "⚡ OPEX DAY\n"
                "   → Maximum gamma decay\n"
                "   → Expect pinning at major strike OR explosive move\n"
                "   → Best 0DTE setups happen here"
            )
            score_bonus += 20
        elif days_to_opex <= 2:
            flags.append(
                f"⚡ OPEX IN {days_to_opex} DAYS\n"
                f"   → Gamma acceleration zone\n"
                f"   → Best window for 400-700% setups\n"
                f"   → Negative GEX amplifying moves"
            )
            score_bonus += 15
        elif days_to_opex <= 5:
            flags.append(
                f"📅 OPEX IN {days_to_opex} DAYS\n"
                f"   → Elevated gamma activity beginning\n"
                f"   → Watch for GEX build"
            )
            score_bonus += 8
        else:
            flags.append(f"📅 OPEX IN {days_to_opex} DAYS — Standard conditions")

    return flags, score_bonus, days_to_opex


# ─────────────────────────────────────────────
# MODULE 2: GEX FETCH (original gamma bot)
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
    mid_roc = recent[-1] - recent[-2]

    vol_positive = vol_gex > 0
    oi_positive = oi_gex > 0

    if vol_positive and oi_positive:
        return "BULLISH_MOMENTUM", 95
    if vol_positive and not oi_positive:
        return "HEDGE_UNWIND_CONFIRMED", 88
    if not vol_positive:
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
            "MMs long gamma — selling into strength.\n"
            "Moves dampened. Good for steady gains.\n"
            "→ Calls OK but don't expect 400%+ today."
        ),
        "HEDGE_UNWIND_CONFIRMED": (
            "🚀 HEDGE UNWIND CONFIRMED\n"
            "Vol GEX flipped positive — OI still negative.\n"
            "Institutions CLOSING put hedges.\n"
            "MMs buying shares as delta collateral = price rises.\n"
            "No new call buying needed — put SELLING is the fuel.\n"
            "→ CALLS strongly favored. 400-700% setup."
        ),
        "HEDGE_UNWIND_EARLY": (
            "🔄 EARLY HEDGE UNWIND ← CAUGHT EARLY\n"
            "Vol GEX still negative but improving rapidly.\n"
            "Institutions BEGINNING to close hedges.\n"
            "Signal BEFORE the rocket fires.\n"
            "→ Prepare call entry. Watch for Vol GEX flip."
        ),
        "TRANSITION_ZONE": (
            "⚠️ TRANSITION ZONE\n"
            "Vol GEX decelerating but not reversing yet.\n"
            "Genuine decision point — could go either way.\n"
            "→ No new entries. Wait for confirmation."
        ),
        "BEARISH_HEDGE_BUILD": (
            "🔴 BEARISH HEDGE BUILD\n"
            "Vol GEX accelerating negative.\n"
            "Institutions BUYING put protection.\n"
            "MMs selling shares to hedge = downward pressure.\n"
            "→ PUTS favored. Calls at serious risk."
        ),
        "NEUTRAL": (
            "⚪ NEUTRAL\n"
            "No clear directional bias.\n"
            "→ Stay out. Wait for regime to establish."
        ),
        "INSUFFICIENT_DATA": (
            "📊 COLLECTING DATA\n"
            "Need 3+ readings to establish regime.\n"
            "→ Check back in 30-45 minutes."
        ),
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
                        f"🚀 PUT HEDGE CLOSING: ${strike:.0f}P\n"
                        f"   Vol/OI: {round(vol_oi_ratio)}x ({int(volume/1000)}K vol)\n"
                        f"   → Institutions unwinding protection"
                    )
                elif is_put and vol_oi_ratio >= 10 and volume >= 5000:
                    unwind_score += 10
                    unwind_signals.append(
                        f"⚠️ PUT CLOSING (moderate): ${strike:.0f}P "
                        f"Vol/OI: {round(vol_oi_ratio)}x"
                    )
                if is_put and is_descending and volume >= 5000:
                    unwind_score += 15
                    unwind_signals.append(
                        f"🔽 DESCENDING FILL PUT: ${strike:.0f}P\n"
                        f"   → Aggressive institutional put selling"
                    )
                if is_call and is_sweep and volume >= 5000:
                    unwind_score += 8
                    unwind_signals.append(
                        f"📈 CALL SWEEP: ${strike:.0f}C\n"
                        f"   → Fresh bullish entry, {int(volume/1000)}K contracts"
                    )
            except Exception:
                continue

        unwind_score = min(unwind_score, 100)

        if unwind_score >= 40:
            return True, unwind_score, unwind_signals[:6], "BULLISH — Hedge unwind active"
        elif unwind_score >= 20:
            return True, unwind_score, unwind_signals[:6], "LEANING BULLISH — Early unwind signs"
        return False, unwind_score, unwind_signals[:6], "NEUTRAL — No unwind detected"

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
        v_data = requests.get(url, headers=headers, params=params, timeout=10).json().get("data", [])

        params["greek"] = "charm"
        c_data = requests.get(url, headers=headers, params=params, timeout=10).json().get("data", [])

        vanna_target = vanna_strength = charm_target = charm_strength = None

        if v_data:
            best = max(v_data, key=lambda x: float(x.get("vanna", 0) or 0))
            vanna_target = float(best.get("strike", 0))
            vanna_strength = float(best.get("vanna", 0) or 0)

        if c_data:
            worst = min(c_data, key=lambda x: float(x.get("charm", 0) or 0))
            charm_target = float(worst.get("strike", 0))
            charm_strength = float(worst.get("charm", 0) or 0)

        conflict = (
            vanna_target and charm_target and
            abs(vanna_target - charm_target) <= 1.0
        )

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
        lines.append(
            "⚠️ CONFLICT: Vanna + Charm stacked at same strike\n"
            "   Forces canceling = CONSOLIDATION TRAP risk"
        )

    if vanna_window_open:
        lines.append(f"⏰ Vanna window: ~{mins_left} min left before charm dominates")
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
                f"⚠️ Price within 0.5% of vanna ${vanna_target} "
                f"({prox:.2f}% away) — magnetic stall zone"
            )

    if state["open_iv"] and current_iv:
        iv_chg = abs(current_iv - state["open_iv"]) / state["open_iv"] * 100
        if iv_chg < 2.0:
            score += 25
            signals.append(
                f"⚠️ IV only {iv_chg:.1f}% from open — "
                f"no directional conviction. Need >2% for real move."
            )
        else:
            signals.append(f"✅ IV moving {iv_chg:.1f}% — conviction present")

    if state["open_volume"] and current_volume:
        vol_ratio = current_volume / state["open_volume"]
        if vol_ratio < 0.7:
            score += 20
            signals.append(
                f"⚠️ Volume only {round(vol_ratio*100)}% of open — "
                f"institutions not participating"
            )

    if conflict:
        score += 15
        signals.append(
            "⚠️ Vanna + charm stacked at same strike — "
            "directional forces canceling out"
        )

    if len(state["open_time_prices"]) >= 4:
        prices = state["open_time_prices"]
        changes = sum(
            1 for i in range(1, len(prices)-1)
            if (prices[i]-prices[i-1]) * (prices[i+1]-prices[i]) < 0
        )
        rng = max(prices) - min(prices)
        if changes >= 3 and rng < 1.5:
            score += 10
            signals.append(
                f"⚠️ {changes} direction changes, ${rng:.2f} range — "
                f"choppy, no follow-through"
            )

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

        # Track VIX history for momentum
        if vix_spot:
            state["vix_history"].append(vix_spot)
            if len(state["vix_history"]) > 5:
                state["vix_history"].pop(0)

        # Term structure
        if vix_spot and vix3m_val:
            if vix_spot > vix3m_val * 1.02:
                vix_term = "BACKWARDATION"
                term_sig = "⚡ BACKWARDATION — Fear spike. Explosive moves both ways."
            elif vix_spot < vix3m_val * 0.98:
                vix_term = "CONTANGO"
                term_sig = "😴 CONTANGO — Calm market. High chop risk."
            else:
                vix_term = "FLAT"
                term_sig = "⚠️ FLAT — Neutral term structure."
        else:
            vix_term = "UNKNOWN"
            term_sig = "Term structure unavailable"

        # VIX momentum (is VIX rising or falling?)
        vix_momentum = ""
        if len(state["vix_history"]) >= 3:
            vix_roc = state["vix_history"][-1] - state["vix_history"][0]
            if vix_roc > 1.5:
                vix_momentum = " ↑ RISING — fear building"
            elif vix_roc < -1.5:
                vix_momentum = " ↓ FALLING — IV crush, vanna fuel active"
            else:
                vix_momentum = " → STABLE"

        # VIX level signal
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

        # VVIX signal
        if vvix_val:
            if vvix_val >= 100:
                vvix_sig = f"🔥 EXPLOSIVE ({round(vvix_val,1)}) — Velocity day likely. Full size OK."
            elif vvix_val >= 90:
                vvix_sig = f"⚡ ACTIVE ({round(vvix_val,1)}) — Good momentum conditions"
            elif vvix_val >= 85:
                vvix_sig = f"⚠️ BORDERLINE ({round(vvix_val,1)}) — Needs confirmation"
            else:
                vvix_sig = f"😴 QUIET ({round(vvix_val,1)}) — Chop day likely. Reduce size."
        else:
            vvix_sig = "Unavailable"

        return vix_spot, vvix_val, vix_term, term_sig, vix_sig, vvix_sig

    except Exception as e:
        print(f"VIX error: {e}")
        return None, None, "UNKNOWN", "Unavailable", "Unavailable", "Unavailable"


# ─────────────────────────────────────────────
# MODULE 8: PRECISION TICK / INVENTORY READ
# Adds accuracy from image data patterns observed
# ─────────────────────────────────────────────
def fetch_tick_and_inventory():
    """
    Approximates TICK and inventory bias using SPY intraday data.
    Based on observed patterns:
    - TICK neutral (0-400) + 100% neutral inventory = MONITOR/chop
    - TICK above 600 sustained = institutional accumulation
    - TICK below -600 sustained = institutional distribution
    - Inventory MGT score correlates to directional conviction
    """
    try:
        spy = yf.download("SPY", period="1d", interval="1m", progress=False)
        if spy.empty or len(spy) < 10:
            return "UNAVAILABLE", 0, "NEUTRAL", False

        closes = spy["Close"].iloc[-10:].values.flatten()
        opens = spy["Open"].iloc[-10:].values.flatten()
        volumes = spy["Volume"].iloc[-10:].values.flatten()

        # TICK approximation
        up_bars = sum(1 for c, o in zip(closes, opens) if c > o)
        down_bars = sum(1 for c, o in zip(closes, opens) if c < o)
        tick_approx = (up_bars - down_bars) * 100

        # Track tick history
        state["tick_history"].append(tick_approx)
        if len(state["tick_history"]) > 6:
            state["tick_history"].pop(0)

        # Sustained tick check (is it consistently one direction?)
        sustained_bull = len(state["tick_history"]) >= 3 and all(
            t > 300 for t in state["tick_history"][-3:]
        )
        sustained_bear = len(state["tick_history"]) >= 3 and all(
            t < -300 for t in state["tick_history"][-3:]
        )

        # Session high/low tracking
        session_high = float(spy["High"].max())
        session_low = float(spy["Low"].min())
        state["session_high"] = session_high
        state["session_low"] = session_low

        # Volume surge detection (open drive signal)
        avg_vol = float(np.mean(volumes[:-3])) if len(volumes) > 3 else 0
        recent_vol = float(np.mean(volumes[-3:]))
        volume_surge = recent_vol > avg_vol * 1.5

        # Open drive detection (PDT time)
        pdt = now_pdt()
        mins_since_open = (pdt.hour - 6) * 60 + pdt.minute - 30
        open_drive = False
        if mins_since_open <= 30 and volume_surge and (sustained_bull or sustained_bear):
            open_drive = True
            state["open_drive_detected"] = True

        # Inventory bias from price vs VWAP position
        try:
            spy["vwap"] = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
            current_price = float(spy["Close"].iloc[-1])
            current_vwap = float(spy["vwap"].iloc[-1])

            if current_price > current_vwap * 1.002:
                inventory_bias = "BULL ZONE"
                state["inventory_bias"] = "BULL"
            elif current_price < current_vwap * 0.998:
                inventory_bias = "BEAR ZONE"
                state["inventory_bias"] = "BEAR"
            else:
                inventory_bias = "NEUTRAL (100%)"
                state["inventory_bias"] = "NEUTRAL"
        except:
            inventory_bias = "NEUTRAL"

        # Plain English TICK signal
        if tick_approx >= 600 or sustained_bull:
            tick_signal = (
                f"📈 STRONG BUYING (TICK ~+{tick_approx})\n"
                f"   {'✅ SUSTAINED' if sustained_bull else 'Single reading'} — "
                f"{'Open drive detected!' if open_drive else 'Accumulation'}"
            )
        elif tick_approx >= 200:
            tick_signal = f"🟡 MILD BUYING (TICK ~+{tick_approx}) — moderate bullish pressure"
        elif tick_approx <= -600 or sustained_bear:
            tick_signal = (
                f"📉 STRONG SELLING (TICK ~{tick_approx})\n"
                f"   {'✅ SUSTAINED' if sustained_bear else 'Single reading'} — "
                f"{'Open drive DOWN!' if open_drive else 'Distribution'}"
            )
        elif tick_approx <= -200:
            tick_signal = f"🟡 MILD SELLING (TICK ~{tick_approx}) — moderate bearish pressure"
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
# MODULE 9: PRECISION DIRECTIONAL SCORER
# Incorporates: GEX + TICK + Inventory + Vanna +
# VIX momentum + Open Drive + Calendar
# ─────────────────────────────────────────────
def score_conviction(vix_spot, vvix_val, vix_term, vol_gex, prev_vol_gex,
                      regime, unwind_score, cal_bonus, vanna_window_open,
                      conflict, ratio, tick_approx, inventory_bias,
                      open_drive):
    score = 0
    checklist = []

    # VVIX (25pts)
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

    # VIX term (15pts)
    if vix_term == "BACKWARDATION":
        score += 15
        checklist.append("✅ VIX Backwardation — fear elevated (+15)")
    elif vix_term == "FLAT":
        score += 7
        checklist.append("⚠️ VIX Flat (+7)")
    else:
        checklist.append("❌ VIX Contango — calm (+0)")

    # VIX momentum bonus (5pts)
    if len(state["vix_history"]) >= 3:
        vix_roc = state["vix_history"][-1] - state["vix_history"][0]
        if abs(vix_roc) > 1.5:
            score += 5
            direction = "falling ✅ (vanna fuel)" if vix_roc < 0 else "rising ⚠️ (headwind)"
            checklist.append(f"✅ VIX momentum {direction} (+5)")

    # Regime (20pts)
    regime_pts = {
        "HEDGE_UNWIND_CONFIRMED": 20,
        "BULLISH_MOMENTUM": 18,
        "BEARISH_HEDGE_BUILD": 16,
        "HEDGE_UNWIND_EARLY": 14,
        "TRANSITION_ZONE": 8,
        "NEUTRAL": 0,
        "INSUFFICIENT_DATA": 0,
    }
    rpts = regime_pts.get(regime, 0)
    score += rpts
    emoji = "✅" if rpts >= 14 else "⚠️" if rpts >= 8 else "❌"
    checklist.append(f"{emoji} Regime: {regime} (+{rpts})")

    # Hedge unwind (15pts)
    if unwind_score >= 40:
        score += 15
        checklist.append(f"✅ Hedge unwind confirmed ({unwind_score}/100) (+15)")
    elif unwind_score >= 20:
        score += 8
        checklist.append(f"⚠️ Early unwind signs ({unwind_score}/100) (+8)")
    else:
        checklist.append("❌ No unwind detected (+0)")

    # Calendar (max 15pts)
    cal_pts = min(cal_bonus, 15)
    score += cal_pts
    if cal_pts >= 10:
        checklist.append(f"✅ Calendar catalyst (+{cal_pts})")
    elif cal_pts >= 5:
        checklist.append(f"⚠️ Calendar bonus (+{cal_pts})")

    # Vanna window (10pts)
    if vanna_window_open:
        score += 10
        checklist.append("✅ Vanna window open (+10)")
    else:
        checklist.append("❌ Vanna window closed (+0)")

    # TICK precision (10pts)
    if abs(tick_approx) >= 600:
        score += 10
        checklist.append(f"✅ TICK strong {'buying' if tick_approx > 0 else 'selling'} (+10)")
    elif abs(tick_approx) >= 300:
        score += 5
        checklist.append(f"⚠️ TICK moderate (+5)")
    else:
        checklist.append(f"❌ TICK neutral ({tick_approx}) — chop risk (+0)")

    # Inventory bias (5pts)
    if inventory_bias in ["BULL ZONE", "BEAR ZONE"]:
        score += 5
        checklist.append(f"✅ Inventory: {inventory_bias} — directional (+5)")
    else:
        checklist.append(f"❌ Inventory: NEUTRAL — no bias (+0)")

    # Open drive bonus (10pts)
    if open_drive:
        score += 10
        checklist.append("🚀 OPEN DRIVE DETECTED — institutional conviction (+10)")

    # Consolidation penalty (-20pts)
    if conflict:
        score -= 20
        checklist.append("🚨 Vanna/charm conflict — consolidation risk (-20)")

    score = max(0, min(100, score))

    if score >= 80:
        grade = "A+ 🔥"
        rec = "FULL SIZE. 400-700% day setup confirmed."
    elif score >= 65:
        grade = "B+ ✅"
        rec = "NORMAL SIZE. 200-400% realistic with right entry."
    elif score >= 50:
        grade = "C ⚠️"
        rec = "HALF SIZE ONLY. Wait for open confirmation."
    elif score >= 35:
        grade = "D 🔴"
        rec = "MINIMAL or sit out. High chop risk."
    else:
        grade = "F ❌"
        rec = "DO NOT TRADE. Theta will destroy premium."

    return score, grade, rec, checklist


# ─────────────────────────────────────────────
# HELPERS (original gamma bot)
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


async def _send(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)


def alert(text):
    asyncio.run(_send(text))


def get_vwap():
    try:
        spy = yf.download("SPY", period="1d", interval="5m", progress=False)
        if spy.empty:
            return None, None, None, None, None
        spy["vwap"] = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
        cp = float(spy["Close"].iloc[-1])
        cv = float(spy["vwap"].iloc[-1])
        pp = float(spy["Close"].iloc[-2]) if len(spy) > 1 else cp
        pv = float(spy["vwap"].iloc[-2]) if len(spy) > 1 else cv
        vol = float(spy["Volume"].iloc[-1])
        return cp, cv, pp, pv, vol
    except:
        return None, None, None, None, None


# ─────────────────────────────────────────────
# VWAP CROSS (original gamma bot — preserved)
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
                f"Regime: {state['regime']} | Time: {now_str} PDT\n"
                f"Inventory: {state['inventory_bias']}\n\n"
                f"⚠️ Confirm candle close below VWAP before entering."
            )
            state["vwap_alert_sent"] = True

    elif bullish and not state["vwap_alert_sent"]:
        if pp <= pv and cp > cv:
            alert(
                f"🔼 VWAP CROSS — BULLISH ENTRY\n"
                f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
                f"Regime: {state['regime']} | Time: {now_str} PDT\n"
                f"Inventory: {state['inventory_bias']}\n\n"
                f"⚠️ Confirm candle close above VWAP before entering."
            )
            state["vwap_alert_sent"] = True

    if (bearish and cp > cv) or (bullish and cp < cv):
        state["vwap_alert_sent"] = False


# ─────────────────────────────────────────────
# NEW MODULE: HEARTBEAT WATCHDOG
# Fires every 60 min during market hours
# regardless of all other alert flags
# So you always know bot is alive
# ─────────────────────────────────────────────
def check_heartbeat():
    """
    Sends a simple alive ping every 60 minutes
    during market hours. Prevents silent failures.
    """
    if not is_market_open():
        return
    try:
        now_epoch = time.time()
        last_beat = state.get("last_heartbeat", 0)
        if now_epoch - last_beat < 3600:  # 60 min
            return

        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, _, _, _ = r
        now_str = now_pdt().strftime("%H:%M")
        regime = state.get("regime", "UNKNOWN")
        gex_state = state.get("previous_gex_state", "UNKNOWN")
        conv = state.get("last_conviction_score", 0)
        vwap_dist = round(cp - cv, 2)
        vwap_side = "above" if vwap_dist > 0 else "below"

        alert(
            f"💓 BOT ALIVE — SPY\n"
            f"{'─'*30}\n"
            f"{now_str} PDT | ${round(cp,2)}\n\n"
            f"Regime: {regime}\n"
            f"GEX: {gex_state}\n"
            f"Conviction: {conv}/100\n"
            f"VWAP: ${round(cv,2)} "
            f"(${abs(vwap_dist):.2f} {vwap_side})\n\n"
            f"Bot running normally ✅"
        )
        state["last_heartbeat"] = now_epoch
        print(f"💓 Heartbeat sent at {now_str} PDT")

    except Exception as e:
        print(f"Heartbeat error: {e}")


# ─────────────────────────────────────────────
# NEW MODULE: DOJI TRANSITION DETECTOR
# Catches the exact setup you described:
# Price near VWAP + Vol GEX decelerating
# + small candle body = transition forming
# ─────────────────────────────────────────────
def check_doji_transition():
    """
    Detects when a bearish/bullish trend
    is transitioning at VWAP.

    Three conditions must align:
    1. Price within $0.75 of VWAP (doji zone)
    2. Vol GEX rate of change decelerating
       (negative but getting less negative
        OR positive but getting less positive)
    3. Vol GEX and OI GEX behavior consistent
       with regime flip starting

    This is the "warning before the warning"
    — fires BEFORE the regime transition
    so you can prepare your entry
    """
    if not is_market_open():
        return
    try:
        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, pp, pv, _ = r

        # Condition 1: Price within $0.75 of VWAP
        vwap_dist = abs(cp - cv)
        if vwap_dist > 0.75:
            state["doji_transition_sent"] = False
            return

        # Condition 2: Vol GEX decelerating
        history = state.get("vol_gex_history", [])
        if len(history) < 3:
            return

        recent = history[-3:]
        vol_gex_current = recent[-1]
        vol_gex_prev = recent[-2]
        vol_gex_older = recent[-3]

        # Rate of change slowing down
        roc_recent = abs(vol_gex_current) - abs(vol_gex_prev)
        roc_older = abs(vol_gex_prev) - abs(vol_gex_older)
        decelerating = roc_older < 0 and roc_recent > roc_older

        if not decelerating:
            return

        # Condition 3: Don't fire if already in transition/unwind
        regime = state.get("regime", "")
        if regime in ["HEDGE_UNWIND_CONFIRMED", "BULLISH_MOMENTUM"]:
            return

        # Already sent for this doji formation
        if state.get("doji_transition_sent"):
            return

        now_str = now_pdt().strftime("%H:%M")
        prev_regime = state.get("previous_regime", "")

        # Determine likely transition direction
        if vol_gex_current < 0 and roc_recent > 0:
            # Negative but improving = bullish transition
            direction = "BEARISH → BULLISH"
            action = "Watch for Vol GEX flip positive\n→ Calls on VWAP break above"
            emoji = "🔄"
        elif vol_gex_current > 0 and roc_recent < 0:
            # Positive but deteriorating = bearish transition
            direction = "BULLISH → BEARISH"
            action = "Watch for Vol GEX flip negative\n→ Puts on VWAP break below"
            emoji = "🔄"
        else:
            direction = "CONSOLIDATING"
            action = "No clear direction yet\n→ Wait for Vol GEX commitment"
            emoji = "⚠️"

        vol_fmt_current = fmt(vol_gex_current / (650 * 6.31) / 1e6)
        vol_fmt_prev = fmt(vol_gex_prev / (650 * 6.31) / 1e6)

        alert(
            f"{emoji} DOJI TRANSITION FORMING — SPY\n"
            f"{'─'*35}\n"
            f"Time: {now_str} PDT | Price: ${round(cp,2)}\n"
            f"VWAP: ${round(cv,2)} "
            f"(only ${vwap_dist:.2f} away)\n\n"
            f"🔍 WHAT IS HAPPENING\n"
            f"Transition: {direction}\n"
            f"Vol GEX decelerating:\n"
            f"  Previous: {vol_fmt_prev}\n"
            f"  Current:  {vol_fmt_current}\n"
            f"  → Momentum losing steam\n\n"
            f"📊 CANDLE SIGNAL\n"
            f"Doji forming at VWAP\n"
            f"Price indecision = institutions\n"
            f"pausing before next move\n\n"
            f"🎯 ACTION\n"
            f"{action}\n\n"
            f"⚠️ NOT A TRADE SIGNAL YET\n"
            f"Wait for Vol GEX to confirm direction\n"
            f"Next scheduled reading will clarify"
        )
        state["doji_transition_sent"] = True
        print(f"Doji transition alert: {direction} at {now_str} PDT")

    except Exception as e:
        print(f"Doji transition error: {e}")


# ─────────────────────────────────────────────
# NEW MODULE: GAMMA WALL APPROACH / TP ALERT
# Fires when price approaches a major gamma
# wall — tells you TP zone is near
# Solves the "missed 655-657 exit" problem
# ─────────────────────────────────────────────
def check_gamma_wall_approach():
    """
    Detects when price is within $1.50 of
    a significant gamma wall and alerts:
    - If approaching from below (bounce):
      → Take profits on calls, watch for rejection
    - If approaching from above (dump):
      → Take profits on puts, watch for bounce

    Uses vanna targets and OI concentration
    as wall estimates.
    """
    if not is_market_open():
        return
    try:
        r = get_vwap()
        if r[0] is None:
            return
        cp, cv, _, _, _ = r

        vanna_target = state.get("current_vanna_target")
        charm_target = state.get("current_charm_target")
        regime = state.get("previous_regime", "")
        gex_state = state.get("previous_gex_state", "")
        now_str = now_pdt().strftime("%H:%M")

        # Don't spam — only fire once per wall per session
        last_wall_alert = state.get("last_wall_alert_price", 0)
        if abs(cp - last_wall_alert) < 2.0:
            return

        # Check vanna target approach
        if vanna_target:
            dist_to_vanna = vanna_target - cp
            abs_dist = abs(dist_to_vanna)

            if abs_dist <= 1.5:
                approaching_from = "below" if dist_to_vanna > 0 else "above"

                if approaching_from == "below":
                    # Price running UP toward vanna target
                    # Tell user to take profits on calls
                    alert(
                        f"🎯 VANNA TARGET APPROACHING — SPY\n"
                        f"{'─'*35}\n"
                        f"Time: {now_str} PDT | Price: ${round(cp,2)}\n"
                        f"Vanna target: ${vanna_target} "
                        f"(${abs_dist:.2f} away)\n\n"
                        f"💰 PROFIT ZONE — CALLS\n"
                        f"If holding calls from lower:\n"
                        f"→ THIS IS YOUR EXIT ZONE\n"
                        f"→ Sell calls between "
                        f"${round(vanna_target-0.5,0)}-${vanna_target}\n"
                        f"→ Do NOT hold past the target\n\n"
                        f"⚠️ WHAT HAPPENS AT THE WALL\n"
                        f"Charm force reverses at ${vanna_target}\n"
                        f"MMs start selling delta above here\n"
                        f"Rally stalls or reverses\n\n"
                        f"🔄 AFTER TARGET HIT\n"
                        f"Watch Vol GEX — if flips negative\n"
                        f"→ Put re-entry at ${vanna_target}"
                    )
                    state["last_wall_alert_price"] = cp
                    print(f"Vanna target approach alert (from below) at {now_str}")

                elif approaching_from == "above":
                    # Price falling DOWN toward vanna target
                    # Tell user to take profits on puts
                    alert(
                        f"🎯 SUPPORT ZONE APPROACHING — SPY\n"
                        f"{'─'*35}\n"
                        f"Time: {now_str} PDT | Price: ${round(cp,2)}\n"
                        f"Vanna support: ${vanna_target} "
                        f"(${abs_dist:.2f} away)\n\n"
                        f"💰 PROFIT ZONE — PUTS\n"
                        f"If holding puts from higher:\n"
                        f"→ Consider partial profit here\n"
                        f"→ Vanna support at ${vanna_target} "
                        f"may cause bounce\n"
                        f"→ Sell half, keep half for break\n\n"
                        f"⚠️ WHAT HAPPENS AT THE WALL\n"
                        f"Charm +{round(abs(charm_target or 0)/1e6,0)}M "
                        f"support at this level\n"
                        f"MMs forced to buy delta here\n"
                        f"Puts may stall or reverse\n\n"
                        f"🔄 IF WALL BREAKS\n"
                        f"Hold remaining puts\n"
                        f"Next target: ${round(vanna_target - 5, 0)}"
                    )
                    state["last_wall_alert_price"] = cp
                    print(f"Support approach alert (from above) at {now_str}")

    except Exception as e:
        print(f"Gamma wall approach error: {e}")


# ─────────────────────────────────────────────
# CONSOLIDATION JOB
# ─────────────────────────────────────────────
def check_consolidation_job():
    if not is_market_open():
        return
    pdt = now_pdt()
    now = pdt
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
        now_str = now.strftime("%H:%M")

        if is_cons:
            sigs = "\n".join(signals)
            alert(
                f"🚨 CONSOLIDATION TRAP WARNING — SPY\n"
                f"{'─'*35}\n"
                f"Time: {now_str} | Price: ${cp:.2f}\n"
                f"Score: {score}/100\n\n"
                f"⚠️ SIGNALS\n{sigs}\n\n"
                f"💡 WHAT IS HAPPENING\n"
                f"Vanna pulling UP toward ${vt}\n"
                f"Charm pushing DOWN toward ${ct}\n"
                f"Forces CANCELING = premium decays fast.\n\n"
                f"🚫 DO NOT TRADE THIS OPEN\n"
                f"Wait for:\n"
                f"→ Price moves >1% from ${vt} with volume\n"
                f"→ IV expands >3% from open\n"
                f"→ TICK holds +600 or -600 for 15+ min\n"
                f"→ Hedge unwind score >40\n\n"
                f"Re-assess after 10:00am PDT."
            )
            state["consolidation_alert_sent"] = True
            state["consolidation_gex_state"] = gex_state

    except Exception as e:
        print(f"Consolidation job error: {e}")


# ─────────────────────────────────────────────
# END OF DAY AUTO-FILL
# Runs at 1pm PDT — automatically fills outcome
# columns for today's rows in the CSV log
# ─────────────────────────────────────────────
def eod_autofill(close_price):
    """
    At market close (1pm PDT), automatically fills:
    - outcome_direction: UP / DOWN / CHOP
    - outcome_points: how many points SPY moved
    - signal_correct: whether the morning GEX signal was right

    Only fills rows from today that are still blank.
    You still manually fill: notes column for context.
    """
    try:
        today_str = date.today().strftime("%Y-%m-%d")

        if not os.path.exists(LOG_FILE):
            return

        rows = []
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        # Find today's opening price from first log entry of today
        today_rows = [r for r in rows if r["date"] == today_str]
        if not today_rows:
            return

        open_price = float(today_rows[0]["price"])
        point_move = round(close_price - open_price, 2)

        if abs(point_move) < 1.0:
            direction = "CHOP"
        elif point_move > 0:
            direction = "UP"
        else:
            direction = "DOWN"

        # Get morning signal — first directional reading of the day
        morning_signal = None
        for r in today_rows:
            if "DIRECTIONAL" in r.get("gex_state", ""):
                morning_signal = r["gex_state"]
                break
            elif r.get("gex_state") in ["NEUTRAL", "WATCH", "COUNTER"]:
                morning_signal = r["gex_state"]
                break

        # Determine if signal was correct
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
        else:
            correct = ""

        # Pull session stats BEFORE the rows loop
        session_high = state.get("session_high") or close_price
        session_low = state.get("session_low") or close_price

        # Update today's rows that have blank outcome columns
        updated = 0
        for r in rows:
            if r["date"] == today_str and r["outcome_direction"] == "":
                r["outcome_direction"] = direction
                r["outcome_points"] = point_move
                r["signal_correct"] = correct
                r["max_move_up"] = round(session_high - open_price, 2) if session_high else ""
                r["max_move_down"] = round(open_price - session_low, 2) if session_low else ""
                updated += 1

        # Write back to CSV
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

        # Auto-commit to GitHub — data survives redeployments
        git_commit_log()

        # Send end of day summary to Telegram
        now_str = now_pdt().strftime("%H:%M")
        signal_emoji = "✅" if correct == "YES" else "❌" if correct == "NO" else "⚠️"
        max_up = round(session_high - open_price, 2)
        max_down = round(open_price - session_low, 2)

        alert(
            f"📊 END OF DAY SUMMARY — SPY\n"
            f"{'─'*35}\n"
            f"Date: {today_str} | {now_str}\n\n"
            f"Open:  ${round(open_price, 2)}\n"
            f"Close: ${round(close_price, 2)}\n"
            f"Move:  {'+' if point_move > 0 else ''}{point_move} pts\n"
            f"Direction: {direction}\n"
            f"Max Up: +{max_up} | Max Down: -{max_down}\n\n"
            f"Signal: {morning_signal or 'None'}\n"
            f"Correct: {signal_emoji} {correct}\n\n"
            f"📝 {updated} rows logged automatically\n"
            f"💾 CSV saved to GitHub\n\n"
            f"💬 Add context? Reply:\n"
            f"/notes Iran news drove the gap"
        )

        print(f"EOD complete — {updated} rows | {direction} {point_move}pts | {correct}")

    except Exception as e:
        print(f"EOD autofill error: {e}")


# ─────────────────────────────────────────────
# MAIN JOB — unified
# ─────────────────────────────────────────────
def run_job():
    pdt = now_pdt()
    now_str = pdt.strftime("%H:%M")
    print(f"\n{'='*60}\nJob: {now_str} PDT\n{'='*60}")

    try:
        # ── Core GEX (original gamma bot) ──
        oi_gex, vol_gex, price = fetch_gex()
        if oi_gex is None:
            print("No GEX data")
            return
        if vol_gex == 0:
            print(f"Pre-market. OI: {round(oi_gex/1e9,2)}B | Price: ${price}")
            return

        # Update vol GEX history
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

        # ── All intelligence modules ──
        vix_spot, vvix_val, vix_term, term_sig, vix_sig, vvix_sig = fetch_vix_data()
        vt, vs, ct, cs, conflict = fetch_vanna_charm()
        vc_text, vanna_window = get_vanna_charm_read(vt, vs, ct, cs, price, conflict)

        # Store for gamma wall checker
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

        # Console log
        print(f"${price} | {gex_state} | Regime:{regime} | Score:{conv}")
        print(f"OI:{oi_b}B ({oi_fmt}) | VOL:{vol_b}B ({vol_fmt}) | Ratio:{ratio_r}x")
        print(f"VIX:{vix_sig} | VVIX:{vvix_sig}")
        print(f"Flow:{flow_dir} | Grade:{grade}")

        # ── LOG EVERY READING TO CSV ──
        log_reading(
            price=price,
            oi_gex=oi_gex,
            vol_gex=vol_gex,
            oi_m=oi_m,
            vol_m=vol_m,
            ratio=ratio,
            gex_state=gex_state,
            regime=regime,
            conv=conv,
            grade=grade,
            vix_spot=vix_spot,
            vvix_val=vvix_val,
            vix_term=vix_term,
            tick_approx=tick_approx,
            inventory_bias=inventory_bias,
            unwind_score=unwind_score,
            open_drive=open_drive,
            vanna_target=vt,
            charm_target=ct,
            cal_flags=cal_flags,
            days_opex=days_opex
        )

        # ── ORIGINAL GAMMA BOT: GEX STATE CHANGE ──
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
                f"OI Net GEX: {oi_fmt} ({oi_b}B raw)\n"
                f"VOL Net GEX: {vol_fmt} ({vol_b}B raw)\n"
                f"Ratio: {ratio_r}x\n"
                f"Spot: ${price}\n"
                f"Time: {now_str}\n"
                f"Previous: {state['previous_gex_state'] or 'None'}\n\n"
                f"📊 REGIME: {regime} ({reg_conf}%)\n"
                f"🎯 CONVICTION: {conv}/100 — {grade}\n"
                f"→ {rec}"
            )
            print(f"GEX state alert: {state['previous_gex_state']} → {gex_state}")

        # ── REGIME CHANGE ALERT ──
        if regime != state["previous_regime"] and regime not in ["INSUFFICIENT_DATA"]:
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
            uw_text = "\n".join(unwind_sigs) if unwind_sigs else "None"
            cal_text = "\n".join(cal_flags) if cal_flags else "No special events"

            alert(
                f"{r_emoji} REGIME TRANSITION — SPY\n"
                f"{'─'*35}\n"
                f"NEW: {regime} | WAS: {prev}\n"
                f"Confidence: {reg_conf}%\n"
                f"Time: {now_str} | Price: ${price}\n\n"
                f"📊 REGIME MEANING\n{reg_signal}\n\n"
                f"📈 VANNA / CHARM\n{vc_text}\n\n"
                f"🔍 OPTIONS FLOW\n"
                f"Direction: {flow_dir}\n"
                f"Unwind: {unwind_score}/100\n"
                f"{uw_text}\n\n"
                f"📊 PRECISION\n"
                f"{tick_signal}\n"
                f"Inventory: {inventory_bias}\n"
                f"{'🚀 OPEN DRIVE ACTIVE' if open_drive else ''}\n\n"
                f"🗓️ CALENDAR\n{cal_text}\n\n"
                f"🎯 CONVICTION: {conv}/100 — {grade}\n"
                f"→ {rec}\n\n"
                f"📋 CHECKLIST\n{cl_text}"
            )
            print(f"Regime alert: {prev} → {regime}")

        # ── HEDGE UNWIND STANDALONE ──
        elif (unwind_det and not state["hedge_unwind_alert_sent"]
              and unwind_score >= 40
              and regime == state["previous_regime"]):
            uw_text = "\n".join(unwind_sigs)
            alert(
                f"🚀 HEDGE UNWIND DETECTED — SPY\n"
                f"{'─'*35}\n"
                f"Score: {unwind_score}/100 | Flow: {flow_dir}\n"
                f"Time: {now_str} | Price: ${price}\n\n"
                f"📊 FLOW SIGNALS\n{uw_text}\n\n"
                f"💡 MECHANICAL EXPLANATION\n"
                f"Institutions selling puts = MMs buy shares back.\n"
                f"Price rises with ZERO new call buying needed.\n\n"
                f"📈 VANNA TARGET: ${vt}\n{vc_text}\n\n"
                f"🎯 CONVICTION: {conv}/100 — {grade}\n→ {rec}"
            )
            state["hedge_unwind_alert_sent"] = True
            state["last_unwind_alert_time"] = time.time()

        if unwind_score < 20:
            state["hedge_unwind_alert_sent"] = False

        # ── ORIGINAL GAMMA BOT: CONVICTION INCREASE ──
        if ("DIRECTIONAL" in gex_state and state["previous_ratio"]
                and ratio - state["previous_ratio"] > 0.3):
            direction = "BEARISH" if oi_gex < 0 else "BULLISH"
            alert(
                f"📈 CONVICTION INCREASING — SPY\n\n"
                f"Direction: {direction}\n"
                f"Ratio: {ratio_r}x (was {round(state['previous_ratio'],2)}x)\n"
                f"OI Net GEX: {oi_fmt}\n"
                f"VOL Net GEX: {vol_fmt}\n"
                f"Spot: ${price}\n"
                f"Time: {now_str}\n\n"
                f"VIX: {vix_sig}\n"
                f"VVIX: {vvix_sig}\n"
                f"TICK: {tick_signal}\n\n"
                f"🎯 Score: {conv}/100 — {grade}\n→ {rec}"
            )

        # ── MORNING VELOCITY REPORT ──
        pdt_now = now_pdt()
        h, m = pdt_now.hour, pdt_now.minute

        # FIX 1: Morning report fires once at open
        # BUT re-fires if conviction score changes significantly
        # so you never miss a major setup shift after 6:30am
        if (h == 6 and m >= 25) or (h == 7 and m <= 15):
            conviction_changed = (
                state["last_conviction_score"] is not None and
                abs(conv - state["last_conviction_score"]) >= 15
            )
            if not state["velocity_score_sent"] or conviction_changed:
                cl_text = "\n".join(checklist)
                cal_text = "\n".join(cal_flags) if cal_flags else "No special events"
                uw_text = "\n".join(unwind_sigs) if unwind_sigs else "None yet"
                update_tag = "🔄 UPDATED — " if conviction_changed else ""

                alert(
                    f"🌅 {update_tag}PRE-MARKET REPORT — SPY\n"
                    f"{'─'*35}\n"
                    f"{now_str} | ${price}\n\n"
                    f"🎯 CONVICTION: {conv}/100\n"
                    f"Grade: {grade}\n"
                    f"→ {rec}\n\n"
                    f"📋 CHECKLIST\n{cl_text}\n\n"
                    f"🔄 REGIME: {regime}\n{reg_signal}\n\n"
                    f"📈 VANNA / CHARM\n{vc_text}\n\n"
                    f"📊 PRECISION READ\n"
                    f"{tick_signal}\n"
                    f"Inventory: {inventory_bias}\n\n"
                    f"🗓️ CALENDAR\n{cal_text}\n\n"
                    f"🔍 EARLY FLOW\n{uw_text}\n\n"
                    f"VIX: {vix_sig}\n"
                    f"VVIX: {vvix_sig}\n"
                    f"Term: {term_sig}\n\n"
                    f"⚡ GUIDE\n"
                    f"80-100 → Full size. 400-700% day.\n"
                    f"65-79 → Normal size. 200-400% realistic.\n"
                    f"50-64 → Half size. Wait for open.\n"
                    f"Below 50 → Sit out. Theta wins."
                )
                state["velocity_score_sent"] = True
                print(f"Morning report sent. Score: {conv}/100")

        # FIX 2: Consolidation alert resets if market structure
        # changes — so it can re-fire if setup resolves and returns
        if state["consolidation_alert_sent"]:
            if gex_state != state.get("consolidation_gex_state"):
                state["consolidation_alert_sent"] = False
                print("Consolidation alert reset — GEX state changed")

        # FIX 3: Hedge unwind alert allows re-firing every 45min
        # if unwind score keeps growing — catches escalating flow
        now_epoch = time.time()
        last_unwind_time = state.get("last_unwind_alert_time", 0)
        if (state["hedge_unwind_alert_sent"] and
                unwind_score >= 60 and
                now_epoch - last_unwind_time > 2700):  # 45 min
            state["hedge_unwind_alert_sent"] = False
            print("Hedge unwind alert reset — score elevated, allowing re-fire")

        # FIX 4: Mid-day GEX summary every 90 min during market hours
        # Fires regardless of other flags — always keep you informed
        last_summary_time = state.get("last_summary_time", 0)
        if (6 <= h <= 12 and
                now_epoch - last_summary_time > 5400):  # 90 min
            alert(
                f"📊 MID-SESSION UPDATE — SPY\n"
                f"{'─'*35}\n"
                f"{now_str} | ${price}\n\n"
                f"GEX State: {gex_state}\n"
                f"Regime: {regime}\n"
                f"Ratio: {ratio_r}x\n"
                f"OI: {oi_fmt} | VOL: {vol_fmt}\n\n"
                f"VIX: {vix_sig} | VVIX: {vvix_sig}\n"
                f"Flow: {flow_dir}\n"
                f"Unwind Score: {unwind_score}/100\n\n"
                f"🎯 Conviction: {conv}/100 — {grade}\n"
                f"→ {rec}"
            )
            state["last_summary_time"] = now_epoch
            print(f"Mid-session update sent at {now_str}")

        # Reset daily flags + end of day auto-fill
        if h >= 13:
            eod_autofill(price)
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
        state["last_conviction_score"] = conv

    except Exception as e:
        print(f"run_job error: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────
# TELEGRAM /notes COMMAND
# Only manual input needed — add context
# to today's log rows via Telegram message
# Usage: /notes Iran news drove the gap
# ─────────────────────────────────────────────
async def handle_telegram_updates():
    """
    Polls Telegram for /notes commands.
    Writes the note to today's CSV rows.
    Run every 5 minutes during market hours.
    """
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        updates = await bot.get_updates(timeout=5)
        today_str = date.today().strftime("%Y-%m-%d")

        for update in updates:
            if not update.message:
                continue
            text = update.message.text or ""
            if not text.startswith("/notes "):
                continue

            note = text[7:].strip()
            if not note:
                continue

            # Write note to today's rows
            if not os.path.exists(LOG_FILE):
                continue

            rows = []
            with open(LOG_FILE, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

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
            print(f"Note saved: {note}")

    except Exception as e:
        print(f"Telegram command error: {e}")


def check_telegram_commands():
    asyncio.run(handle_telegram_updates())


# ─────────────────────────────────────────────
# SCHEDULE (original gamma bot times preserved)
# ─────────────────────────────────────────────
schedule.every().day.at("13:00").do(run_job)  # 6:00am PDT
schedule.every().day.at("13:30").do(run_job)  # 6:30am PDT
schedule.every().day.at("14:00").do(run_job)  # 7:00am PDT
schedule.every().day.at("14:45").do(run_job)  # 7:45am PDT
schedule.every().day.at("15:30").do(run_job)  # 8:30am PDT
schedule.every().day.at("16:15").do(run_job)  # 9:15am PDT
schedule.every().day.at("17:00").do(run_job)  # 10:00am PDT
schedule.every().day.at("17:45").do(run_job)  # 10:45am PDT
schedule.every().day.at("18:30").do(run_job)  # 11:30am PDT
schedule.every().day.at("19:15").do(run_job)  # 12:15pm PDT
schedule.every().day.at("20:00").do(run_job)  # 1:00pm PDT (EOD)

schedule.every(5).minutes.do(check_vwap)
schedule.every(5).minutes.do(check_consolidation_job)
schedule.every(5).minutes.do(check_telegram_commands)
schedule.every(5).minutes.do(check_doji_transition)
schedule.every(5).minutes.do(check_gamma_wall_approach)
schedule.every(60).minutes.do(check_heartbeat)

print("SPY UNIFIED BOT v4.1 — ML READY")
print("=" * 60)
print("Modules:")
print("  1. GEX Core")
print("  2. VWAP Cross Alerts")
print("  3. Conviction Spike")
print("  4. Regime Detection Engine")
print("  5. Hedge Unwind Detector")
print("  6. Vanna / Charm Engine")
print("  7. Consolidation Trap Detector")
print("  8. VIX / VVIX + Momentum")
print("  9. TICK + Inventory Precision")
print(" 10. Calendar / Quarter System")
print(" 11. Unified Conviction Scorer")
print(" 12. ML Data Logger — FULLY AUTOMATED")
print(" 13. Heartbeat Watchdog (60min)")
print(" 14. Doji Transition Detector")
print(" 15. Gamma Wall / TP Zone Alert")
print("=" * 60)
print("Timezone: All times PDT (UTC-7)")
print("Schedule: 6:00am - 1:00pm PDT")
print("=" * 60)
init_log()
run_job()

while True:
    schedule.run_pending()
    time.sleep(30)
