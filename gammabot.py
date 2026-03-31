import os
import csv
import requests
import schedule
import time
import asyncio
import yfinance as yf
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

# ─────────────────────────────────────────────
# LOGGING SYSTEM
# Records every reading to CSV for future
# pattern analysis and ML training data
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
    "outcome_direction",   # filled manually: UP / DOWN / CHOP
    "outcome_points",      # filled manually: e.g. +8.5 or -3.2
    "signal_correct",      # filled manually: YES / NO / PARTIAL
    "notes"                # filled manually: any context
]

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
    Appends one row per bot reading to the CSV log.
    Outcome columns are left blank — you fill those in manually
    at end of day using the journal process.
    """
    now = datetime.now()
    cal_summary = "QUARTER_END" if "QUARTER END TODAY" in str(cal_flags) else (
                  "OPEX_DAY" if "OPEX DAY" in str(cal_flags) else
                  f"OPEX_IN_{days_opex}D" if days_opex else "NORMAL")

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
        "grade": grade.replace("🔥","").replace("✅","").replace("⚠️","").replace("🔴","").replace("❌","").strip(),
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
        # Outcome columns — fill manually at end of day
        "outcome_direction": "",
        "outcome_points": "",
        "signal_correct": "",
        "notes": ""
    }

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        writer.writerow(row)

    print(f"📝 Logged reading at {row['time']} | {gex_state} | Score:{conv} | Ratio:{ratio:.2f}x")

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
    now = datetime.now()
    minutes_since_open = (now.hour - 6) * 60 + now.minute - 30
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
    now = datetime.now()
    mins = (now.hour - 6) * 60 + now.minute - 30
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

        # Open drive detection
        # Pattern: large gap up/down + sustained TICK + volume surge in first 30min
        now = datetime.now()
        mins_since_open = (now.hour - 6) * 60 + now.minute - 30
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


def is_market_open():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=6, minute=30, second=0)
    c = now.replace(hour=13, minute=0, second=0)
    return o <= now <= c


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
    now = datetime.now().strftime("%H:%M")
    bearish = "BEARISH" in gex_s
    bullish = "BULLISH" in gex_s

    if bearish and not state["vwap_alert_sent"]:
        if pp >= pv and cp < cv:
            alert(
                f"🔽 VWAP CROSS — BEARISH ENTRY\n"
                f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
                f"Regime: {state['regime']} | Time: {now}\n"
                f"Inventory: {state['inventory_bias']}\n\n"
                f"⚠️ Confirm candle close below VWAP before entering."
            )
            state["vwap_alert_sent"] = True

    elif bullish and not state["vwap_alert_sent"]:
        if pp <= pv and cp > cv:
            alert(
                f"🔼 VWAP CROSS — BULLISH ENTRY\n"
                f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
                f"Regime: {state['regime']} | Time: {now}\n"
                f"Inventory: {state['inventory_bias']}\n\n"
                f"⚠️ Confirm candle close above VWAP before entering."
            )
            state["vwap_alert_sent"] = True

    if (bearish and cp > cv) or (bullish and cp < cv):
        state["vwap_alert_sent"] = False


# ─────────────────────────────────────────────
# CONSOLIDATION JOB
# ─────────────────────────────────────────────
def check_consolidation_job():
    if not is_market_open():
        return
    now = datetime.now()
    mins = (now.hour - 6) * 60 + now.minute - 30
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

        # Update today's rows that have blank outcome columns
        updated = 0
        for r in rows:
            if r["date"] == today_str and r["outcome_direction"] == "":
                r["outcome_direction"] = direction
                r["outcome_points"] = point_move
                r["signal_correct"] = correct
                updated += 1

        # Write back to CSV
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

        # Send end of day summary to Telegram
        now_str = datetime.now().strftime("%H:%M")
        signal_emoji = "✅" if correct == "YES" else "❌" if correct == "NO" else "⚠️"

        alert(
            f"📊 END OF DAY SUMMARY — SPY\n"
            f"{'─'*35}\n"
            f"Date: {today_str} | Time: {now_str}\n\n"
            f"Open: ${round(open_price, 2)}\n"
            f"Close: ${round(close_price, 2)}\n"
            f"Move: {'+' if point_move > 0 else ''}{point_move} pts\n"
            f"Direction: {direction}\n\n"
            f"Morning Signal: {morning_signal or 'None'}\n"
            f"Signal Correct: {signal_emoji} {correct}\n\n"
            f"📝 {updated} log rows updated automatically\n"
            f"→ Open spy_gex_log.csv to add notes column\n"
            f"   (e.g. Iran news, Fed day, OPEX expiry)"
        )

        print(f"EOD autofill complete — {updated} rows updated | {direction} {point_move}pts | Correct: {correct}")

    except Exception as e:
        print(f"EOD autofill error: {e}")


# ─────────────────────────────────────────────
# MAIN JOB — unified
# ─────────────────────────────────────────────
def run_job():
    now_str = datetime.now().strftime("%H:%M")
    print(f"\n{'='*60}\nJob: {now_str}\n{'='*60}")

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
        h, m = datetime.now().hour, datetime.now().minute
        if (h == 6 and m >= 25) or (h == 7 and m <= 15):
            if not state["velocity_score_sent"]:
                cl_text = "\n".join(checklist)
                cal_text = "\n".join(cal_flags) if cal_flags else "No special events"
                uw_text = "\n".join(unwind_sigs) if unwind_sigs else "None yet"

                alert(
                    f"🌅 PRE-MARKET REPORT — SPY\n"
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

        # Reset daily flags + end of day auto-fill
        if h >= 13:
            eod_autofill(price)
            state["velocity_score_sent"] = False
            state["consolidation_alert_sent"] = False
            state["hedge_unwind_alert_sent"] = False
            state["open_time_prices"] = []
            state["open_price"] = None
            state["open_iv"] = None
            state["open_volume"] = None
            state["open_drive_detected"] = False
            state["tick_history"] = []

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
# SCHEDULE (original gamma bot times preserved)
# ─────────────────────────────────────────────
schedule.every().day.at("06:00").do(run_job)
schedule.every().day.at("06:30").do(run_job)
schedule.every().day.at("07:00").do(run_job)
schedule.every().day.at("07:45").do(run_job)
schedule.every().day.at("08:30").do(run_job)
schedule.every().day.at("09:15").do(run_job)
schedule.every().day.at("10:00").do(run_job)
schedule.every().day.at("10:45").do(run_job)
schedule.every().day.at("11:30").do(run_job)
schedule.every().day.at("12:15").do(run_job)
schedule.every().day.at("13:00").do(run_job)

schedule.every(5).minutes.do(check_vwap)
schedule.every(5).minutes.do(check_consolidation_job)

print("SPY UNIFIED BOT v3.0")
print("=" * 60)
print("Original Gamma Bot + Intelligence Bot — MERGED")
print("Modules:")
print("  1. GEX Core (original)")
print("  2. VWAP Cross Alerts (original)")
print("  3. Conviction Spike (original)")
print("  4. Regime Detection Engine")
print("  5. Hedge Unwind Detector")
print("  6. Vanna / Charm Engine")
print("  7. Consolidation Trap Detector")
print("  8. VIX / VVIX + Momentum")
print("  9. TICK + Inventory Precision")
print(" 10. Calendar / Quarter System")
print(" 11. Unified Conviction Scorer")
print(" 12. CSV Data Logger")
print("=" * 60)
init_log()
run_job()

while True:
    schedule.run_pending()
    time.sleep(30)
