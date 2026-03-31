import os
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

UW_TOKEN = os.getenv("UW_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
TICKER = "SPY"

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
state = {
    "previous_gex_state": None,
    "previous_ratio": None,
    "previous_vol_gex": None,
    "previous_oi_gex": None,
    "vol_gex_history": [],
    "regime": None,
    "previous_regime": None,
    "vwap_alert_sent": False,
    "velocity_score_sent": False,
    "consolidation_alert_sent": False,
    "hedge_unwind_alert_sent": False,
    "open_price": None,
    "open_iv": None,
    "open_volume": None,
    "open_time_prices": [],
    "last_conviction_score": None,
}

# Update OPEX dates quarterly
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

    # Quarter end
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

    # OPEX proximity
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
    """
    Detects market regime from GEX trajectory.
    Key insight: catch TRANSITION before it completes.

    Regimes:
    BEARISH_HEDGE_BUILD     → Vol GEX accelerating negative
    TRANSITION_ZONE         → Vol GEX decelerating (early warning)
    HEDGE_UNWIND_EARLY      → Vol GEX inflecting positive (caught early)
    HEDGE_UNWIND_CONFIRMED  → Vol GEX flipped positive
    BULLISH_MOMENTUM        → Both OI + Vol positive
    NEUTRAL                 → No clear bias
    """
    if len(vol_gex_history) < 3:
        return "INSUFFICIENT_DATA", 50

    recent = vol_gex_history[-3:]
    roc = recent[-1] - recent[0]
    mid_roc = recent[-1] - recent[-2]
    early_roc = recent[-2] - recent[0]

    vol_positive = vol_gex > 0
    oi_positive = oi_gex > 0

    if vol_positive and oi_positive:
        return "BULLISH_MOMENTUM", 95

    if vol_positive and not oi_positive:
        return "HEDGE_UNWIND_CONFIRMED", 88

    if not vol_positive:
        # Check if rate of change is improving (getting less negative)
        if roc > 0 and mid_roc > 0:
            # Vol GEX improving for 2 consecutive readings — early unwind
            return "HEDGE_UNWIND_EARLY", 72
        elif roc > 0 and mid_roc <= 0:
            # Mixed — transitioning
            return "TRANSITION_ZONE", 58
        else:
            # Still accelerating negative
            roc_pct = abs(roc / recent[0]) * 100 if recent[0] != 0 else 0
            confidence = min(85, 60 + roc_pct)
            return "BEARISH_HEDGE_BUILD", int(confidence)

    return "NEUTRAL", 50


def get_regime_signal(regime, confidence, oi_b, vol_b):
    explanations = {
        "BULLISH_MOMENTUM": (
            "🟢 BULLISH MOMENTUM\n"
            "Both OI and Vol GEX positive.\n"
            "MMs are long gamma — they SELL into strength.\n"
            "Moves are dampened. Good for steady gains, not explosions.\n"
            "→ Calls OK but don't expect 400%+ today."
        ),
        "HEDGE_UNWIND_CONFIRMED": (
            "🚀 HEDGE UNWIND CONFIRMED\n"
            "Vol GEX flipped positive — OI still negative.\n"
            "Institutions CLOSING put hedges.\n"
            "MMs buying shares as delta collateral = price rises mechanically.\n"
            "No new call buying needed — put SELLING is the rocket fuel.\n"
            "→ CALLS strongly favored. This is the 400-700% setup."
        ),
        "HEDGE_UNWIND_EARLY": (
            "🔄 EARLY HEDGE UNWIND SIGNAL ← CAUGHT EARLY\n"
            "Vol GEX still negative but improving rapidly.\n"
            "Institutions BEGINNING to close hedges.\n"
            "This is the signal BEFORE the rocket fires.\n"
            "→ Prepare call entry. Watch for Vol GEX to flip positive."
        ),
        "TRANSITION_ZONE": (
            "⚠️ TRANSITION ZONE\n"
            "Vol GEX decelerating but not reversing yet.\n"
            "Market at genuine decision point.\n"
            "Could become hedge unwind OR resume bearish build.\n"
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
            "No clear directional bias from GEX.\n"
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
    """
    Scans options flow for institutional hedge closing patterns.

    Key signals:
    - PUT Vol/OI > 50x = existing position being CLOSED (bullish)
    - Repeated Hits Descending Fill on puts = aggressive put selling
    - High volume puts at bearish strikes being closed = institutions
      no longer need downside protection = bullish conviction

    The mechanical link:
    Institution sells put → MM who sold the put originally
    had SHORTED shares to hedge → MM now BUYS those shares back
    → Price rises with no new call buying needed
    """
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
        put_close_count = 0
        call_open_count = 0

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

                # HIGH CONVICTION: massive vol/OI on puts = closing hedges
                if is_put and vol_oi_ratio >= 50 and volume >= 10000:
                    unwind_score += 25
                    put_close_count += 1
                    unwind_signals.append(
                        f"🚀 PUT HEDGE CLOSING: ${strike:.0f}P\n"
                        f"   Vol/OI: {round(vol_oi_ratio)}x ({int(volume/1000)}K vol)\n"
                        f"   → Institutions unwinding downside protection"
                    )

                # MODERATE: decent vol/OI on puts
                elif is_put and vol_oi_ratio >= 10 and volume >= 5000:
                    unwind_score += 10
                    put_close_count += 1
                    unwind_signals.append(
                        f"⚠️ PUT CLOSING (moderate): ${strike:.0f}P "
                        f"Vol/OI: {round(vol_oi_ratio)}x"
                    )

                # Descending fill = aggressive put selling
                if is_put and is_descending and volume >= 5000:
                    unwind_score += 15
                    unwind_signals.append(
                        f"🔽 DESCENDING FILL PUT: ${strike:.0f}P\n"
                        f"   → Aggressive institutional put selling"
                    )

                # Call sweeps = fresh bullish positioning
                if is_call and is_sweep and volume >= 5000:
                    call_open_count += 1
                    unwind_score += 8
                    unwind_signals.append(
                        f"📈 CALL SWEEP: ${strike:.0f}C\n"
                        f"   → Fresh bullish entry, {int(volume/1000)}K contracts"
                    )

            except Exception:
                continue

        unwind_score = min(unwind_score, 100)

        if unwind_score >= 40:
            flow_direction = "BULLISH — Hedge unwind active"
            unwind_detected = True
        elif unwind_score >= 20:
            flow_direction = "LEANING BULLISH — Early unwind signs"
            unwind_detected = True
        else:
            flow_direction = "NEUTRAL — No unwind detected"
            unwind_detected = False

        return unwind_detected, unwind_score, unwind_signals[:6], flow_direction

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
        v_resp = requests.get(url, headers=headers, params=params, timeout=10)
        v_data = v_resp.json().get("data", [])

        params["greek"] = "charm"
        c_resp = requests.get(url, headers=headers, params=params, timeout=10)
        c_data = c_resp.json().get("data", [])

        vanna_target = vanna_strength = charm_target = charm_strength = None

        if v_data:
            best = max(v_data, key=lambda x: float(x.get("vanna", 0) or 0))
            vanna_target = float(best.get("strike", 0))
            vanna_strength = float(best.get("vanna", 0) or 0)

        if c_data:
            worst = min(c_data, key=lambda x: float(x.get("charm", 0) or 0))
            charm_target = float(worst.get("strike", 0))
            charm_strength = float(worst.get("charm", 0) or 0)

        # Conflict = both stacked within $1 of each other
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
    vanna_window_open = minutes_since_open < 270  # ~4.5hrs after open
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
    """
    Fires BEFORE the consolidation trap kills premium.
    Checks first 45 minutes for canceling flow patterns.

    Confirmed consolidation requires 50+ score from:
    - Price within 0.5% of vanna target (30pts)
    - IV barely moving < 2% from open (25pts)
    - Volume thin vs open (20pts)
    - Vanna/charm conflict at same strike (15pts)
    - Choppy back/forth < $1.50 range (10pts)
    """
    now = datetime.now()
    mins = (now.hour - 6) * 60 + now.minute - 30
    if mins > 45 or mins < 5:
        return False, 0, []

    state["open_time_prices"].append(current_price)
    score = 0
    signals = []

    # 1. Price proximity to vanna target
    if vanna_target:
        prox = abs(current_price - vanna_target) / vanna_target * 100
        if prox <= 0.5:
            score += 30
            signals.append(
                f"⚠️ Price within 0.5% of vanna ${vanna_target} "
                f"({prox:.2f}% away) — magnetic stall zone"
            )

    # 2. IV barely moving
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

    # 3. Volume thin
    if state["open_volume"] and current_volume:
        vol_ratio = current_volume / state["open_volume"]
        if vol_ratio < 0.7:
            score += 20
            signals.append(
                f"⚠️ Volume only {round(vol_ratio*100)}% of open — "
                f"institutions not participating"
            )

    # 4. Conflict
    if conflict:
        score += 15
        signals.append(
            "⚠️ Vanna + charm stacked at same strike — "
            "directional forces canceling out"
        )

    # 5. Choppy price action
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
        vix_spot = float(
            yf.Ticker("^VIX").history(period="5d", interval="1d")["Close"].iloc[-1]
        )
        vvix_hist = yf.Ticker("^VVIX").history(period="5d", interval="1d")
        vix3m_hist = yf.Ticker("^VIX3M").history(period="2d", interval="1d")

        vvix_val = float(vvix_hist["Close"].iloc[-1]) if not vvix_hist.empty else None
        vix3m_val = float(vix3m_hist["Close"].iloc[-1]) if not vix3m_hist.empty else None

        # Term structure
        if vix3m_val:
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

        # VIX level
        if vix_spot >= 30:
            vix_sig = f"🔴 EXTREME FEAR ({round(vix_spot,1)}) — Explosive moves likely"
        elif vix_spot >= 22:
            vix_sig = f"🟠 ELEVATED ({round(vix_spot,1)}) — Volatile, trending possible"
        elif vix_spot >= 16:
            vix_sig = f"🟡 MODERATE ({round(vix_spot,1)}) — Normal range"
        else:
            vix_sig = f"🟢 LOW ({round(vix_spot,1)}) — Calm, chop risk high"

        # VVIX
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
# MODULE 8: CONVICTION SCORER
# ─────────────────────────────────────────────
def score_conviction(vix_spot, vvix_val, vix_term, vol_gex, prev_vol_gex,
                      regime, unwind_score, cal_bonus,
                      vanna_window_open, conflict, ratio):
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
        checklist.append("✅ VIX Backwardation (+15)")
    elif vix_term == "FLAT":
        score += 7
        checklist.append("⚠️ VIX Flat (+7)")
    else:
        checklist.append("❌ VIX Contango — calm (+0)")

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

    # Unwind flow (15pts)
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
# VWAP CROSS
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
                f"Regime: {state['regime']} | Time: {now}\n\n"
                f"⚠️ Confirm candle close below VWAP before entering."
            )
            state["vwap_alert_sent"] = True

    elif bullish and not state["vwap_alert_sent"]:
        if pp <= pv and cp > cv:
            alert(
                f"🔼 VWAP CROSS — BULLISH ENTRY\n"
                f"Price: ${round(cp,2)} | VWAP: ${round(cv,2)}\n"
                f"Regime: {state['regime']} | Time: {now}\n\n"
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
                f"Consolidation Score: {score}/100\n\n"

                f"⚠️ SIGNALS\n{sigs}\n\n"

                f"💡 WHAT IS HAPPENING\n"
                f"Vanna pulling price UP toward ${vt}\n"
                f"Charm pushing price DOWN toward ${ct}\n"
                f"These forces are CANCELING each other.\n"
                f"Result: tight choppy range — premium decays fast.\n\n"

                f"🚫 DO NOT TRADE THIS OPEN\n"
                f"Wait for ONE of these breaks:\n"
                f"→ Price moves >1% from ${vt} with strong volume\n"
                f"→ IV expands >3% from open\n"
                f"→ TICK holds +600 or -600 for 15+ minutes\n"
                f"→ Hedge unwind score >40 detected in flow\n\n"

                f"Re-assess after 10:00am PDT."
            )
            state["consolidation_alert_sent"] = True
            print(f"Consolidation alert fired. Score: {score}/100")

    except Exception as e:
        print(f"Consolidation job error: {e}")


# ─────────────────────────────────────────────
# MAIN JOB
# ─────────────────────────────────────────────
def run_job():
    now_str = datetime.now().strftime("%H:%M")
    print(f"\n{'='*60}\nJob: {now_str}\n{'='*60}")

    try:
        # Core GEX
        oi_gex, vol_gex, price = fetch_gex()
        if oi_gex is None:
            print("No GEX data")
            return
        if vol_gex == 0:
            print(f"Pre-market. OI: {round(oi_gex/1e9,2)}B | Price: ${price}")
            return

        # Update history
        state["vol_gex_history"].append(vol_gex)
        if len(state["vol_gex_history"]) > 10:
            state["vol_gex_history"].pop(0)

        ratio = abs(vol_gex) / abs(oi_gex) if oi_gex != 0 else 0
        gex_state = get_gex_state(oi_gex, vol_gex)
        oi_b = round(oi_gex / 1e9, 2)
        vol_b = round(vol_gex / 1e9, 2)
        oi_m = oi_gex / (price * 6.31) / 1e6
        vol_m = vol_gex / (price * 6.31) / 1e6
        ratio_r = round(ratio, 2)

        # All modules
        vix_spot, vvix_val, vix_term, term_sig, vix_sig, vvix_sig = fetch_vix_data()
        vt, vs, ct, cs, conflict = fetch_vanna_charm()
        vc_text, vanna_window = get_vanna_charm_read(vt, vs, ct, cs, price, conflict)
        unwind_det, unwind_score, unwind_sigs, flow_dir = fetch_hedge_unwind_signals()
        regime, reg_conf = detect_regime(oi_gex, vol_gex, state["vol_gex_history"])
        reg_signal = get_regime_signal(regime, reg_conf, oi_b, vol_b)
        cal_flags, cal_bonus, days_opex = get_calendar_flags()
        conv, grade, rec, checklist = score_conviction(
            vix_spot, vvix_val, vix_term, vol_gex,
            state["previous_vol_gex"], regime, unwind_score,
            cal_bonus, vanna_window, conflict, ratio
        )

        # Console
        print(f"${price} | {regime} | Score:{conv} | Ratio:{ratio_r}x")
        print(f"OI:{oi_b}B | Vol:{vol_b}B | Flow:{flow_dir}")
        print(f"Grade: {grade} | {rec}")

        # ── REGIME CHANGE ──
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
            emoji = "🚨" if is_key else "🔄"

            cl_text = "\n".join(checklist)
            uw_text = "\n".join(unwind_sigs) if unwind_sigs else "None"
            cal_text = "\n".join(cal_flags) if cal_flags else "No special events"

            alert(
                f"{emoji} REGIME TRANSITION — SPY\n"
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

                f"🗓️ CALENDAR\n{cal_text}\n\n"

                f"🎯 CONVICTION: {conv}/100 — {grade}\n"
                f"→ {rec}\n\n"

                f"📋 CHECKLIST\n{cl_text}"
            )
            print(f"Regime alert: {prev} → {regime}")

        # ── HEDGE UNWIND (standalone) ──
        elif (unwind_det and not state["hedge_unwind_alert_sent"]
              and unwind_score >= 40):
            uw_text = "\n".join(unwind_sigs)
            alert(
                f"🚀 HEDGE UNWIND DETECTED — SPY\n"
                f"{'─'*35}\n"
                f"Score: {unwind_score}/100 | Flow: {flow_dir}\n"
                f"Time: {now_str} | Price: ${price}\n\n"

                f"📊 FLOW SIGNALS\n{uw_text}\n\n"

                f"💡 MECHANICAL EXPLANATION\n"
                f"Institutions selling puts they bought as hedges.\n"
                f"MMs who sold those puts had shorted shares to hedge.\n"
                f"Now MMs BUY those shares back = price rises.\n"
                f"Zero new call buying needed — this IS the fuel.\n\n"

                f"📈 VANNA TARGET: ${vt}\n{vc_text}\n\n"

                f"🎯 CONVICTION: {conv}/100 — {grade}\n→ {rec}"
            )
            state["hedge_unwind_alert_sent"] = True

        if unwind_score < 20:
            state["hedge_unwind_alert_sent"] = False

        # ── CONVICTION SPIKE ──
        if ("DIRECTIONAL" in gex_state and state["previous_ratio"]
                and ratio - state["previous_ratio"] > 0.3):
            direction = "BEARISH" if oi_gex < 0 else "BULLISH"
            alert(
                f"📈 CONVICTION SPIKE — {direction}\n"
                f"Ratio: {ratio_r}x (was {round(state['previous_ratio'],2)}x)\n"
                f"OI: {fmt(oi_m)} | Vol: {fmt(vol_m)}\n"
                f"Regime: {regime} | Price: ${price}\n"
                f"VIX: {vix_sig} | VVIX: {vvix_sig}\n\n"
                f"🎯 Score: {conv}/100 — {grade}\n→ {rec}"
            )

        # ── MORNING REPORT ──
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

        # Reset daily
        if h >= 13:
            state["velocity_score_sent"] = False
            state["consolidation_alert_sent"] = False
            state["hedge_unwind_alert_sent"] = False
            state["open_time_prices"] = []
            state["open_price"] = None
            state["open_iv"] = None
            state["open_volume"] = None

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
# SCHEDULE
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

schedule.every(5).minutes.do(check_consolidation_job)
schedule.every(5).minutes.do(check_vwap)

print("SPY INTELLIGENCE BOT v2.0")
print("=" * 60)
print("Modules Active:")
print("  1. Calendar / Quarter System")
print("  2. GEX Core")
print("  3. Regime Detection Engine")
print("  4. Hedge Unwind Detector")
print("  5. Vanna / Charm Engine")
print("  6. Consolidation Trap Detector")
print("  7. VIX / VVIX")
print("  8. Conviction Scorer")
print("  9. VWAP Cross Alerts")
print("=" * 60)
run_job()

while True:
    schedule.run_pending()
    time.sleep(30)
