"""
🎯 SPY 0DTE Gamma & OI Morning Alert Bot
─────────────────────────────────────────
Sends a pre-market morning brief to Telegram every trading day at 9:00 AM ET covering:
  - Net Gamma Exposure (GEX proxy from OI)
  - Gamma Flip Level (bull/bear line)
  - High OI Strike Walls (call & put)
  - Max Pain level
  - Day Type Classification (Trend / Inventory / Morning Spike)
  - Bias & 0DTE guidance

Run:
  /Users/s19_72/Desktop/PythonXynth/env/bin/python gamma_bot.py
"""

import os
import time
import logging
import datetime
import pytz
import requests
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIG  — paste your new token here
# ─────────────────────────────────────────
BOT_TOKEN = "8432038309:AAHaaseWEWiFr_ybWW_0k0K9nS-sMLlc5Js"
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "8103550091")

TICKER         = "SPY"
ET             = pytz.timezone("America/New_York")
ALERT_HOUR     = 9       # 9:00 AM ET
ALERT_MINUTE   = 0
CHECK_SLEEP_S  = 30      # check time every 30 seconds

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gamma_bot.log")],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────
def send(msg: str) -> None:
    try:
        r = requests.post(
            "https://api.telegram.org/bot" + BOT_TOKEN + "/sendMessage",
            json={
                "chat_id":    CHAT_ID,
                "text":       msg,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        r.raise_for_status()
        log.info("Telegram message sent.")
    except Exception as e:
        log.error("Telegram error: " + str(e))


# ─────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────
def get_spot_price() -> float:
    ticker = yf.Ticker(TICKER)
    hist = ticker.history(period="1d", interval="1m")
    return round(float(hist["Close"].iloc[-1]), 2)


def get_vix() -> float:
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d", interval="1m")
        return round(float(hist["Close"].iloc[-1]), 2)
    except:
        return 20.0


def get_prev_close() -> float:
    hist = yf.Ticker(TICKER).history(period="5d", interval="1d")
    if len(hist) >= 2:
        return round(float(hist["Close"].iloc[-2]), 2)
    return get_spot_price()


def get_options_chain(spot: float):
    """
    Fetch options chain for the nearest 0DTE expiry.
    Returns (calls_df, puts_df, expiry_str).
    """
    ticker = yf.Ticker(TICKER)
    expirations = ticker.options

    if not expirations:
        return None, None, None

    today_str = datetime.datetime.now(ET).strftime("%Y-%m-%d")

    # Pick nearest expiry (today if available, else soonest)
    expiry = expirations[0]
    for exp in expirations:
        if exp >= today_str:
            expiry = exp
            break

    chain = ticker.option_chain(expiry)
    calls = chain.calls
    puts  = chain.puts

    # Filter to strikes within 5% of spot (relevant range)
    lo = spot * 0.95
    hi = spot * 1.05
    calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].copy()
    puts  = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].copy()

    return calls, puts, expiry


# ─────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────
def calc_max_pain(calls, puts) -> float:
    """
    Max Pain = strike where total options losses are minimized.
    At each strike, calculate total ITM value for all calls + puts.
    The strike with the minimum total pain to option buyers = max pain.
    """
    strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]

    for test_strike in strikes:
        call_pain = 0.0
        for _, row in calls.iterrows():
            if test_strike > row["strike"]:
                call_pain += (test_strike - row["strike"]) * row["openInterest"]

        put_pain = 0.0
        for _, row in puts.iterrows():
            if test_strike < row["strike"]:
                put_pain += (row["strike"] - test_strike) * row["openInterest"]

        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = test_strike

    return float(max_pain_strike)


def calc_gex_by_strike(calls, puts, spot: float):
    """
    Approximate GEX per strike.
    Dealer GEX = -1 * OI * Gamma * 100 * spot^2 * 0.01
    We approximate gamma using a simplified Black-Scholes-like formula.
    Positive bars = dealers long gamma (dampens moves).
    Negative bars = dealers short gamma (amplifies moves).
    Returns dict of {strike: gex_value}
    """
    from math import exp, log, sqrt, pi

    def approx_gamma(S, K, T, sigma=0.15):
        if T <= 0 or sigma <= 0:
            return 0
        try:
            d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
            return exp(-0.5 * d1 ** 2) / (S * sigma * sqrt(2 * pi * T))
        except:
            return 0

    today = datetime.datetime.now(ET).date()

    gex = {}

    for _, row in calls.iterrows():
        K  = row["strike"]
        OI = row["openInterest"] or 0
        iv = row.get("impliedVolatility", 0.15) or 0.15
        # Use mid-point expiry T ~ 1 day as proxy for 0DTE
        T  = max(1 / 252, 1 / 252)
        g  = approx_gamma(spot, K, T, iv)
        # Dealers are short calls → negative GEX contribution
        gex[K] = gex.get(K, 0) - OI * g * 100 * spot * spot * 0.01

    for _, row in puts.iterrows():
        K  = row["strike"]
        OI = row["openInterest"] or 0
        iv = row.get("impliedVolatility", 0.15) or 0.15
        T  = max(1 / 252, 1 / 252)
        g  = approx_gamma(spot, K, T, iv)
        # Dealers are long puts → positive GEX contribution (they bought puts)
        gex[K] = gex.get(K, 0) + OI * g * 100 * spot * spot * 0.01

    return gex


def calc_net_gex(gex: dict) -> float:
    return round(sum(gex.values()), 2)


def calc_gamma_flip(gex: dict, spot: float) -> float:
    """
    Gamma flip = strike closest to where cumulative GEX crosses zero.
    Above flip = positive gamma (dealers stabilize price).
    Below flip = negative gamma (dealers amplify price moves).
    """
    strikes = sorted(gex.keys())
    cumulative = 0.0
    flip = spot  # default to spot if no cross found

    for s in strikes:
        prev = cumulative
        cumulative += gex[s]
        if prev * cumulative < 0:  # sign change = zero cross
            flip = s
            break

    return float(flip)


def calc_oi_walls(calls, puts, spot: float, n: int = 3):
    """
    Find top N call OI strikes (resistance walls) and put OI strikes (support walls)
    above and below spot respectively.
    """
    call_above = calls[calls["strike"] > spot].nlargest(n, "openInterest")
    put_below  = puts[puts["strike"] < spot].nlargest(n, "openInterest")

    call_walls = sorted(call_above["strike"].tolist())
    put_walls  = sorted(put_below["strike"].tolist(), reverse=True)

    return call_walls, put_walls


# ─────────────────────────────────────────
# DAY TYPE CLASSIFICATION
# ─────────────────────────────────────────
def classify_day(spot: float, prev_close: float, vix: float,
                 net_gex: float, max_pain: float, gamma_flip: float) -> dict:
    """
    Classify the likely day type based on:
    - Gap size vs prev close
    - VIX level
    - Net GEX (negative = amplify, positive = dampen)
    - Distance from Max Pain
    - Position relative to Gamma Flip
    """
    gap_pct        = (spot - prev_close) / prev_close * 100
    above_flip     = spot > gamma_flip
    pain_distance  = abs(spot - max_pain)
    pain_pct       = pain_distance / spot * 100
    large_gap      = abs(gap_pct) >= 0.5
    high_vix       = vix >= 20
    deep_neg_gex   = net_gex < -1_000_000

    # ── Trend Day ─────────────────────────────────────────────────
    # Strong gap + negative GEX + high VIX + away from max pain
    if deep_neg_gex and high_vix and pain_pct > 0.3:
        day_type = "TREND"
        emoji    = "🟢" if gap_pct > 0 else "🔴"
        label    = emoji + " TREND DAY"
        guidance = (
            "Dealers are SHORT gamma — they must chase price to hedge.\n"
            "Moves will be amplified. DO NOT fade early momentum.\n"
            "Strategy: Buy the first pullback, ride in direction of gap.\n"
            "Let 0DTE winners run. Trail stops, don't take quick profits."
        )

    # ── Morning Spike / Fade Day ───────────────────────────────────
    # Large gap but positive GEX or price near max pain
    elif large_gap and (not deep_neg_gex or pain_pct < 0.3):
        day_type = "MORNING_SPIKE"
        emoji    = "🟡"
        label    = "🟡 MORNING SPIKE / FADE"
        guidance = (
            "Large gap but gamma environment suggests dampening.\n"
            "Expect a strong open followed by mean reversion.\n"
            "Strategy: Fade the morning spike after first 15-30 min.\n"
            "Take profits quickly — consolidation likely by 11 AM ET.\n"
            "Avoid holding 0DTE past midday."
        )

    # ── Inventory / Balance Day ────────────────────────────────────
    # Small gap, low VIX, price near max pain, positive/neutral GEX
    else:
        day_type = "INVENTORY"
        emoji    = "⚪"
        label    = "⚪ INVENTORY / BALANCE DAY"
        guidance = (
            "Price is near Max Pain — market makers want it here.\n"
            "Low conviction directional moves expected. Choppy range.\n"
            "Strategy: Fade extremes of the range. Sell premium.\n"
            "AVOID momentum 0DTE plays — theta decay kills directional bets.\n"
            "Wait for a clear break of OI walls before committing."
        )

    return {
        "day_type":  day_type,
        "label":     label,
        "guidance":  guidance,
        "gap_pct":   round(gap_pct, 2),
        "above_flip": above_flip,
    }


# ─────────────────────────────────────────
# BUILD & SEND MORNING BRIEF
# ─────────────────────────────────────────
def send_morning_brief() -> None:
    log.info("Building morning brief...")

    try:
        spot       = get_spot_price()
        prev_close = get_prev_close()
        vix        = get_vix()
        calls, puts, expiry = get_options_chain(spot)

        if calls is None or calls.empty or puts is None or puts.empty:
            send("⚠️ Could not fetch options chain data for " + TICKER + " today.")
            return

        max_pain    = calc_max_pain(calls, puts)
        gex         = calc_gex_by_strike(calls, puts, spot)
        net_gex     = calc_net_gex(gex)
        gamma_flip  = calc_gamma_flip(gex, spot)
        call_walls, put_walls = calc_oi_walls(calls, puts, spot)
        day         = classify_day(spot, prev_close, vix, net_gex, max_pain, gamma_flip)

        # Format OI walls
        call_wall_str = "  |  ".join(["$" + str(w) for w in call_walls]) if call_walls else "N/A"
        put_wall_str  = "  |  ".join(["$" + str(w) for w in put_walls])  if put_walls  else "N/A"

        # GEX sentiment
        if net_gex < -5_000_000:
            gex_label = "🔴 DEEPLY NEGATIVE — High volatility, trending"
        elif net_gex < 0:
            gex_label = "🟠 NEGATIVE — Elevated volatility likely"
        elif net_gex < 5_000_000:
            gex_label = "🟡 SLIGHTLY POSITIVE — Mixed, range possible"
        else:
            gex_label = "🟢 POSITIVE — Dealers dampen moves, chop likely"

        # Flip bias
        flip_bias = "BULLISH (above flip)" if day["above_flip"] else "BEARISH (below flip)"

        now_str = datetime.datetime.now(ET).strftime("%A %b %d, %Y — %I:%M %p ET")

        msg = (
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 *SPY 0DTE MORNING BRIEF*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "_" + now_str + "_\n\n"

            "📍 *SPOT:* `$" + str(spot) + "`\n"
            "📉 *Prev Close:* `$" + str(prev_close) + "`\n"
            "📊 *Gap:* `" + ("+" if day["gap_pct"] >= 0 else "") + str(day["gap_pct"]) + "%`\n"
            "💥 *VIX:* `" + str(vix) + "`\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚡ *GAMMA ENVIRONMENT*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Net GEX: `" + str(round(net_gex / 1_000_000, 2)) + "M`\n"
            "" + gex_label + "\n\n"
            "🔀 *Gamma Flip:* `$" + str(round(gamma_flip, 2)) + "`\n"
            "Bias: *" + flip_bias + "*\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🧱 *KEY LEVELS*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 Max Pain: `$" + str(round(max_pain, 2)) + "`\n"
            "🔴 Call Walls (resistance): `" + call_wall_str + "`\n"
            "🟢 Put Walls (support): `" + put_wall_str + "`\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "*" + day["label"] + "*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "" + day["guidance"] + "\n\n"

            "_Expiry used: " + str(expiry) + " | Data: Yahoo Finance_"
        )

        send(msg)
        log.info("Morning brief sent.")

    except Exception as e:
        log.error("Error building brief: " + str(e))
        send("⚠️ Gamma bot error: " + str(e))


# ─────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────
def is_market_day() -> bool:
    """Skip weekends."""
    return datetime.datetime.now(ET).weekday() < 5


def main() -> None:
    log.info("Gamma bot starting...")
    send(
        "🤖 *SPY Gamma Bot is live!*\n"
        "I'll send your 0DTE morning brief every trading day at 9:00 AM ET.\n"
        "Covering: GEX, Gamma Flip, OI Walls, Max Pain & Day Type."
    )

    last_sent_date = None

    while True:
        now = datetime.datetime.now(ET)
        today = now.date()

        if (
            is_market_day()
            and now.hour == ALERT_HOUR
            and now.minute == ALERT_MINUTE
            and last_sent_date != today
        ):
            send_morning_brief()
            last_sent_date = today

        time.sleep(CHECK_SLEEP_S)


if __name__ == "__main__":
    # Uncomment the line below to test immediately without waiting for 9 AM:
    # send_morning_brief()
    main()