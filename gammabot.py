import os
import requests
import schedule
import time
import asyncio
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

UW_TOKEN = os.getenv("UW_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
TICKER = "SPY"

previous_state = None
previous_ratio = None
vwap_alert_sent = False

def fetch_gex():
    url = f"https://api.unusualwhales.com/api/stock/{TICKER}/spot-exposures"
    headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()["data"]
    if not data:
        return None, None, None
    latest = data[-1]
    oi_gex = float(latest["gamma_per_one_percent_move_oi"])
    vol_gex = float(latest["gamma_per_one_percent_move_vol"])
    price = float(latest["price"])
    return oi_gex, vol_gex, price

def get_state(oi_gex, vol_gex):
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
    else:
        direction = "BEARISH" if oi_gex < 0 else "BULLISH"
        return f"DIRECTIONAL_{direction}"

def format_gex(value_m):
    abs_val = abs(value_m)
    sign = "-" if value_m < 0 else ""
    if abs_val >= 1000:
        return f"{sign}{abs_val/1000:.1f}B"
    elif abs_val >= 1:
        return f"{sign}{abs_val:.1f}M"
    else:
        return f"{sign}{abs_val*1000:.0f}K"

def get_vwap():
    try:
        spy = yf.download("SPY", period="1d", interval="5m", progress=False)
        if spy.empty:
            return None, None, None, None
        spy["vwap"] = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
        current_price = float(spy["Close"].iloc[-1])
        current_vwap = float(spy["vwap"].iloc[-1])
        prev_price = float(spy["Close"].iloc[-2]) if len(spy) > 1 else current_price
        prev_vwap = float(spy["vwap"].iloc[-2]) if len(spy) > 1 else current_vwap
        return current_price, current_vwap, prev_price, prev_vwap
    except Exception as e:
        print(f"VWAP error: {e}")
        return None, None, None, None

def is_market_open():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=6, minute=30, second=0)
    market_close = now.replace(hour=13, minute=0, second=0)
    return market_open <= now <= market_close

async def send_message(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)

def alert(text):
    asyncio.run(send_message(text))

def check_vwap():
    global vwap_alert_sent
    if not previous_state or not is_market_open():
        return
    if "DIRECTIONAL" not in previous_state:
        vwap_alert_sent = False
        return

    result = get_vwap()
    if result[0] is None:
        return

    current_price, current_vwap, prev_price, prev_vwap = result
    now = datetime.now().strftime("%H:%M")
    is_bearish = "BEARISH" in previous_state
    is_bullish = "BULLISH" in previous_state

    if is_bearish and not vwap_alert_sent:
        if prev_price >= prev_vwap and current_price < current_vwap:
            msg = (f"🔽 VWAP CROSS — BEARISH ENTRY SIGNAL\n\n"
                   f"SPY dropped below VWAP\n"
                   f"Price: ${round(current_price, 2)}\n"
                   f"VWAP: ${round(current_vwap, 2)}\n"
                   f"GEX Bias: {previous_state}\n"
                   f"Time: {now}\n\n"
                   f"Check chart for candle confirmation.")
            alert(msg)
            vwap_alert_sent = True
            print(f"VWAP cross alert sent (bearish) at {now}")

    elif is_bullish and not vwap_alert_sent:
        if prev_price <= prev_vwap and current_price > current_vwap:
            msg = (f"🔼 VWAP CROSS — BULLISH ENTRY SIGNAL\n\n"
                   f"SPY broke above VWAP\n"
                   f"Price: ${round(current_price, 2)}\n"
                   f"VWAP: ${round(current_vwap, 2)}\n"
                   f"GEX Bias: {previous_state}\n"
                   f"Time: {now}\n\n"
                   f"Check chart for candle confirmation.")
            alert(msg)
            vwap_alert_sent = True
            print(f"VWAP cross alert sent (bullish) at {now}")

    if is_bearish and current_price > current_vwap:
        vwap_alert_sent = False
    elif is_bullish and current_price < current_vwap:
        vwap_alert_sent = False

def run_job():
    global previous_state, previous_ratio
    try:
        oi_gex, vol_gex, price = fetch_gex()
        if oi_gex is None:
            print("No data returned")
            return
        if vol_gex == 0:
            print(f"VOL GEX is 0 — pre-market. OI GEX: {round(oi_gex/1e9,2)}B | Price: {price}")
            return
        ratio = abs(vol_gex) / abs(oi_gex)
        state = get_state(oi_gex, vol_gex)

        # Raw billions
        oi_b = round(oi_gex / 1_000_000_000, 2)
        vol_b = round(vol_gex / 1_000_000_000, 2)

        # Net GEX in millions (UW style)
        oi_m = oi_gex / (price * 6.31) / 1_000_000
        vol_m = vol_gex / (price * 6.31) / 1_000_000

        oi_fmt = format_gex(oi_m)
        vol_fmt = format_gex(vol_m)

        ratio_r = round(ratio, 2)
        now = datetime.now().strftime("%H:%M")

        print(f"{now} | Spot: ${price} | OI: {oi_b}B ({oi_fmt}) | VOL: {vol_b}B ({vol_fmt}) | Ratio: {ratio_r} | {state}")

        if state != previous_state:
            emojis = {"NEUTRAL":"⚪","WATCH":"⚠️","DIRECTIONAL_BEARISH":"🔴","DIRECTIONAL_BULLISH":"🟢","COUNTER":"🔄"}
            emoji = emojis.get(state, "❓")
            msg = (f"{emoji} SPY GEX SIGNAL\n\n"
                   f"State: {state}\n"
                   f"OI Net GEX: {oi_fmt} ({oi_b}B raw)\n"
                   f"VOL Net GEX: {vol_fmt} ({vol_b}B raw)\n"
                   f"Ratio: {ratio_r}x\n"
                   f"Spot: ${price}\n"
                   f"Time: {now}\n\n"
                   f"Previous: {previous_state or 'None'}")
            alert(msg)
            print(f"Alert sent: {state}")

        elif "DIRECTIONAL" in (state or "") and previous_ratio:
            if ratio - previous_ratio > 0.3:
                direction = "BEARISH" if oi_gex < 0 else "BULLISH"
                msg = (f"📈 CONVICTION INCREASING\n\n"
                       f"Direction: {direction}\n"
                       f"Ratio: {ratio_r}x (was {round(previous_ratio,2)}x)\n"
                       f"OI Net GEX: {oi_fmt}\n"
                       f"VOL Net GEX: {vol_fmt}\n"
                       f"Spot: ${price}\n"
                       f"Time: {now}")
                alert(msg)

        previous_state = state
        previous_ratio = ratio

    except Exception as e:
        print(f"Error: {e}")

# GEX checks
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

# VWAP checks every 5 minutes during market hours
schedule.every(5).minutes.do(check_vwap)

print("SPY GEX Bot running...")
run_job()

while True:
    schedule.run_pending()
    time.sleep(30)