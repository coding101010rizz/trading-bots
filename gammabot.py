import os
import requests
import schedule
import time
import asyncio
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

async def send_message(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)

def alert(text):
    asyncio.run(send_message(text))

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
        oi_b = round(oi_gex / 1_000_000_000, 2)
        vol_b = round(vol_gex / 1_000_000_000, 2)
        ratio_r = round(ratio, 2)
        now = datetime.now().strftime("%H:%M")
        print(f"{now} | Spot: ${price} | OI: {oi_b}B | VOL: {vol_b}B | Ratio: {ratio_r} | {state}")
        if state != previous_state:
            emojis = {"NEUTRAL":"⚪","WATCH":"⚠️","DIRECTIONAL_BEARISH":"🔴","DIRECTIONAL_BULLISH":"🟢","COUNTER":"🔄"}
            emoji = emojis.get(state, "❓")
            msg = f"{emoji} SPY GEX SIGNAL\n\nState: {state}\nOI GEX: ${oi_b}B\nVOL GEX: ${vol_b}B\nRatio: {ratio_r}x\nSpot: ${price}\nTime: {now}\n\nPrevious: {previous_state or 'None'}"
            alert(msg)
            print(f"Alert sent: {state}")
        elif "DIRECTIONAL" in (state or "") and previous_ratio:
            if ratio - previous_ratio > 0.3:
                direction = "BEARISH" if oi_gex < 0 else "BULLISH"
                msg = f"📈 CONVICTION INCREASING\n\nDirection: {direction}\nRatio: {ratio_r}x (was {round(previous_ratio,2)}x)\nSpot: ${price}\nTime: {now}"
                alert(msg)
        previous_state = state
        previous_ratio = ratio
    except Exception as e:
        print(f"Error: {e}")

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

print("SPY GEX Bot running...")
run_job()

while True:
    schedule.run_pending()
    time.sleep(30)