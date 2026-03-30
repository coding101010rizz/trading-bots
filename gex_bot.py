import requests
import schedule
import time
import asyncio
from telegram import Bot

# --- CONFIG ---
MASSIVE_API_KEY = "ITZeJKUWMFMsMqDed9m9moXhe3pIl6gU"
TELEGRAM_TOKEN = "8748913300:AAE27NoB0TMo05sGK5iVRq0kaXtEZeVkvPE"
CHAT_ID = 8103550091
TICKER = "SPY"

# --- STATE ---
previous_state = None
previous_ratio = None

# --- FETCH GEX DATA ---
def fetch_gex():
    url = f"https://api.massive.com/v3/snapshot/options/{TICKER}"
    params = {
        "apiKey": MASSIVE_API_KEY,
        "limit": 50,
        "order": "asc"
    }
    
    net_oi_gex = 0
    net_vol_gex = 0
    spot_price = 637
    next_url = url
    page = 0

    while next_url and page < 20:
        if page == 0:
            response = requests.get(next_url, params=params)
        else:
            response = requests.get(next_url)
        
        data = response.json()
        results = data.get("results", [])

        for contract in results:
            greeks = contract.get("greeks", {})
            gamma = greeks.get("gamma")
            if not gamma:
                continue

            oi = contract.get("open_interest", 0) or 0
            volume = contract.get("day", {}).get("volume", 0) or 0
            contract_type = contract.get("details", {}).get("contract_type", "")
            sign = 1 if contract_type == "call" else -1

            net_oi_gex += gamma * oi * 100 * spot_price * sign
            net_vol_gex += gamma * volume * 100 * spot_price * sign

        next_url = data.get("next_url")
        page += 1

    return net_oi_gex, net_vol_gex, spot_price

# --- DETERMINE SIGNAL STATE ---
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

# --- SEND TELEGRAM MESSAGE ---
async def send_message(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)

def alert(text):
    asyncio.run(send_message(text))

# --- MAIN JOB ---
def run_job():
    global previous_state, previous_ratio

    try:
        oi_gex, vol_gex, spot = fetch_gex()

        if oi_gex == 0:
            print("No GEX data returned")
            return

        ratio = abs(vol_gex) / abs(oi_gex) if oi_gex != 0 else 0
        state = get_state(oi_gex, vol_gex)

        oi_m = round(oi_gex / 1_000_000, 2)
        vol_m = round(vol_gex / 1_000_000, 2)
        ratio_r = round(ratio, 2)

        print(f"OI GEX: {oi_m}M | VOL GEX: {vol_m}M | Ratio: {ratio_r} | State: {state}")

        if state != previous_state:
            if state == "NEUTRAL":
                emoji = "⚪"
            elif state == "WATCH":
                emoji = "⚠️"
            elif state == "DIRECTIONAL_BEARISH":
                emoji = "🔴"
            elif state == "DIRECTIONAL_BULLISH":
                emoji = "��"
            elif state == "COUNTER":
                emoji = "🔄"
            else:
                emoji = "❓"

            msg = f"""{emoji} SPY GEX ALERT

State: {state}
OI GEX: ${oi_m}M
VOL GEX: ${vol_m}M
Ratio: {ratio_r}x
Spot: ~{spot}

Previous: {previous_state or 'None'}"""

            alert(msg)
            print(f"Alert sent: {state}")

        elif "DIRECTIONAL" in state and previous_ratio:
            if ratio - previous_ratio > 0.3:
                direction = "BEARISH" if oi_gex < 0 else "BULLISH"
                msg = f"""📈 CONVICTION INCREASING — SPY

Direction: {direction}
Ratio: {ratio_r}x (was {round(previous_ratio, 2)}x)
OI GEX: ${oi_m}M
VOL GEX: ${vol_m}M"""
                alert(msg)

        previous_state = state
        previous_ratio = ratio

    except Exception as e:
        print(f"Error: {e}")

# --- SCHEDULE PDT ---
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
