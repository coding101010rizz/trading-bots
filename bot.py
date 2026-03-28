import os
import time
import logging
from datetime import datetime, timezone
import requests
import yfinance as yf
import feedparser
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = "8740327960:AAGyX2S2kP6AgLx9XGJ09szkoLs8MAWAzdA"
CHAT_ID = "8103550091"
WATCHLIST = ["SPY", "QQQ"]
PRICE_ALERT_PCT = float(os.getenv("PRICE_ALERT_PCT", "1.0"))
CHECK_INTERVAL_S = int(os.getenv("CHECK_INTERVAL_S", "300"))

NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ&region=US&lang=en-US",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
]

MACRO_KEYWORDS = [
    "fed", "federal reserve", "fomc", "rate hike", "rate cut", "interest rate",
    "jerome powell", "cpi", "inflation", "pce", "gdp", "jobs report",
    "nonfarm", "unemployment", "recession", "yield curve", "treasury",
    "tariff", "trade war", "s&p 500", "nasdaq", "spy", "qqq",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

prev_prices = {}
seen_news = set()


def send(msg):
    try:
        requests.post(
            "https://api.telegram.org/bot" + BOT_TOKEN + "/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": msg,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception as e:
        log.error("Telegram error: " + str(e))


def check_prices():
    for ticker in WATCHLIST:
        try:
            hist = yf.Ticker(ticker).history(period="1d", interval="1m")
            if hist.empty:
                continue
            px = round(float(hist["Close"].iloc[-1]), 2)
            if ticker in prev_prices:
                chg = (px - prev_prices[ticker]) / prev_prices[ticker] * 100
                if abs(chg) >= PRICE_ALERT_PCT:
                    arrow = "DOWN" if chg < 0 else "UP"
                    send("Price Alert: " + ticker + " " + arrow + "\nPrice: $" + str(px) + " (" + str(round(chg, 2)) + "%)")
            prev_prices[ticker] = px
            log.info(ticker + " = $" + str(px))
        except Exception as e:
            log.error(ticker + " error: " + str(e))


def check_news():
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                eid = entry.get("id") or entry.get("link", "")
                if eid in seen_news:
                    continue
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                link = entry.get("link", "")
                combined = (title + " " + summary).lower()
                if any(k in combined for k in MACRO_KEYWORDS):
                    short = summary[:280]
                    send("Macro News: " + title + "\n\n" + short + "\n\n" + link)
                seen_news.add(eid)
                if len(seen_news) > 1000:
                    seen_news.clear()
        except Exception as e:
            log.error("RSS error: " + str(e))


def main():
    log.info("Bot starting...")
    send("SPY & QQQ Alert Bot is live! Watching SPY and QQQ. Price threshold: +-" + str(PRICE_ALERT_PCT) + "%")
    while True:
        check_prices()
        check_news()
        log.info("Sleeping " + str(CHECK_INTERVAL_S) + "s...")
        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
