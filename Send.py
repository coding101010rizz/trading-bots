from flask import Flask, request
import alpaca_trade_api as tradeapi

app = Flask(__name__)

# Alpaca credentials - Use environment variables for better security
API_KEY = 'YOUR_API_KEY'
SECRET_KEY = 'YOUR_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize the Alpaca REST API
api = tradeapi.REST(API_KEY, SECRET_KEY, base_url=BASE_URL, api_version='v2')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return {'status': 'error', 'message': 'No data received'}, 400

    # Parse TradingView alert
    action = data.get('action') # 'buy' or 'sell'
    ticker = data.get('ticker')
    quantity = data.get('quantity', 1)

    try:
        # Execute order on Alpaca
        if action in ['buy', 'sell']:
            api.submit_order(
                symbol=ticker,
                qty=quantity,
                side=action,
                type='market',
                time_in_force='gtc'
            )
            return {'status': 'success', 'message': f'Order for {ticker} submitted'}, 200
        else:
            return {'status': 'error', 'message': 'Invalid action'}, 400
            
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

if __name__ == '__main__':
    # Port 80 often requires root privileges; port 5000 is the Flask default
    app.run(host='0.0.0.0', port=80)
