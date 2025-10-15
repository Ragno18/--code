from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# IMPORTANT: In a real-world application, you would need to get an API key
# from Alpha Vantage and store it securely, for example, as an environment variable.
# For this example, we will use 'demo' which has limitations.
ALPHA_VANTAGE_API_KEY = "demo"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# This is a simplified list of stocks to query. A real application might
# get this list from another source or have a more dynamic way of selecting stocks.
SYMBOLS_TO_QUERY = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "JNJ"]


def get_stock_data(symbol):
    """
    Fetches real-time quote for a given stock symbol from Alpha Vantage.
    Includes fallback to dummy data if API limit is reached or other errors occur.
    """
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"API Response for {symbol}: {data}")  # More detailed logging

        if "Note" in data:
            print(f"API limit note for {symbol}: {data['Note']}")
            return create_dummy_stock(symbol)

        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            if not quote:
                return create_dummy_stock(symbol)
            return {
                "symbol": quote.get("01. symbol"),
                "price": float(quote.get("05. price", 0)),
            }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
    except (KeyError, ValueError) as e:
        print(f"Error processing data for {symbol}: {e}")

    return create_dummy_stock(symbol)

def create_dummy_stock(symbol):
    """Creates a dummy stock object for a given symbol."""
    price = (hash(symbol) % 500) + 100
    print(f"Creating dummy data for {symbol}: Price ${price}")
    return {"symbol": symbol, "price": price}


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    min_price = request.args.get('min_price', default=0, type=float)
    max_price = request.args.get('max_price', default=10000, type=float)

    all_stocks = []
    for symbol in SYMBOLS_TO_QUERY:
        stock_data = get_stock_data(symbol)
        if stock_data:
            all_stocks.append(stock_data)

    filtered_stocks = [
        stock for stock in all_stocks
        if stock and min_price <= stock.get('price', 0) <= max_price
    ]

    sorted_stocks = sorted(filtered_stocks, key=lambda x: x.get('price', 0), reverse=True)
    top_5_stocks = sorted_stocks[:5]

    return jsonify(top_5_stocks)

if __name__ == '__main__':
    app.run(debug=True, port=5000)