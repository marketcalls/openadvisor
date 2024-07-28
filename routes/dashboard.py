from flask import Blueprint, render_template
from flask_login import login_required
import yfinance as yf
import pandas as pd

dashboard_bp = Blueprint('dashboard', __name__)

def get_index_data(symbol):
    try:
        index = yf.Ticker(symbol)
        hist = index.history(period="5d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]
            change_percent = ((current_price - previous_close) / previous_close) * 100
            return current_price, change_percent
        else:
            return 0, 0
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return 0, 0

@dashboard_bp.route('/dashboard')
@login_required
def show_dashboard():
    stocks = [
        "DELHIVERY.NS", "DALBHARAT.NS", "SHREECEM.NS", "LALPATHLAB.NS", "HDFCLIFE.NS",
        "LTIM.NS", "SYNGENE.NS", "BIOCON.NS", "PEL.NS", "JINDALSTEL.NS"
    ]
    
    stock_data = {}
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            hist = ticker.history(period="5d")
            if not hist.empty:
                stock_data[stock] = {
                    'current_price': hist['Close'].iloc[-1]
                }
            else:
                stock_data[stock] = {'current_price': 0}
        except Exception as e:
            print(f"Error fetching data for {stock}: {str(e)}")
            stock_data[stock] = {'current_price': 0}
    
    # Fetch market indices data
    market_indices = {}
    indices = {
        "Nifty": "^NSEI",
        "Sensex": "^BSESN",
        "Bank Nifty": "^NSEBANK"
    }
    for index_name, symbol in indices.items():
        value, change = get_index_data(symbol)
        market_indices[index_name] = {
            'value': value,
            'change': change
        }

    # Simulated data for account balance and weekly profit
    account_balance = 4200000  # ₹42,00,000
    weekly_profit = 27000  # ₹27,000
    
    return render_template('dashboard.html', 
                           stock_data=stock_data, 
                           account_balance=account_balance, 
                           weekly_profit=weekly_profit,
                           market_indices=market_indices)