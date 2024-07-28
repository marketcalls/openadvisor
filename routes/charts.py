# routes/charts.py
from flask import Blueprint, render_template, jsonify, request
from sqlalchemy import create_engine
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os

load_dotenv()

charts_bp = Blueprint('charts', __name__)

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set the path to the instance folder
instance_folder = os.path.join(current_directory, '..', 'instance')

# Construct the absolute path to the database file within the instance folder
db_file_path = os.path.join(instance_folder, os.getenv('SQLALCHEMY_DATABASE_URI').replace('sqlite:///', ''))

# Set up SQLAlchemy engine
engine = create_engine(f'sqlite:///{db_file_path}', echo=False)

def fetch_db_data(ticker, ema_period=20, rsi_period=14):
    query = f"SELECT date, open, high, low, close, volume FROM finance_data WHERE symbol = '{ticker}'"
    data = pd.read_sql(query, engine)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data['EMA'] = ta.ema(data['close'], length=ema_period)
    data['RSI'] = ta.rsi(data['close'], length=rsi_period)

    candlestick_data = [
        {
            'time': int(row.Index.timestamp()),
            'open': row.open,
            'high': row.high,
            'low': row.low,
            'close': row.close
        }
        for row in data.itertuples()
    ]

    ema_data = [
        {
            'time': int(row.Index.timestamp()),
            'value': row.EMA
        }
        for row in data.itertuples() if not pd.isna(row.EMA)
    ]

    rsi_data = [
        {
            'time': int(row.Index.timestamp()),
            'value': row.RSI if not pd.isna(row.RSI) else 0  # Convert NaN to zero
        }
        for row in data.itertuples()
    ]

    return candlestick_data, ema_data, rsi_data

@charts_bp.route('/charts')
def charts():
    symbol = request.args.get('symbol', 'RELIANCE.NS')
    return render_template('charts.html', symbol=symbol)

@charts_bp.route('/api/data/<ticker>/<int:ema_period>/<int:rsi_period>')
def get_data(ticker, ema_period, rsi_period):
    candlestick_data, ema_data, rsi_data = fetch_db_data(ticker, ema_period, rsi_period)
    return jsonify({'candlestick': candlestick_data, 'ema': ema_data, 'rsi': rsi_data})

@charts_bp.route('/api/symbols')
def get_symbols():
    query = "SELECT DISTINCT symbol FROM finance_data"
    symbols = pd.read_sql(query, engine)['symbol'].tolist()
    return jsonify(symbols)
