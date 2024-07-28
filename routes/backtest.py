from flask import Blueprint, render_template, request, jsonify
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback

load_dotenv()

backtest_bp = Blueprint('backtest', __name__)

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set the path to the instance folder
instance_folder = os.path.join(current_directory, '..', 'instance')

# Construct the absolute path to the database file within the instance folder
db_file_path = os.path.join(instance_folder, os.getenv('SQLALCHEMY_DATABASE_URI').replace('sqlite:///', ''))

# Set the path to the training folder
training_folder = os.path.join(current_directory, '..', os.getenv('TRAINING_FOLDER', 'training'))

# SQLAlchemy setup
engine = create_engine(f'sqlite:///{db_file_path}', echo=False)

@backtest_bp.route('/backtest', methods=['GET', 'POST'])
def backtest():
    if request.method == 'POST':
        try:
            data = request.get_json()
            print("Received POST request with data:", data)
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
            initial_capital = float(data['initial_capital'])

            results = perform_backtest(start_date, end_date, initial_capital)
            return jsonify(results)
        except ValueError as ve:
            print(f"ValueError: {str(ve)}")
            return jsonify({"error": f"Invalid input data: {str(ve)}"}), 400
        except ZeroDivisionError as zde:
            print(f"ZeroDivisionError: {str(zde)}")
            return jsonify({"error": "Division by zero occurred during backtesting. This may be due to insufficient capital or zero-priced stocks."}), 400
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(traceback.format_exc())  # This will print the full traceback
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return render_template('backtest.html')

def perform_backtest(start_date, end_date, initial_capital):
    print(f"Performing backtest from {start_date} to {end_date}")
    
    # Load historical data
    query = text("""
    SELECT fd.symbol, fd.date, fd.open, fd.high, fd.low, fd.close, fd.volume
    FROM finance_data fd
    WHERE fd.date BETWEEN :start_date AND :end_date
    ORDER BY fd.date, fd.symbol
    """)
    
    df = pd.read_sql(query, engine, params={'start_date': start_date, 'end_date': end_date})
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data fetched: {len(df)} rows")

    if df.empty:
        print("No data available for the given date range")
        return {
            'error': 'No data available for the given date range',
            'equity_curve': [],
            'trades': [],
            'metrics': {
                'total_return': 0,
                'total_return_percentage': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        }

    # Load trained models
    models = load_trained_models()
    print(f"Models loaded: {len(models)} models")

    # Initialize variables
    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = [{'date': start_date.strftime('%Y-%m-%d'), 'equity': capital}]
    
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() == 0:  # Monday
            # Get data up to the current date
            current_data = df[df['date'] <= current_date]
            print(f"Current data for {current_date}: {len(current_data)} rows")
            
            # Make predictions for all stocks
            predictions = make_predictions(current_data, models, current_date)
            print(f"Predictions made for {current_date}: {predictions}")
            
            # Rank stocks based on predictions
            ranked_stocks = rank_stocks(predictions)
            print(f"Ranked stocks for {current_date}: {ranked_stocks}")
            
            # Sell stocks that are no longer in top 12
            for symbol in list(positions.keys()):
                if symbol not in ranked_stocks[:12]:
                    sell_data = current_data[
                        (current_data['date'] == current_date) & 
                        (current_data['symbol'] == symbol)
                    ]
                    if not sell_data.empty:
                        sell_price = sell_data['close'].values[0]
                        profit = (sell_price - positions[symbol]['price']) * positions[symbol]['quantity']
                        capital += sell_price * positions[symbol]['quantity']
                        trades.append({
                            'symbol': symbol,
                            'entry_date': positions[symbol]['date'].strftime('%Y-%m-%d'),
                            'entry_price': positions[symbol]['price'],
                            'exit_date': current_date.strftime('%Y-%m-%d'),
                            'exit_price': sell_price,
                            'quantity': positions[symbol]['quantity'],
                            'profit_loss': profit
                        })
                        del positions[symbol]
                    else:
                        print(f"No sell data for {symbol} on {current_date}")
            
            # Buy top 10 stocks
            available_positions = 10 - len(positions)
            if available_positions > 0:
                available_capital = capital / available_positions
                for symbol in ranked_stocks[:10]:
                    if symbol not in positions:
                        buy_data = current_data[
                            (current_data['date'] == current_date) & 
                            (current_data['symbol'] == symbol)
                        ]
                        if not buy_data.empty:
                            buy_price = buy_data['close'].values[0]
                            if buy_price > 0:
                                quantity = int(available_capital / buy_price)
                                if quantity > 0:
                                    positions[symbol] = {
                                        'price': buy_price,
                                        'quantity': quantity,
                                        'date': current_date
                                    }
                                    capital -= buy_price * quantity
                            else:
                                print(f"Skipping {symbol} due to zero or negative price on {current_date}")
                        else:
                            print(f"No buy data for {symbol} on {current_date}")
            else:
                print(f"No available positions to buy on {current_date}")
        
        # Calculate current equity
        current_equity = capital
        for symbol, position in positions.items():
            current_price_data = df[
                (df['date'] == current_date) & 
                (df['symbol'] == symbol)
            ]
            if not current_price_data.empty:
                current_price = current_price_data['close'].values[0]
                current_equity += current_price * position['quantity']
            else:
                print(f"No current price data for {symbol} on {current_date}")
        
        equity_curve.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'equity': current_equity
        })
        
        current_date += timedelta(days=1)
    
    # Close all positions at the end
    for symbol, position in positions.items():
        end_price_data = df[
            (df['date'] == end_date) & 
            (df['symbol'] == symbol)
        ]
        if not end_price_data.empty:
            sell_price = end_price_data['close'].values[0]
            profit = (sell_price - position['price']) * position['quantity']
            trades.append({
                'symbol': symbol,
                'entry_date': position['date'].strftime('%Y-%m-%d'),
                'entry_price': position['price'],
                'exit_date': end_date.strftime('%Y-%m-%d'),
                'exit_price': sell_price,
                'quantity': position['quantity'],
                'profit_loss': profit
            })
        else:
            print(f"No end price data for {symbol} on {end_date}")
    
    # Calculate metrics
    total_return = equity_curve[-1]['equity'] - initial_capital
    total_return_percentage = (total_return / initial_capital) * 100
    max_drawdown = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': {
            'total_return': total_return,
            'total_return_percentage': total_return_percentage,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    }

def load_trained_models():
    models = {}
    for filename in os.listdir(training_folder):
        if filename.endswith('.cbm'):
            symbol = filename[:-4]
            model_path = os.path.join(training_folder, filename)
            models[symbol] = CatBoostRegressor().load_model(model_path)
    return models

def make_predictions(df, models, current_date):
    predictions = {}
    for symbol, model in models.items():
        symbol_data = df[(df['symbol'] == symbol) & (df['date'] <= current_date)].copy()
        if len(symbol_data) > 0:
            symbol_data = create_features(symbol_data)
            features = ['open', 'high', 'low', 'volume', 'returns', 'prev_day_returns', 'ema5', 'ema10', 'hl2', 'hlc3', 'rsi', 'atr']
            X = symbol_data[features].iloc[-1].values.reshape(1, -1)
            predictions[symbol] = model.predict(X)[0]
    return predictions

def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['prev_day_returns'] = df['returns'].shift(1)
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['hl2'] = (df['high'] + df['low']) / 2
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def rank_stocks(predictions):
    return sorted(predictions.keys(), key=lambda x: predictions[x], reverse=True)

def calculate_max_drawdown(equity_curve):
    equity_values = [point['equity'] for point in equity_curve]
    peak = equity_values[0]
    max_drawdown = 0
    
    for value in equity_values[1:]:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown * 100

def calculate_win_rate(trades):
    winning_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
    return (winning_trades / len(trades)) * 100 if trades else 0

def calculate_profit_factor(trades):
    gross_profit = sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0)
    gross_loss = abs(sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] < 0))
    return gross_profit / gross_loss if gross_loss != 0 else float('inf')