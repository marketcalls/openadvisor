import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sqlalchemy import create_engine, text
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from app.models import Symbol, FinanceData, Prediction
from app import db
import yfinance as yf
from app.models import FinanceData

# SQLAlchemy setup
db_file_path = 'sqlite:///finance_symbols.db'
engine = create_engine(db_file_path, echo=False)


def download_and_store_data(symbol):
    data = yf.download(symbol, start='1994-01-01')
    for date, row in data.iterrows():
        existing_data = FinanceData.query.filter_by(symbol=symbol, date=date.date()).first()
        if existing_data:
            # Update existing data
            existing_data.open = row['Open']
            existing_data.high = row['High']
            existing_data.low = row['Low']
            existing_data.close = row['Close']
            existing_data.volume = row['Volume']
        else:
            # Add new data
            new_data = FinanceData(
                symbol=symbol,
                date=date.date(),
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume']
            )
            db.session.add(new_data)
    db.session.commit()
    print(f"Data for {symbol} downloaded and stored/updated in the database.")

def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['prev_day_returns'] = df['returns'].shift(1)
    df['ema5'] = ta.ema(df['close'], length=5)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['hl2'] = (df['high'] + df['low']) / 2
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df.dropna(inplace=True)
    return df

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def calculate_atr(high, low, close, period=14):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def train_and_predict(symbol, days=30):
    query = f"SELECT * FROM finance_data WHERE symbol = '{symbol}' ORDER BY date"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    df = create_features(df)
    
    features = ['open', 'high', 'low', 'volume', 'returns', 'prev_day_returns', 'ema5', 'ema10', 'hl2', 'hlc3', 'rsi', 'atr']
    X = df[features]
    y = df['close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_state=42)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
    
    last_known_date = df.index[-1]
    predictions = []
    last_data = df.iloc[-1].copy()
    prediction_data = df.iloc[-30:].copy()
    
    for i in range(days):
        next_day_prediction = model.predict(last_data[features].values.reshape(1, -1))[0]
        predictions.append(next_day_prediction)
        
        new_row = pd.DataFrame([{
            'open': last_data['close'],
            'high': next_day_prediction * 1.01,
            'low': next_day_prediction * 0.99,
            'close': next_day_prediction,
            'volume': last_data['volume']
        }])
        
        prediction_data = pd.concat([prediction_data, new_row], ignore_index=True)
        
        prediction_data['returns'] = prediction_data['close'].pct_change()
        prediction_data['prev_day_returns'] = prediction_data['returns'].shift(1)
        prediction_data['ema5'] = ta.ema(prediction_data['close'], length=5)
        prediction_data['ema10'] = ta.ema(prediction_data['close'], length=10)
        prediction_data['hl2'] = (prediction_data['high'] + prediction_data['low']) / 2
        prediction_data['hlc3'] = (prediction_data['high'] + prediction_data['low'] + prediction_data['close']) / 3
        prediction_data['rsi'] = calculate_rsi(prediction_data['close'])
        prediction_data['atr'] = calculate_atr(prediction_data['high'], prediction_data['low'], prediction_data['close'])
        
        last_data = prediction_data.iloc[-1]
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    prediction_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=days)
    predictions_df = pd.DataFrame({'date': prediction_dates, 'predicted_close': predictions})
    predictions_df.set_index('date', inplace=True)
    
    return predictions_df, mse, r2, model

def get_last_close(symbol):
    query = text(f"SELECT close FROM finance_data WHERE symbol = :symbol ORDER BY date DESC LIMIT 1")
    with engine.connect() as connection:
        result = connection.execute(query, {"symbol": symbol}).fetchone()
    return result[0] if result else None

def generate_aggregate_predictions():
    symbols = Symbol.query.all()
    aggregate_results = []

    for symbol in symbols:
        predictions_df, mse, r2, _ = train_and_predict(symbol.yahoo_symbol)
        closing_price_30day = predictions_df['predicted_close'].iloc[-1]
        last_known_close = get_last_close(symbol.yahoo_symbol)
        
        if last_known_close is not None:
            returns_30day = (closing_price_30day - last_known_close) / last_known_close * 100
        else:
            returns_30day = None
        
        aggregate_results.append({
            'Stock': symbol.yahoo_symbol,
            '30 day Closing': round(closing_price_30day, 2),
            '30 day returns': round(returns_30day, 2) if returns_30day is not None else None
        })

    aggregate_df = pd.DataFrame(aggregate_results)
    aggregate_df = aggregate_df.sort_values('30 day returns', ascending=False)
    
    return aggregate_df