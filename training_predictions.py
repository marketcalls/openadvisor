import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sqlalchemy import create_engine, text
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set the path to the instance folder
instance_folder = os.path.join(current_directory, 'instance')

# Construct the absolute path to the database file within the instance folder
db_file_path = os.path.join(instance_folder, os.getenv('SQLALCHEMY_DATABASE_URI').replace('sqlite:///', ''))

# Ensure the training folder exists
training_folder = os.getenv('TRAINING_FOLDER', 'training')
os.makedirs(training_folder, exist_ok=True)

# Ensure the predictions folder exists
predictions_folder = os.getenv('PREDICTIONS_FOLDER', 'predictions')
os.makedirs(predictions_folder, exist_ok=True)

# SQLAlchemy setup
engine = create_engine(f'sqlite:///{db_file_path}', echo=False)

# Function to create features using pandas_ta
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

# Function to calculate RSI
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

# Function to calculate ATR
def calculate_atr(high, low, close, period=14):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def train_and_predict(symbol, days=30):
    # Load data from SQLite
    query = f"SELECT * FROM finance_data WHERE symbol = '{symbol}' ORDER BY date"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Create features
    df = create_features(df)
    
    # Prepare features and target
    features = ['open', 'high', 'low', 'volume', 'returns', 'prev_day_returns', 'ema5', 'ema10', 'hl2', 'hlc3', 'rsi', 'atr']
    X = df[features]
    y = df['close']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train CatBoost model
    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_state=42)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
    
    # Save the model
    model_path = os.path.join(training_folder, f'{symbol}.cbm')
    model.save_model(model_path)
    
    # Make predictions for the next 30 days
    last_known_date = df.index[-1]
    predictions = []
    last_data = df.iloc[-1].copy()
    prediction_data = df.iloc[-30:].copy()  # Get last 30 days for rolling calculations
    
    for i in range(days):
        next_day_prediction = model.predict(last_data[features].values.reshape(1, -1))[0]
        predictions.append(next_day_prediction)
        
        # Create a new row for the predicted day
        new_row = pd.DataFrame([{
            'open': last_data['close'],  # Assume next day's open is the same as previous close
            'high': next_day_prediction * 1.01,  # Assume 1% higher than predicted close
            'low': next_day_prediction * 0.99,   # Assume 1% lower than predicted close
            'close': next_day_prediction,
            'volume': last_data['volume']  # Keep the same volume (you might want to adjust this)
        }])
        
        # Add the new row to prediction_data
        prediction_data = pd.concat([prediction_data, new_row], ignore_index=True)
        
        # Recalculate features
        prediction_data['returns'] = prediction_data['close'].pct_change()
        prediction_data['prev_day_returns'] = prediction_data['returns'].shift(1)
        prediction_data['ema5'] = ta.ema(prediction_data['close'], length=5)
        prediction_data['ema10'] = ta.ema(prediction_data['close'], length=10)
        prediction_data['hl2'] = (prediction_data['high'] + prediction_data['low']) / 2
        prediction_data['hlc3'] = (prediction_data['high'] + prediction_data['low'] + prediction_data['close']) / 3
        prediction_data['rsi'] = calculate_rsi(prediction_data['close'])
        prediction_data['atr'] = calculate_atr(prediction_data['high'], prediction_data['low'], prediction_data['close'])
        
        # Update last_data with the latest row
        last_data = prediction_data.iloc[-1]
    
    # Calculate metrics on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create a DataFrame with dates and predictions
    prediction_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=days)
    predictions_df = pd.DataFrame({'date': prediction_dates, 'predicted_close': predictions})
    predictions_df.set_index('date', inplace=True)
    
    return predictions_df, mse, r2, model

def main():
    # Get all unique symbols from the database
    query = "SELECT DISTINCT symbol FROM finance_data"
    symbols = pd.read_sql(query, engine)['symbol'].tolist()

    # Train models and make predictions for all symbols
    results = {}
    for symbol in tqdm(symbols, desc="Training models"):
        try:
            predictions_df, mse, r2, model = train_and_predict(symbol)
            results[symbol] = {
                'predictions': predictions_df,
                'mse': mse,
                'r2': r2,
                'model': model
            }
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

    # Print results
    for symbol, data in results.items():
        print(f"\nSymbol: {symbol}")
        print(f"Mean Squared Error: {data['mse']:.4f}")
        print(f"R-squared Score: {data['r2']:.4f}")
        print("Predictions for the next 30 days:")
        print(data['predictions'])

    # Save results to a CSV file
    all_predictions = pd.DataFrame()
    for symbol, data in results.items():
        symbol_predictions = data['predictions'].copy()
        symbol_predictions['symbol'] = symbol
        all_predictions = pd.concat([all_predictions, symbol_predictions])

    portfolio_predictions_path = os.path.join(predictions_folder, 'portfolio_30day_predictions.csv')
    all_predictions.to_csv(portfolio_predictions_path)
    print(f"Predictions saved to '{portfolio_predictions_path}'")

    # Load the predictions
    predictions_df = pd.read_csv(portfolio_predictions_path)

    # Function to get the last known closing price for a symbol
    def get_last_close(symbol):
        query = text(f"SELECT close FROM finance_data WHERE symbol = :symbol ORDER BY date DESC LIMIT 1")
        with engine.connect() as connection:
            result = connection.execute(query, {"symbol": symbol}).fetchone()
        return result[0] if result else None

    # Create a new DataFrame to store the aggregate results
    aggregate_results = []

    # Process each unique symbol
    for symbol in predictions_df['symbol'].unique():
        symbol_predictions = predictions_df[predictions_df['symbol'] == symbol]
        
        # Get the last predicted closing price (30th day)
        closing_price_30day = symbol_predictions['predicted_close'].iloc[-1]
        
        # Get the last known closing price
        last_known_close = get_last_close(symbol)
        
        if last_known_close is not None:
            # Calculate 30-day returns
            returns_30day = (closing_price_30day - last_known_close) / last_known_close * 100
        else:
            returns_30day = None
        
        # Append to results
        aggregate_results.append({
            'Stock': symbol,
            '30 day Closing': round(closing_price_30day, 2),
            '30 day returns': round(returns_30day, 2) if returns_30day is not None else None
        })

    # Create DataFrame from aggregate results
    aggregate_df = pd.DataFrame(aggregate_results)

    # Sort by 30-day returns in descending order
    aggregate_df = aggregate_df.sort_values('30 day returns', ascending=False)

    aggregate_predictions_path = os.path.join(predictions_folder, 'aggregate_30day_predictions.csv')
    aggregate_df.to_csv(aggregate_predictions_path, index=False)
    print(f"Aggregate results saved to '{aggregate_predictions_path}'")

    # Save to SQLite database
    aggregate_df.to_sql('aggregate_predictions', engine, if_exists='replace', index=False)
    print("Aggregate results saved to 'aggregate_predictions' table in the database")

    # Optional: Query and display the table from the database to confirm
    print("\nConfirming data in the database:")
    query_result = pd.read_sql_query("SELECT * FROM aggregate_predictions LIMIT 10", engine)
    print(query_result)

if __name__ == '__main__':
    main()
