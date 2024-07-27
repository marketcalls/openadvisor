# %%
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
import yfinance as yf

# %%
# Read the CSV file
csv_file_path = 'NIFTY_200_with_Yahoo_Symbols.csv'
df = pd.read_csv(csv_file_path)
df

# %%
# SQLAlchemy setup
db_file_path = 'sqlite:///finance_symbols.db'
engine = create_engine(db_file_path, echo=False)
Base = declarative_base()

# %%
# Define the Symbols table
class Symbol(Base):
    __tablename__ = 'symbols'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    yahoo_symbol = Column(String, unique=True)
    __table_args__ = (UniqueConstraint('yahoo_symbol', name='_yahoo_symbol_uc'),)

# Define the Data table
class FinanceData(Base):
    __tablename__ = 'finance_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    date = Column(Date)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'date', name='_symbol_date_uc'),)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# %%
# Insert or update symbols in the database
inserted_symbols = 0
updated_symbols = 0
for index, row in df.iterrows():
    name = row['SYMBOL']
    yahoo_symbol = row['YAHOO_SYMBOL']
    
    symbol = session.query(Symbol).filter_by(yahoo_symbol=yahoo_symbol).first()
    
    if symbol:
        symbol.name = name
        updated_symbols += 1
    else:
        new_symbol = Symbol(name=name, yahoo_symbol=yahoo_symbol)
        session.add(new_symbol)
        inserted_symbols += 1

session.commit()
print(f'Inserted {inserted_symbols} new symbols.')
print(f'Updated {updated_symbols} existing symbols.')

# %%
# Fetch and insert/update yfinance data
inserted_data = 0
updated_data = 0
failed_downloads = []

for yahoo_symbol in df['YAHOO_SYMBOL']:
    try:
        data = yf.download(yahoo_symbol, start='1994-01-01', end='2024-01-01')
        for date, row in data.iterrows():
            finance_data = session.query(FinanceData).filter_by(symbol=yahoo_symbol, date=date).first()
            if finance_data:
                finance_data.open = row['Open']
                finance_data.high = row['High']
                finance_data.low = row['Low']
                finance_data.close = row['Close']
                finance_data.volume = row['Volume']
                updated_data += 1
            else:
                new_finance_data = FinanceData(
                    symbol=yahoo_symbol,
                    date=date,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=row['Volume']
                )
                session.add(new_finance_data)
                inserted_data += 1
        print(f'Successfully downloaded data for {yahoo_symbol}')
    except Exception as e:
        print(f'Failed to download data for {yahoo_symbol}: {e}')
        failed_downloads.append(yahoo_symbol)

session.commit()

# Print final status
print(f'Inserted {inserted_data} new data entries.')
print(f'Updated {updated_data} existing data entries.')

# %%


# %%
# Print final status
print(f'Inserted {inserted_data} new data entries.')
print(f'Updated {updated_data} existing data entries.')
if failed_downloads:
    print('Failed to download data for the following symbols:')
    for symbol in failed_downloads:
        print(symbol)

session.close()


# %%
# Retrieve and print the list of symbols from the database
symbols_in_db = session.query(Symbol).all()
print("List of symbols from the database:")
for symbol in symbols_in_db:
    print(f"ID: {symbol.id}, Name: {symbol.name}, Yahoo Symbol: {symbol.yahoo_symbol}")

session.close()

# %%
from sqlalchemy import func

# SQLAlchemy setup
db_file_path = 'sqlite:///finance_symbols.db'
engine = create_engine(db_file_path, echo=False)
Base = declarative_base()

# Define the FinanceData table
class FinanceData(Base):
    __tablename__ = 'finance_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    date = Column(Date)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'date', name='_symbol_date_uc'),)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Query to get unique symbols and their count
unique_symbols = session.query(FinanceData.symbol).distinct().all()
unique_symbols_count = len(unique_symbols)

# Print the total number of symbols and the list of symbols
print(f"Total Symbols are: {unique_symbols_count}\n")
print("The List of Symbols are:")
for idx, (symbol,) in enumerate(unique_symbols, start=1):
    print(f"{idx}. {symbol}")

session.close()


# %%
# Query to get data for the symbol "RELIANCE.NS"
symbol_to_plot = "RELIANCE.NS"
data = session.query(FinanceData).filter_by(symbol=symbol_to_plot).all()

# Convert the queried data to a pandas DataFrame
data_df = pd.DataFrame([(d.date, d.open, d.high, d.low, d.close, d.volume) for d in data],
                       columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Set the date as the index
data_df.set_index('Date', inplace=True)

# Plot the data using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data_df.index,
    y=data_df['Close'],
    mode='lines',
    name='Close Price'
))

fig.update_layout(
    title=f'Close Price of {symbol_to_plot} Over Time',
    xaxis_title='Date',
    yaxis_title='Close Price',
    template='plotly_dark'
)

fig.show()

session.close()

# %%
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, UniqueConstraint, func
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLAlchemy setup
db_file_path = 'sqlite:///finance_symbols.db'
engine = create_engine(db_file_path, echo=False)
Base = declarative_base()

# Define the FinanceData table
class FinanceData(Base):
    __tablename__ = 'finance_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    date = Column(Date)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'date', name='_symbol_date_uc'),)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Query to get the count of records for each symbol
symbol_counts = session.query(FinanceData.symbol, func.count(FinanceData.id)).group_by(FinanceData.symbol).all()

# Print the count of records for each symbol
print("Number of records for each symbol:")
for symbol, count in symbol_counts:
    print(f"Symbol: {symbol}, Count: {count}")

session.close()


# %%
import pandas as pd
import numpy as np
import pandas_ta as ta
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# SQLAlchemy setup
db_file_path = 'sqlite:///finance_symbols.db'
engine = create_engine(db_file_path, echo=False)

# %%
# Function to create features using pandas_ta
def create_features(df):
    # Calculate returns and previous day returns
    df['returns'] = df['close'].pct_change()
    df['prev_day_returns'] = df['returns'].shift(1)
    
    # Calculate EMAs
    df['ema5'] = ta.ema(df['close'], length=5)
    df['ema10'] = ta.ema(df['close'], length=10)
    
    # Calculate high+low/2 and high+low+close/3
    df['hl2'] = (df['high'] + df['low']) / 2
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # Calculate ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df


# %%
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

# %%
import pandas as pd
import numpy as np
import pandas_ta as ta
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ... (keep all the previous functions and setup code) ...

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

# Get list of unique symbols
symbols_query = "SELECT DISTINCT symbol FROM finance_data"
symbols = pd.read_sql(symbols_query, engine)['symbol'].tolist()


# %%
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

# %%
# Plot predictions for a sample symbol (e.g., the first one)
sample_symbol = list(results.keys())[0]
sample_predictions = results[sample_symbol]['predictions']

plt.figure(figsize=(12, 6))
plt.plot(sample_predictions.index, sample_predictions['predicted_close'])
plt.title(f"30-Day Price Predictions for {sample_symbol}")
plt.xlabel("Date")
plt.ylabel("Predicted Close Price")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot feature importance for the sample symbol
sample_model = results[sample_symbol]['model']
feature_importance = sample_model.feature_importances_
feature_names = sample_model.feature_names_

plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance)
plt.title(f"Feature Importance for {sample_symbol}")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Save results to a CSV file
all_predictions = pd.DataFrame()
for symbol, data in results.items():
    symbol_predictions = data['predictions'].copy()
    symbol_predictions['symbol'] = symbol
    all_predictions = pd.concat([all_predictions, symbol_predictions])

all_predictions.to_csv('nifty200_30day_predictions.csv')
print("Predictions saved to 'nifty200_30day_predictions.csv'")

# %%
import pandas as pd
from sqlalchemy import create_engine, text

# Load the predictions
predictions_df = pd.read_csv('nifty200_30day_predictions.csv')

# Connect to the SQLite database
db_file_path = 'sqlite:///finance_symbols.db'
engine = create_engine(db_file_path, echo=False)

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

aggregate_df

# %%
# Plot feature importance for a sample symbol (e.g., the first one)
sample_symbol = list(results.keys())[0]
sample_model = results[sample_symbol]['model']
feature_importance = sample_model.feature_importances_
feature_names = sample_model.feature_names_

plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance)
plt.title(f"Feature Importance for {sample_symbol}")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Save to CSV
aggregate_df.to_csv('aggregate_30day_predictions.csv', index=False)
print("Aggregate results saved to 'aggregate_30day_predictions.csv'")

# Save to SQLite database
aggregate_df.to_sql('aggregate_predictions', engine, if_exists='replace', index=False)
print("Aggregate results saved to 'aggregate_predictions' table in the database")

# Optional: Query and display the table from the database to confirm
print("\nConfirming data in the database:")
query_result = pd.read_sql_query("SELECT * FROM aggregate_predictions LIMIT 10", engine)
print(query_result)


