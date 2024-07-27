from flask import Blueprint, render_template, jsonify, request, Response
from concurrent.futures import ThreadPoolExecutor
from app.models import Symbol, FinanceData, Prediction
from app.utils import download_and_store_data, train_and_predict, generate_aggregate_predictions
from app import db
import json

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/control')
def control_panel():
    symbols = Symbol.query.all()
    return render_template('control.html', symbols=symbols)

@main.route('/download_data', methods=['POST'])
def download_data():
    symbol = request.form.get('symbol')
    if symbol:
        download_and_store_data(symbol)
        return jsonify({"message": f"Data downloaded for {symbol}"}), 200
    return jsonify({"error": "No symbol provided"}), 400

@main.route('/download_all_data', methods=['POST'])
def download_all_data():
    def generate():
        symbols = Symbol.query.all()
        total = len(symbols)
        for index, symbol in enumerate(symbols, 1):
            download_and_store_data(symbol.yahoo_symbol)
            progress = (index / total) * 100
            yield f"data: {json.dumps({'symbol': symbol.yahoo_symbol, 'progress': progress})}\n\n"
        yield f"data: {json.dumps({'message': 'All data downloaded successfully'})}\n\n"
    return Response(generate(), content_type='text/event-stream')


@main.route('/train_all_models', methods=['POST'])
def train_all_models():
    symbols = Symbol.query.all()
    total = len(symbols)
    completed = 0

    def train_with_progress(symbol):
        nonlocal completed
        predictions_df, mse, r2, _ = train_and_predict(symbol.yahoo_symbol)
        completed += 1
        return {
            'symbol': symbol.yahoo_symbol,
            'mse': mse,
            'r2': r2,
            'progress': (completed / total) * 100
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(train_with_progress, symbols))

    return jsonify(results)

@main.route('/train_model', methods=['POST'])
def train_model():
    symbol = request.form.get('symbol')
    if symbol:
        predictions_df, mse, r2, _ = train_and_predict(symbol)
        return jsonify({
            "message": f"Model trained for {symbol}",
            "mse": mse,
            "r2": r2
        }), 200
    return jsonify({"error": "No symbol provided"}), 400

@main.route('/predictions')
def predictions():
    aggregate_predictions = generate_aggregate_predictions()
    return render_template('predictions.html', predictions=aggregate_predictions.to_dict(orient='records'))

@main.route('/api/predictions')
def get_predictions():
    predictions = Prediction.query.order_by(Prediction.date.desc()).limit(100).all()
    return jsonify([{
        'symbol': p.symbol,
        'date': p.date.isoformat(),
        'predicted_close': p.predicted_close
    } for p in predictions])

@main.route('/api/portfolio')
def get_portfolio():
    # Fetch 10 random symbols from the database
    symbols = Symbol.query.order_by(db.func.random()).limit(10).all()
    
    portfolio = []
    for symbol in symbols:
        # Get the latest data for this symbol
        latest_data = FinanceData.query.filter_by(symbol=symbol.yahoo_symbol).order_by(FinanceData.date.desc()).first()
        if latest_data:
            portfolio.append({
                'symbol': symbol.yahoo_symbol,
                'name': symbol.name,
                'quantity': 10,  # You can adjust this or make it random
                'current_price': latest_data.close
            })
    
    return jsonify(portfolio)

@main.route('/api/chart/<symbol>')
def get_chart_data(symbol):
    data = FinanceData.query.filter_by(symbol=symbol).order_by(FinanceData.date).all()
    return jsonify([{
        'date': d.date.isoformat(),
        'close': d.close
    } for d in data])

@main.route('/api/aggregate_predictions')
def get_aggregate_predictions():
    aggregate_predictions = generate_aggregate_predictions()
    return jsonify(aggregate_predictions.to_dict(orient='records'))

@main.route('/api/symbols')
def get_symbols():
    symbols = Symbol.query.all()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'yahoo_symbol': s.yahoo_symbol
    } for s in symbols])

@main.route('/check_data/<symbol>')
def check_data(symbol):
    data = FinanceData.query.filter_by(symbol=symbol).order_by(FinanceData.date.desc()).limit(10).all()
    return jsonify([{
        'date': d.date.isoformat(),
        'close': d.close
    } for d in data])