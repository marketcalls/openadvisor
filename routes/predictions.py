from flask import Blueprint, render_template, request
from extensions import db
import pandas as pd

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predictions', methods=['GET'])
def predictions():
    filter_option = request.args.get('filter', 'TOP 10')
    limit = {
        'TOP 10': 10,
        'TOP 20': 20,
        'TOP 50': 50,
        'TOP 100': 100,
        'ALL': None
    }.get(filter_option, 10)

    query = "SELECT * FROM aggregate_predictions ORDER BY `30 day returns` DESC"
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, db.engine)
    predictions = df.to_dict(orient='records')

    return render_template('predictions.html', predictions=predictions, filter_option=filter_option)
