from app import scheduler, db
from app.models import Symbol, Prediction
from app.utils import download_and_store_data, train_and_predict, generate_aggregate_predictions

@scheduler.task('cron', id='download_data', minute='*/15')
def scheduled_data_download():
    with scheduler.app.app_context():
        symbols = Symbol.query.all()
        for symbol in symbols:
            download_and_store_data(symbol.yahoo_symbol)

@scheduler.task('cron', id='make_predictions', minute='*/15')
def scheduled_predictions():
    with scheduler.app.app_context():
        symbols = Symbol.query.all()
        for symbol in symbols:
            predictions_df, _, _, _ = train_and_predict(symbol.yahoo_symbol)
            for date, row in predictions_df.iterrows():
                prediction = Prediction(
                    symbol=symbol.yahoo_symbol,
                    date=date.date(),
                    predicted_close=row['predicted_close']
                )
                db.session.add(prediction)
        db.session.commit()

@scheduler.task('cron', id='generate_aggregate', minute='0')
def scheduled_aggregate_predictions():
    with scheduler.app.app_context():
        aggregate_df = generate_aggregate_predictions()
        aggregate_df.to_sql('aggregate_predictions', db.engine, if_exists='replace', index=False)