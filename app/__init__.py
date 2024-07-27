from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_apscheduler import APScheduler
import pandas as pd

db = SQLAlchemy()
scheduler = APScheduler()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)
    scheduler.init_app(app)

    with app.app_context():
        from app import models  # Import models here to avoid circular imports
        db.create_all()  # This line creates the database tables
        initialize_symbols()  # This function will populate the symbols table

        # Set up scheduler job
        @scheduler.task('cron', id='download_data', hour=0)  # Run daily at midnight
        def scheduled_data_download():
            from app.models import Symbol
            from app.utils import download_and_store_data
            symbols = Symbol.query.all()
            for symbol in symbols:
                download_and_store_data(symbol.yahoo_symbol)
            print("Scheduled data download completed.")

    scheduler.start()

    from app import routes
    app.register_blueprint(routes.main)

    return app

def initialize_symbols():
    from app.models import Symbol  # Import here to avoid circular imports
    
    # Check if symbols table is empty
    if db.session.query(Symbol).first() is None:
        # Read symbols from CSV file
        csv_file_path = 'NIFTY_200_with_Yahoo_Symbols.csv'
        df = pd.read_csv(csv_file_path)
        
        # Add symbols to the database
        for _, row in df.iterrows():
            symbol = Symbol(name=row['SYMBOL'], yahoo_symbol=row['YAHOO_SYMBOL'])
            db.session.add(symbol)
        
        db.session.commit()
        print("Symbols initialized successfully.")
    else:
        print("Symbols table already populated.")