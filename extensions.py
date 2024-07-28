from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_bcrypt import Bcrypt
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
import os
import atexit
from data_downloader import update_database  # Import your scheduler function
from training_predictions import train_all_models, make_all_predictions  # Import your training and prediction functions

load_dotenv()

db = SQLAlchemy()
mail = Mail()
login_manager = LoginManager()
limiter = Limiter(key_func=get_remote_address)
bcrypt = Bcrypt()
scheduler = BackgroundScheduler()

def start_scheduler():
    if scheduler.state == 1:  # 1 means running
        return

    # General intervals
    interval = int(os.getenv('SCHEDULER_INTERVAL', 86400))
    test_interval = int(os.getenv('SCHEDULER_TEST_INTERVAL', 300))
    use_test_interval = os.getenv('USE_TEST_INTERVAL', 'False').lower() in ['true', '1', 't']

    # Training intervals
    training_interval = int(os.getenv('TRAINING_INTERVAL', 2592000))  # Default to 30 days
    training_test_interval = int(os.getenv('TRAINING_TEST_INTERVAL', 300))
    use_training_test_interval = os.getenv('USE_TRAINING_TEST_INTERVAL', 'False').lower() in ['true', '1', 't']

    # Prediction intervals
    prediction_interval = int(os.getenv('PREDICTION_INTERVAL', 604800))  # Default to 7 days
    prediction_test_interval = int(os.getenv('PREDICTION_TEST_INTERVAL', 420))
    use_prediction_test_interval = os.getenv('USE_PREDICTION_TEST_INTERVAL', 'False').lower() in ['true', '1', 't']

    # Update database job
    scheduler.add_job(
        func=lambda: update_database(os.getenv('PORTFOLIO_CSV')),
        trigger=IntervalTrigger(seconds=test_interval if use_test_interval else interval),
        id='update_database_job',
        name='Update database with financial data',
        replace_existing=True
    )

    # Train models job
    scheduler.add_job(
        func=train_all_models,
        trigger=IntervalTrigger(seconds=training_test_interval if use_training_test_interval else training_interval),
        id='train_models_job',
        name='Train models with financial data',
        replace_existing=True
    )

    # Make predictions job
    scheduler.add_job(
        func=make_all_predictions,
        trigger=IntervalTrigger(seconds=prediction_test_interval if use_prediction_test_interval else prediction_interval),
        id='make_predictions_job',
        name='Make predictions with trained models',
        replace_existing=True
    )

    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

start_scheduler()
