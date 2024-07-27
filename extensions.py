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
from scheduler import update_database  # Import your scheduler function

load_dotenv()

db = SQLAlchemy()
mail = Mail()
login_manager = LoginManager()
limiter = Limiter(key_func=get_remote_address)
bcrypt = Bcrypt()
scheduler = BackgroundScheduler()

def start_scheduler():
    interval = int(os.getenv('SCHEDULER_INTERVAL', 86400))
    test_interval = int(os.getenv('SCHEDULER_TEST_INTERVAL', 300))
    use_test_interval = os.getenv('USE_TEST_INTERVAL', 'False').lower() in ['true', '1', 't']

    scheduler.add_job(
        func=lambda: update_database(os.getenv('PORTFOLIO_CSV')),
        trigger=IntervalTrigger(seconds=test_interval if use_test_interval else interval),
        id='update_database_job',
        name='Update database with financial data',
        replace_existing=True
    )
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

