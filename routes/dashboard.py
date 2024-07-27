from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required
from extensions import db
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


