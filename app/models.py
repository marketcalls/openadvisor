from app import db

class Symbol(db.Model):
    __tablename__ = 'symbols'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String, nullable=False)
    yahoo_symbol = db.Column(db.String, unique=True, nullable=False)

class FinanceData(db.Model):
    __tablename__ = 'finance_data'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String, nullable=False)
    date = db.Column(db.Date, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String, nullable=False)
    date = db.Column(db.Date, nullable=False)
    predicted_close = db.Column(db.Float, nullable=False)