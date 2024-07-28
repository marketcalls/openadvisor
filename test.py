import os
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the database URI from the environment variables
db_file_path = os.getenv('SQLALCHEMY_DATABASE_URI')

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set the path to the instance folder
instance_folder = os.path.join(current_directory, 'instance')

# Construct the absolute path to the database file within the instance folder
db_file_path = os.path.join(instance_folder, db_file_path.replace('sqlite:///', ''))

# Create the SQLAlchemy engine
engine = create_engine(f'sqlite:///{db_file_path}', echo=False)

# Function to list all tables in the database
def list_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return tables

if __name__ == '__main__':
    tables = list_tables()
    print("Tables in the database:")
    for table in tables:
        print(table)
