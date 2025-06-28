import os
import pandas as pd  # Import pandas
from sqlalchemy import create_engine

# Correct the path to your database file
db_path = os.path.join(os.getcwd(), 'instance', 'logistics.db')
engine = create_engine(f'sqlite:///{db_path}')

# Load data from the correct path
df = pd.read_sql('SELECT * FROM "order"', engine)

# Show the first few rows to confirm it's working
print(df.head())
