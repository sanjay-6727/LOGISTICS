import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import create_engine
import os
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the path to the database
db_path = os.path.join(os.getcwd(), 'instance', 'logistics.db')
engine = create_engine(f'sqlite:///{db_path}')

# Load data from the SQLite database
try:
    df = pd.read_sql('SELECT * FROM "order"', engine)
    logger.info(f"Data loaded successfully from database: {len(df)} rows")
except Exception as e:
    logger.error(f"Error loading data from database: {e}")
    exit(1)

# Print column names for debugging
logger.info(f"Columns in the data: {list(df.columns)}")

# Clean and select relevant columns
required_columns = ['pickup_location', 'delivery_location', 'urgency', 'weight', 'volume', 'preferred_time', 'estimated_travel_time']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    logger.error(f"Missing required columns: {missing_columns}")
    exit(1)

df_cleaned = df[required_columns].copy()
logger.info(f"Selected required columns: {len(df_cleaned)} rows")

# Drop rows with missing or invalid data
df_cleaned = df_cleaned.dropna(subset=required_columns)
logger.info(f"After dropping NA: {len(df_cleaned)} rows")
df_cleaned = df_cleaned[
    (df_cleaned['weight'] > 0) & 
    (df_cleaned['volume'] > 0) & 
    (df_cleaned['estimated_travel_time'] > 0)
]
logger.info(f"After filtering invalid values: {len(df_cleaned)} rows")

# Check if there's enough data
if len(df_cleaned) < 10:
    logger.error(f"Insufficient data for training: {len(df_cleaned)} samples")
    exit(1)

# Log unique urgency values for debugging
logger.info(f"Unique urgency values in data: {df_cleaned['urgency'].unique()}")

# Standardize urgency values
urgency_mapping = {
    'High': 'high',
    'Medium': 'medium',
    'Low': 'low',
    'high': 'high',
    'medium': 'medium',
    'low': 'low'
}
df_cleaned['urgency'] = df_cleaned['urgency'].apply(
    lambda x: urgency_mapping.get(x, x.lower()) if isinstance(x, str) else 'medium'  # Fallback to 'medium' for invalid values
)
logger.info(f"Unique urgency values after standardization: {df_cleaned['urgency'].unique()}")

# Validate urgency values
valid_urgencies = {'high', 'medium', 'low'}
invalid_urgencies = set(df_cleaned['urgency'].unique()) - valid_urgencies
if invalid_urgencies:
    logger.warning(f"Unexpected urgency values found: {invalid_urgencies}. Replacing with 'medium'")
    df_cleaned['urgency'] = df_cleaned['urgency'].apply(lambda x: x if x in valid_urgencies else 'medium')
    logger.info(f"Unique urgency values after validation: {df_cleaned['urgency'].unique()}")

# Feature engineering
try:
    df_cleaned['preferred_time'] = pd.to_datetime(df_cleaned['preferred_time'])
    df_cleaned['hour'] = df_cleaned['preferred_time'].dt.hour
    df_cleaned['day_of_week'] = df_cleaned['preferred_time'].dt.dayofweek
    df_cleaned['is_weekend'] = df_cleaned['preferred_time'].dt.dayofweek.isin([5, 6]).astype(int)
except Exception as e:
    logger.error(f"Error parsing preferred_time: {e}")
    exit(1)

# Define location pair distances (copied from logistics_app.py for consistency)
location_pair_distances = {
    ('pallikaranai', 'delhi'): 10,
    ('pallikaranai', 'tambaram'): 5,
    ('pallikaranai', 'warehouse'): 5,
    ('pallikaranai', 'city_center'): 12,
    ('pallikaranai', 'chennai'): 0,
    ('pallikaranai', 'mumbai'): 15,
    ('tambaram', 'delhi'): 10,
    ('tambaram', 'pallikaranai'): 5,
    ('tambaram', 'warehouse'): 0,
    ('tambaram', 'city_center'): 10,
    ('tambaram', 'chennai'): 5,
    ('tambaram', 'mumbai'): 12,
    ('delhi', 'pallikaranai'): 10,
    ('delhi', 'tambaram'): 10,
    ('delhi', 'warehouse'): 10,
    ('delhi', 'city_center'): 3,
    ('delhi', 'chennai'): 10,
    ('delhi', 'mumbai'): 3,
    ('warehouse', 'pallikaranai'): 5,
    ('warehouse', 'tambaram'): 0,
    ('warehouse', 'delhi'): 10,
    ('warehouse', 'city_center'): 10,
    ('warehouse', 'chennai'): 5,
    ('warehouse', 'mumbai'): 12,
    ('city_center', 'pallikaranai'): 12,
    ('city_center', 'tambaram'): 10,
    ('city_center', 'delhi'): 3,
    ('city_center', 'warehouse'): 10,
    ('city_center', 'chennai'): 12,
    ('city_center', 'mumbai'): 0,
    ('chennai', 'pallikaranai'): 0,
    ('chennai', 'tambaram'): 5,
    ('chennai', 'delhi'): 10,
    ('chennai', 'warehouse'): 5,
    ('chennai', 'city_center'): 12,
    ('chennai', 'mumbai'): 15,
    ('mumbai', 'pallikaranai'): 15,
    ('mumbai', 'tambaram'): 12,
    ('mumbai', 'delhi'): 3,
    ('mumbai', 'warehouse'): 12,
    ('mumbai', 'chennai'): 15,
    ('mumbai', 'city_center'): 0,
}

# Calculate approx_distance
df_cleaned['approx_distance'] = df_cleaned.apply(
    lambda row: location_pair_distances.get(
        (row['pickup_location'].lower(), row['delivery_location'].lower()), 10
    ), 
    axis=1
)

# Validate distances
df_cleaned = df_cleaned[df_cleaned['approx_distance'] > 0]
logger.info(f"After filtering zero/negative distances: {len(df_cleaned)} rows")

# Check if there's enough data after distance validation
if len(df_cleaned) < 10:
    logger.error(f"Insufficient data after distance validation: {len(df_cleaned)} samples")
    exit(1)

# Encode categorical features
label_encoders = {}
categorical_cols = ['pickup_location', 'delivery_location', 'urgency']
for col in categorical_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col].str.lower())
    label_encoders[col] = le
    logger.info(f"Encoded {col} with classes: {le.classes_}")

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['weight', 'volume', 'approx_distance']
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

# Features (X) and target (y)
feature_cols = categorical_cols + numerical_cols + ['hour', 'day_of_week', 'is_weekend']
X = df_cleaned[feature_cols]
y = df_cleaned['estimated_travel_time']

# Split data for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")

# Define model and hyperparameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
logger.info(f"Best hyperparameters: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logger.info(f"Validation MAE: {mae:.2f} hours, R²: {r2:.2f}")

# Train final model on all data
final_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X, y)

# Save model, encoders, and scaler
models_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(final_model, os.path.join(models_dir, 'eta_model.pkl'))
joblib.dump(label_encoders, os.path.join(models_dir, 'eta_encoders.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'eta_scaler.pkl'))
joblib.dump(feature_cols, os.path.join(models_dir, 'eta_feature_cols.pkl'))

logger.info(f"Model and artifacts saved successfully in {models_dir}")
logger.info(f"Number of training samples: {len(df_cleaned)}")
logger.info(f"Feature importance: {dict(zip(feature_cols, final_model.feature_importances_))}")

# Optional: Save model performance report
report = {
    'training_samples': len(df_cleaned),
    'validation_mae': mae,
    'validation_r2': r2,
    'best_params': grid_search.best_params_,
    'feature_importance': dict(zip(feature_cols, final_model.feature_importances_)),
    'urgency_classes': label_encoders['urgency'].classes_.tolist(),
    'timestamp': datetime.now().isoformat()
}
pd.DataFrame([report]).to_csv(os.path.join(models_dir, 'model_performance.csv'), index=False)
logger.info("Model performance report saved")