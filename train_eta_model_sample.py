import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample data matching the app's categories
sample_data = pd.DataFrame({
    'pickup_location': ['pallikaranai', 'tambaram', 'delhi', 'warehouse', 'city_center', 'chennai', 'mumbai', 'pallikaranai'],
    'delivery_location': ['tambaram', 'delhi', 'warehouse', 'city_center', 'chennai', 'mumbai', 'pallikaranai', 'delhi'],
    'urgency': ['high', 'medium', 'low', 'high', 'medium', 'low', 'high', 'medium'],
    'weight': [100.0, 50.0, 200.0, 150.0, 80.0, 120.0, 90.0, 60.0],
    'volume': [2.5, 1.0, 5.0, 3.0, 1.5, 2.0, 2.2, 1.2],
    'preferred_time': ['2025-05-01 08:00:00', '2025-05-01 12:00:00', '2025-05-01 16:00:00', 
                       '2025-05-02 09:00:00', '2025-05-02 14:00:00', '2025-05-02 18:00:00', 
                       '2025-05-03 10:00:00', '2025-05-03 15:00:00'],
    'estimated_travel_time': [60.0, 45.0, 90.0, 75.0, 50.0, 65.0, 55.0, 48.0]  # In minutes
})

# Use sample data
df = sample_data
logger.info(f"Sample data loaded: {len(df)} rows")

# Print column names for debugging
logger.info(f"Columns in the data: {list(df.columns)}")

# Clean and select relevant columns
required_columns = ['pickup_location', 'delivery_location', 'urgency', 'weight', 'volume', 'preferred_time', 'estimated_travel_time']
df_cleaned = df[required_columns].copy()
logger.info(f"Selected required columns: {len(df_cleaned)} rows")

# Drop rows with missing or invalid data
df_cleaned = df_cleaned.dropna(subset=required_columns)
df_cleaned = df_cleaned[
    (df_cleaned['weight'] > 0) & 
    (df_cleaned['volume'] > 0) & 
    (df_cleaned['estimated_travel_time'] > 0)
]
logger.info(f"After cleaning: {len(df_cleaned)} rows")

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
    lambda x: urgency_mapping.get(x, x.lower()) if isinstance(x, str) else 'medium'  # Fallback to 'medium'
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
df_cleaned['preferred_time'] = pd.to_datetime(df_cleaned['preferred_time'])
df_cleaned['hour'] = df_cleaned['preferred_time'].dt.hour
df_cleaned['day_of_week'] = df_cleaned['preferred_time'].dt.dayofweek
df_cleaned['is_weekend'] = df_cleaned['preferred_time'].dt.dayofweek.isin([5, 6]).astype(int)

# Define location pair distances
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

# Filter out invalid distances
df_cleaned = df_cleaned[df_cleaned['approx_distance'] > 0]
logger.info(f"After filtering zero/negative distances: {len(df_cleaned)} rows")

# Encode categorical features
label_encoders = {}
categorical_cols = ['pickup_location', 'delivery_location', 'urgency']
for col in categorical_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col].str.lower())
    # Sort classes for consistency (especially for urgency)
    if col == 'urgency':
        sorted_classes = sorted(le.classes_)
        le.classes_ = np.array(sorted_classes)
        df_cleaned[col] = le.transform(df_cleaned[col].str.lower())
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

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model, encoders, scaler, and feature columns
models_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, 'eta_model.pkl'))
joblib.dump(label_encoders, os.path.join(models_dir, 'eta_encoders.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'eta_scaler.pkl'))
joblib.dump(feature_cols, os.path.join(models_dir, 'eta_feature_cols.pkl'))

logger.info(f"Model and artifacts saved successfully in {models_dir}")
logger.info(f"Feature importance: {dict(zip(feature_cols, model.feature_importances_))}")