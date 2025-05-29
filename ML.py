import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from scipy.stats import randint, uniform
import gc  # For garbage collection
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l1_l2
from xgboost import XGBRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Memory efficient data loading
print("Loading data...")
dtypes = {
    'price': str,
    'room_type': str,
    'property_type': str,
    'neighbourhood_cleansed': str,
    'host_is_superhost': str,
    'amenities': str,
    'latitude': float,
    'longitude': float,
    'accommodates': int,
    'bathrooms': float,
    'bedrooms': float,
    'beds': float,
    'number_of_reviews': int,
    'minimum_nights': int,
    'review_scores_rating': float,
    'review_scores_cleanliness': float,
    'review_scores_value': float
}

listings_df = pd.read_csv('data/listings.csv', dtype=dtypes, usecols=dtypes.keys())

# Clean price (only for target variable)
listings_df['price'] = listings_df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Log transform price to handle skewness
listings_df['price'] = np.log1p(listings_df['price'])

# More efficient amenity processing
def extract_amenity_categories(amenities_str):
    categories = {
        'essential': set(['Wifi', 'Air conditioning', 'Heating', 'Kitchen']),
        'safety': set(['Smoke alarm', 'Fire extinguisher', 'First aid kit', 'Carbon monoxide alarm']),
        'luxury': set(['Pool', 'Hot tub', 'Gym', 'TV']),
        'outdoor': set(['Patio', 'Balcony', 'Garden', 'BBQ grill'])
    }
    
    try:
        if pd.isna(amenities_str):
            return [0, 0, 0, 0]
        amenities_set = set(ast.literal_eval(amenities_str))
        return [
            sum(1 for item in categories['essential'] if item in amenities_set),
            sum(1 for item in categories['safety'] if item in amenities_set),
            sum(1 for item in categories['luxury'] if item in amenities_set),
            sum(1 for item in categories['outdoor'] if item in amenities_set)
        ]
    except:
        return [0, 0, 0, 0]

print("Processing amenities...")
amenities_features = listings_df['amenities'].apply(extract_amenity_categories)
listings_df['essential_amenities'] = [x[0] for x in amenities_features]
listings_df['safety_amenities'] = [x[1] for x in amenities_features]
listings_df['luxury_amenities'] = [x[2] for x in amenities_features]
listings_df['outdoor_amenities'] = [x[3] for x in amenities_features]

# Drop original amenities column to save memory
listings_df.drop('amenities', axis=1, inplace=True)
gc.collect()

# Convert superhost to numeric
listings_df['host_is_superhost_numeric'] = listings_df['host_is_superhost'].map({'t': 1, 'f': 0, None: 0})
listings_df.drop('host_is_superhost', axis=1, inplace=True)

# Fill missing values for numeric columns
numeric_columns = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 
                  'review_scores_cleanliness', 'review_scores_value']
for col in numeric_columns:
    median_val = listings_df[col].median()
    listings_df[col] = listings_df[col].fillna(median_val)

# Advanced Feature Engineering
print("Performing advanced feature engineering...")

# Location-based features
listings_df['location_cluster'] = pd.qcut(listings_df['latitude'] + listings_df['longitude'], 10, labels=False)

# Calculate neighborhood average prices
neighborhood_avg_price = listings_df.groupby('neighbourhood_cleansed')['price'].transform('mean')
listings_df['price_vs_neighborhood'] = listings_df['price'] / neighborhood_avg_price

# Room and capacity features
listings_df['total_rooms'] = listings_df['bedrooms'] + listings_df['bathrooms']
listings_df['beds_per_room'] = listings_df['beds'] / listings_df['bedrooms'].replace(0, 1)
listings_df['persons_per_room'] = listings_df['accommodates'] / listings_df['bedrooms'].replace(0, 1)
listings_df['price_per_person'] = listings_df['price'] / listings_df['accommodates']

# Interaction features
listings_df['room_bathroom_ratio'] = listings_df['bedrooms'] / listings_df['bathrooms'].replace(0, 1)
listings_df['review_value_ratio'] = listings_df['review_scores_value'] / listings_df['price']

# Amenity score with weighted importance
listings_df['amenity_score'] = (
    listings_df['essential_amenities'] * 2.0 +
    listings_df['safety_amenities'] * 1.5 +
    listings_df['luxury_amenities'] * 3.0 +
    listings_df['outdoor_amenities'] * 1.0
)

# Review score features
listings_df['review_score_total'] = (
    listings_df['review_scores_rating'] +
    listings_df['review_scores_cleanliness'] +
    listings_df['review_scores_value']
) / 3

# Review engagement score
listings_df['review_engagement'] = listings_df['number_of_reviews'] * listings_df['review_scores_rating'].fillna(0)

# Selected features for prediction
selected_features = [
    'accommodates',
    'bedrooms',
    'bathrooms',
    'beds',
    'room_type',
    'property_type',
    'latitude',
    'longitude',
    'location_cluster',
    'neighbourhood_cleansed',
    'host_is_superhost_numeric',
    'number_of_reviews',
    'minimum_nights',
    'review_scores_rating',
    'review_score_total',
    'total_rooms',
    'beds_per_room',
    'persons_per_room',
    'price_per_person',
    'room_bathroom_ratio',
    'review_value_ratio',
    'price_vs_neighborhood',
    'review_engagement',
    'amenity_score',
    'essential_amenities',
    'safety_amenities',
    'luxury_amenities',
    'outdoor_amenities'
]

# Prepare features and target
print("Preparing features and target...")
X = listings_df[selected_features].copy()
y = listings_df['price']

# Clear unused dataframe to free memory
del listings_df
gc.collect()

# Handle categorical variables
print("Encoding categorical variables...")
categorical_features = ['room_type', 'property_type', 'neighbourhood_cleansed']
for feature in categorical_features:
    # Fill NA values with a placeholder
    X[feature] = X[feature].fillna('Unknown')
    # Convert to string type to ensure compatibility
    X[feature] = X[feature].astype(str)
    # Encode categorical variables
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Remove extreme outliers using IQR method
def remove_outliers(df, target, threshold=2.5):
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (target >= Q1 - threshold * IQR) & (target <= Q3 + threshold * IQR)
    return df[outlier_mask], target[outlier_mask]

X, y = remove_outliers(X, y, threshold=2.5)

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Clear unneeded variables
del X
gc.collect()

# Optimized parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(150, 400),
    'max_depth': [8, 10, 12, 15, 20, None],
    'min_samples_split': randint(5, 20),
    'min_samples_leaf': randint(3, 10),
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'max_samples': [0.7, 0.8, 0.9]  # Added bootstrap sampling size
}

# Initialize base model with better defaults for preventing overfitting
base_rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=200,
    max_features='sqrt',
    bootstrap=True,
    min_samples_leaf=4,
    min_samples_split=8,
    oob_score=True  # Added out-of-bag score estimation
)

# Perform RandomizedSearchCV with cross-validation
print("\nPerforming hyperparameter tuning...")
rf_random = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_dist,
    n_iter=30,  # Increased from 20
    cv=5,       # Increased from 3
    random_state=42,
    n_jobs=-1,
    verbose=1,
    scoring='neg_root_mean_squared_error'
)

# Fit RandomizedSearchCV
rf_random.fit(X_train_scaled, y_train)

# Get best model
best_rf = rf_random.best_estimator_

# Make predictions (transform back from log scale)
print("\nMaking predictions...")
y_train_pred = np.expm1(best_rf.predict(X_train_scaled))
y_test_pred = np.expm1(best_rf.predict(X_test_scaled))
y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)

# Calculate metrics
print("\nCalculating metrics...")
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

print("\nModel Performance Metrics:")
print(f"Train RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Train MAE: ${train_mae:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Create directory for plots if it doesn't exist
if not os.path.exists('plots_rf'):
    os.makedirs('plots_rf')

print("\nGenerating visualizations for model analysis...")

# 1. Predicted vs Actual Prices Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual[:1000], y_test_pred[:1000], alpha=0.5, label='Test Data')
plt.scatter(y_train_actual[:1000], y_train_pred[:1000], alpha=0.5, label='Training Data')
plt.plot([min(y_test_actual.min(), y_train_actual.min()), 
          max(y_test_actual.max(), y_train_actual.max())], 
         [min(y_test_actual.min(), y_train_actual.min()), 
          max(y_test_actual.max(), y_train_actual.max())], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Random Forest: Predicted vs Actual Prices\nTraining and Test Data Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plots_rf/rf_predicted_vs_actual_comparison.png')
plt.close()

# 3. Price Range Analysis
def calculate_metrics_by_range(y_true, y_pred, n_bins=5):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['price_range'] = pd.qcut(df['true'], n_bins)
    
    metrics = df.groupby('price_range', observed=True).apply(
        lambda x: pd.Series({
            'mae': mean_absolute_error(x['true'], x['pred']),
            'rmse': np.sqrt(mean_squared_error(x['true'], x['pred'])),
            'r2': r2_score(x['true'], x['pred']),
            'range': f"${int(x['price_range'].iloc[0].left)}-{int(x['price_range'].iloc[0].right)}"
        })
    )
    
    metrics.index = [f"range_{i}" for i in range(len(metrics))]
    return metrics

train_metrics = calculate_metrics_by_range(y_train_actual, y_train_pred)
test_metrics = calculate_metrics_by_range(y_test_actual, y_test_pred)

# Plot MAE by price range
plt.figure(figsize=(12, 6))
x = range(len(train_metrics))
width = 0.35
plt.bar([i - width/2 for i in x], train_metrics['mae'], width, label='Training MAE', alpha=0.7)
plt.bar([i + width/2 for i in x], test_metrics['mae'], width, label='Test MAE', alpha=0.7)
plt.xticks(x, [metrics['range'] for metrics in train_metrics.to_dict('records')], rotation=45)
plt.xlabel('Price Range')
plt.ylabel('Mean Absolute Error ($)')
plt.title('Error Analysis by Price Range\nTraining vs Testing')
plt.legend()
plt.tight_layout()
plt.savefig('plots_rf/rf_mae_by_price_range_comparison.png')
plt.close()

# Save detailed numerical results to a file
results = {
    'Model Performance Metrics': {
        'Train RMSE': f'${train_rmse:.2f}',
        'Test RMSE': f'${test_rmse:.2f}',
        'Train MAE': f'${train_mae:.2f}',
        'Test MAE': f'${test_mae:.2f}',
        'Train R²': f'{train_r2:.4f}',
        'Test R²': f'{test_r2:.4f}'
    },
    'Best Parameters': rf_random.best_params_,
    'Price Range Analysis': {
        'Training': train_metrics.to_dict(),
        'Testing': test_metrics.to_dict()
    }
}

import json
with open('plots_rf/rf_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nModel Improvements Summary:")
print("1. Added log transformation of target variable")
print("2. Created advanced location-based features")
print("3. Added interaction features for rooms and capacity")
print("4. Improved amenity scoring system")
print("5. Added cross-validation and OOB score estimation")
print("6. Increased number of trees and tuning iterations")
print("7. Added bootstrap sampling size parameter")
print("8. Enhanced feature engineering with domain knowledge")

# Implement robust scaling for price
price_scaler = RobustScaler()
y_scaled = price_scaler.fit_transform(y.values.reshape(-1, 1))

# Handle outliers more aggressively
def remove_price_outliers(df, price_col, n_std=2):
    mean = df[price_col].mean()
    std = df[price_col].std()
    return df[(df[price_col] > mean - n_std * std) & 
             (df[price_col] < mean + n_std * std)]

# Define learning rate schedule
def learning_rate_schedule(epoch):
    initial_lr = 0.001
    decay = 0.1
    lr = initial_lr * (1.0 / (1.0 + decay * epoch))
    return lr

# Define custom loss function
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return 0.7 * mse + 0.3 * mae  # Weighted combination of MSE and MAE

# Create base model function
def create_base_model(input_shape):
    model = Sequential([
        InputLayer(shape=input_shape),
        BatchNormalization(),
        
        Dense(256, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6)),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6)),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6)),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_loss,
        metrics=['mae']
    )
    return model

# Create learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)

# Create ensemble of models
print("\nCreating ensemble of models...")
models = []
for i in range(3):
    model = create_base_model(input_shape=(X_train_scaled.shape[1],))
    models.append(model)

# Train ensemble
print("\nTraining ensemble models...")
for i, model in enumerate(models):
    print(f"\nTraining model {i+1}/3...")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=0.000001,
                verbose=1
            ),
            lr_scheduler
        ],
        verbose=1
    )

# 1. Add Cross-Validation
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Average CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 2. Feature Selection
selector = SelectFromModel(best_rf, prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# 3. Ensemble Method
estimators = [
    ('rf', RandomForestRegressor()),
    ('xgb', XGBRegressor())
]
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor()
)

# Add this after RandomForest training
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
