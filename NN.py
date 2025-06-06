import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import ast
import os
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression

# Read the data
print("Loading data...")
listings_df = pd.read_csv('data/listings.csv')

# Clean and preprocess data
print("Cleaning and preprocessing data...")

# Clean price (only for target variable) - to remove the dollar sign and commas
listings_df['price'] = listings_df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Handle amenities - to get the number of each type of amenity
def extract_amenity_categories(amenities_str):
    categories = {
        'essential': ['Wifi', 'Air conditioning', 'Heating', 'Kitchen'],
        'safety': ['Smoke alarm', 'Fire extinguisher', 'First aid kit', 'Carbon monoxide alarm'],
        'luxury': ['Pool', 'Hot tub', 'Gym', 'TV'],
        'outdoor': ['Patio', 'Balcony', 'Garden', 'BBQ grill']
    }
    
    try:
        if pd.isna(amenities_str): # If the amenities are missing, return 0 for all categories
            return [0, 0, 0, 0]
        amenities_list = ast.literal_eval(amenities_str)
        return [
            sum(1 for item in categories['essential'] if item in amenities_list),
            sum(1 for item in categories['safety'] if item in amenities_list),
            sum(1 for item in categories['luxury'] if item in amenities_list),
            sum(1 for item in categories['outdoor'] if item in amenities_list)
        ]
    except:
        return [0, 0, 0, 0]

amenities_features = listings_df['amenities'].apply(extract_amenity_categories) # Apply the extract_amenity_categories function to the amenities column - to get the number of each type of amenity
listings_df['essential_amenities'] = [x[0] for x in amenities_features]
listings_df['safety_amenities'] = [x[1] for x in amenities_features]
listings_df['luxury_amenities'] = [x[2] for x in amenities_features]
listings_df['outdoor_amenities'] = [x[3] for x in amenities_features]

# Convert superhost to numeric
listings_df['host_is_superhost_numeric'] = listings_df['host_is_superhost'].fillna('f').map({'t': 1, 'f': 0})

# Fill missing values for numeric columns with median
numeric_columns = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 
                  'review_scores_cleanliness', 'review_scores_value']
for col in numeric_columns:
    listings_df[col] = listings_df[col].fillna(listings_df[col].median())

# Calculate review score mean and variance - to create new features that are more informative
review_cols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_value']
listings_df['review_score_mean'] = listings_df[review_cols].mean(axis=1)
listings_df['review_score_variance'] = listings_df[review_cols].var(axis=1)

# Feature engineering
print("Engineering new features...")

# Safe division function - to avoid division by zero
def safe_divide(a, b, fill_value=0):
    return np.divide(a, b, out=np.full_like(a, fill_value, dtype=float), where=b!=0)

# Room and capacity metrics
listings_df['bed_to_accommodates_ratio'] = safe_divide(listings_df['beds'], listings_df['accommodates'])
listings_df['bathrooms_to_bedrooms_ratio'] = safe_divide(listings_df['bathrooms'], listings_df['bedrooms'])
listings_df['beds_to_bedrooms_ratio'] = safe_divide(listings_df['beds'], listings_df['bedrooms'])
listings_df['total_rooms'] = listings_df['bedrooms'] + listings_df['bathrooms']
listings_df['room_efficiency'] = safe_divide(listings_df['accommodates'], listings_df['total_rooms'])

# Location features
location_scaler = RobustScaler()
listings_df[['latitude_scaled', 'longitude_scaled']] = location_scaler.fit_transform(
    listings_df[['latitude', 'longitude']]
)
listings_df['location_score'] = np.sqrt(
    listings_df['latitude_scaled']**2 + listings_df['longitude_scaled']**2
)

# Review-based features - to create a feature that is more informative
listings_df['review_density'] = safe_divide(listings_df['number_of_reviews'], 
                                          (listings_df['review_scores_rating'] + 1))

# Amenity scores
listings_df['amenity_score'] = (listings_df['essential_amenities'] * 1.0 +
                               listings_df['safety_amenities'] * 1.2 +
                               listings_df['luxury_amenities'] * 1.5 +
                               listings_df['outdoor_amenities'] * 1.3)

# Select features for prediction 
selected_features = [
    'accommodates',
    'bedrooms',
    'bathrooms',
    'beds',
    'room_type',
    'property_type',
    'latitude',
    'longitude',
    'neighbourhood_cleansed',
    'host_is_superhost_numeric',
    'number_of_reviews',
    'minimum_nights',
    'bed_to_accommodates_ratio',
    'bathrooms_to_bedrooms_ratio',
    'beds_to_bedrooms_ratio',
    'total_rooms',
    'room_efficiency',
    'location_score',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_value',
    'review_score_mean',
    'review_score_variance',
    'review_density',
    'essential_amenities',
    'safety_amenities',
    'luxury_amenities',
    'outdoor_amenities',
    'amenity_score'
]

# Prepare features and target - to split the data into features and target
print("Preparing features and target...")
X = listings_df[selected_features].copy()
y = listings_df['price']

# Handle categorical variables
print("Encoding categorical variables...")
categorical_features = ['room_type', 'property_type', 'neighbourhood_cleansed']
for feature in categorical_features:
    X[feature] = X[feature].fillna('missing')
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Fill any remaining missing values with median
for column in X.columns:
    X[column] = X[column].fillna(X[column].median())

# Remove extreme outliers using IQR method - determines the outlier by creating a range and then removing the values that are outside of that range
def remove_outliers(df, target, threshold=2.5):
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (target >= Q1 - threshold * IQR) & (target <= Q3 + threshold * IQR)
    return df[outlier_mask], target[outlier_mask]

X, y = remove_outliers(X, y, threshold=2.5)

# Feature selection
print("Selecting best features...")
selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(X, y)
selected_features_mask = selector.get_support()
final_features = [feature for feature, selected in zip(X.columns, selected_features_mask) if selected]
X = X[final_features]

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform features and target
print("Transforming features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transform target using Yeo-Johnson transformation - to make the target more normally distributed - Yeo-Johnson is a power transformation that is used to make the target more normally distributed
pt = PowerTransformer(method='yeo-johnson')
y_train_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_transformed = pt.transform(y_test.values.reshape(-1, 1)).ravel()

# Create the neural network model - layers are the number of neurons in each layer - kernel_regularizer is a regularization term to prevent overfitting - l1_l2 is a regularization term that is used to prevent overfitting
# LeakyReLU is an activation function that is used to introduce non-linearity into the model - BatchNormalization is a technique that is used to normalize the input to the next layer - Dropout is a technique that is used to prevent overfitting
# Each layer is connected to the next layer by a weight matrix - the weights are learned during training
# The model is trained using the Adam optimizer - Adam is a gradient descent optimization algorithm that is used to minimize the loss function
# The loss function is the mean squared error - the model is trained to minimize the loss function
# The metrics are the metrics that are used to evaluate the model - the metrics are the mean absolute error and the root mean squared error
# The model is trained for 150 epochs - the model is trained for 150 epochs to ensure that the model is trained for a sufficient number of epochs
# The batch size is 32 - the batch size is the number of samples that are used to update the weights of the model

print("\nCreating neural network model...")
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    
    Dense(512, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    
    Dense(1)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='huber',
    metrics=['mae']
)

# Create callbacks  - EarlyStopping is a callback that stops the training if the validation loss doesn't improve for a certain number of epochs 
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=2
)
# ReduceLROnPlateau is a callback that reduces the learning rate if the validation loss doesn't improve for a certain number of epochs
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.000001,
    verbose=2
)

# Train the model
print("\nTraining the model...")
print("(Will stop early if validation loss doesn't improve for 25 epochs)")
history = model.fit(
    X_train_scaled,
    y_train_transformed,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Print training summary
print("\nTraining Summary:")
print(f"Trained for {len(history.history['loss'])} epochs")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")

# Make predictions and inverse transform
print("\nMaking predictions...")
y_train_pred = pt.inverse_transform(
    model.predict(X_train_scaled).reshape(-1, 1)
).ravel()
y_test_pred = pt.inverse_transform(
    model.predict(X_test_scaled).reshape(-1, 1)
).ravel()

# Calculate metrics
print("\nCalculating metrics...")
train_rmse = np.sqrt(np.mean((y_train_pred - y_train) ** 2))
test_rmse = np.sqrt(np.mean((y_test_pred - y_test) ** 2))
train_mae = np.mean(np.abs(y_train_pred - y_train))
test_mae = np.mean(np.abs(y_test_pred - y_test))

# Calculate R-squared
train_r2 = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print("\nModel Performance Metrics:")
print(f"Train RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"Train MAE: ${train_mae:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Create directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

print("\nGenerating visualizations for model analysis...")

# 1. Predicted vs Actual Prices Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Neural Network: Predicted vs Actual Prices')
plt.tight_layout()
plt.savefig('plots/nn_predicted_vs_actual.png')
plt.close()

# 3. Price Range Analysis
plt.figure(figsize=(10, 6))
price_ranges = pd.cut(y_test, bins=5)
mae_by_range = pd.DataFrame({
    'price_range': price_ranges,
    'abs_error': abs(y_test_pred - y_test)
}).groupby('price_range', observed=True)['abs_error'].mean()

plt.bar(range(len(mae_by_range)), mae_by_range.values)
plt.xticks(range(len(mae_by_range)), [f'${int(i.left)}-{int(i.right)}' for i in mae_by_range.index], rotation=45)
plt.xlabel('Price Range')
plt.ylabel('Mean Absolute Error ($)')
plt.title('Neural Network: MAE by Price Range')
plt.tight_layout()
plt.savefig('plots/nn_mae_by_price_range.png')
plt.close()

# Save numerical results to a file
results = {
    'Model Performance Metrics': {
        'Train RMSE': f'${train_rmse:.2f}',
        'Test RMSE': f'${test_rmse:.2f}',
        'Train MAE': f'${train_mae:.2f}',
        'Test MAE': f'${test_mae:.2f}',
        'Train R²': f'{train_r2:.4f}',
        'Test R²': f'{test_r2:.4f}'
    },
    'Selected Features': final_features
}

import json
with open('plots/nn_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nAll visualizations have been saved in the 'plots' directory:")
print("1. nn_predicted_vs_actual.png - Scatter plot of predicted vs actual prices")
print("2. nn_mae_by_price_range.png - MAE analysis by price range")
print("3. nn_results.json - Numerical results")

def create_base_model(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        
        Dense(512, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    
    return model
