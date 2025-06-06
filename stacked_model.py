import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from scipy.stats import randint, uniform
import gc
import warnings
import json
import joblib
warnings.filterwarnings('ignore')

# Stacked Model

class StackedRegressor:
    def __init__(self):
        # Base models
        self.rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        # XGBoost
        self.xgb = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            n_jobs=-1,
            random_state=42
        )
        
        # Meta-model - Combination of base models 
        self.meta_model = LassoCV(
            cv=5,
            random_state=42,
            max_iter=2000
        )
        
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        # Create out-of-fold predictions for training meta-model
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf_oof = np.zeros(len(X))
        xgb_oof = np.zeros(len(X))
        
        # Train base models and get OOF (Out of Fold) predictions
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train Random Forest
            self.rf.fit(X_train, y_train)
            rf_oof[val_idx] = self.rf.predict(X_val)
            
            # Train XGBoost
            self.xgb.fit(X_train, y_train)
            xgb_oof[val_idx] = self.xgb.predict(X_val)
        
        # Train meta-model on out-of-fold predictions
        meta_features = np.column_stack([rf_oof, xgb_oof])
        self.meta_model.fit(meta_features, y)
        
        # Final fit on whole dataset
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        
        return self
    
    def predict(self, X):
        # Get predictions from base models
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        
        # Combine predictions using meta-model
        meta_features = np.column_stack([rf_pred, xgb_pred])
        return self.meta_model.predict(meta_features)

def load_and_preprocess_data():
    print("Loading data...")
    # Define data types for each column
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

    df = pd.read_csv('data/listings.csv', dtype=dtypes, usecols=dtypes.keys())
    
    # Clean price
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Log transform price - to make it more normally distributed
    df['price'] = np.log1p(df['price'])
    
    # Process amenities - to get the number of each type of amenity
    print("Processing amenities...")
    amenities_features = df['amenities'].apply(extract_amenity_categories)
    df['essential_amenities'] = [x[0] for x in amenities_features]
    df['safety_amenities'] = [x[1] for x in amenities_features]
    df['luxury_amenities'] = [x[2] for x in amenities_features]
    df['outdoor_amenities'] = [x[3] for x in amenities_features]
    
    # Feature engineering - to create new features that are more informative
    print("Engineering features...")
    
    # Convert superhost to numeric first - to make it easier to use in the model
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
    
    # Location features - to group listings by location
    df['location_cluster'] = pd.qcut(df['latitude'] + df['longitude'], 10, labels=False)
    
    # Room features - to create new features that are more informative
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['beds_per_room'] = df['beds'] / df['bedrooms'].replace(0, 1)
    df['persons_per_room'] = df['accommodates'] / df['bedrooms'].replace(0, 1)
    
    # Price per capacity - to create a feature that is more informative
    df['price_per_person'] = df['price'] / df['accommodates']
    
    # Review features - to create new features that are more informative
    review_cols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_value']
    df['review_scores_mean'] = df[review_cols].mean(axis=1)
    df['review_scores_std'] = df[review_cols].std(axis=1)
    
    # Amenity score - to create a feature that is more informative - weighted by the importance of each amenity
    df['amenity_score'] = (
        df['essential_amenities'] * 2.0 +
        df['safety_amenities'] * 1.5 +
        df['luxury_amenities'] * 3.0 +
        df['outdoor_amenities'] * 1.0
    )
    
    # Handle categorical variables - to convert them to numeric values
    for col in ['room_type', 'property_type', 'neighbourhood_cleansed']:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Fill missing values - to replace them with the median of the column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Drop the original host_is_superhost column
    df = df.drop('amenities', axis=1)
    
    return df

def extract_amenity_categories(amenities_str):
    # Define the categories of amenities
    categories = {
        'essential': set(['Wifi', 'Air conditioning', 'Heating', 'Kitchen']),
        'safety': set(['Smoke alarm', 'Fire extinguisher', 'First aid kit', 'Carbon monoxide alarm']),
        'luxury': set(['Pool', 'Hot tub', 'Gym', 'TV']),
        'outdoor': set(['Patio', 'Balcony', 'Garden', 'BBQ grill'])
    }
    
    try:
        if pd.isna(amenities_str): # If the amenities are missing, return 0 for all categories
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

def create_visualizations(y_true, y_pred, model_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Predicted vs Actual - to see how the model performs
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (log scale)')
    plt.ylabel('Predicted Price (log scale)')
    plt.title(f'{model_name}: Predicted vs Actual Prices')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_vs_actual.png')
    plt.close()

    # 2. MAE by Price Range (in USD) - to see how the model performs at different price ranges
    try:
        # Convert to USD scale for binning
        y_true_usd = np.expm1(y_true)
        y_pred_usd = np.expm1(y_pred)
        
        # Create price ranges in USD
        price_ranges = pd.qcut(y_true_usd, 5, duplicates='drop')
        mae_by_range = pd.DataFrame({
            'price_range': price_ranges,
            'abs_error': abs(y_pred_usd - y_true_usd)  # Error in USD
        }).groupby('price_range', observed=True)['abs_error'].mean()

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(mae_by_range)), mae_by_range.values)
        plt.xticks(range(len(mae_by_range)), 
                   [f'${int(i.left)}-{int(i.right)}' for i in mae_by_range.index], 
                   rotation=45)
        plt.xlabel('Price Range (USD)')
        plt.ylabel('Mean Absolute Error ($)')
        plt.title(f'{model_name}: MAE by Price Range')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mae_by_price_range.png')
        plt.close()
    except Exception as e:
        print(f"\nWarning: Could not create MAE by price range plot: {str(e)}")

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'price']
    X = df[feature_cols].values
    y = df['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train stacked model
    print("\nTraining stacked model...")
    stacked_model = StackedRegressor()
    stacked_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = stacked_model.predict(X_test_scaled)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Save the model and necessary files
    print("\nSaving models and transformers...")
    os.makedirs('models', exist_ok=True)
    
    # Save the stacked model
    joblib.dump(stacked_model, 'models/stacked_model.joblib')
    
    # Save the scaler if not already saved
    if not os.path.exists('models/scaler.joblib'):
        joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save the target transformer if not already saved
    if not os.path.exists('models/target_transformer.joblib'):
        target_transformer = RobustScaler()
        target_transformer.fit(y_train.reshape(-1, 1))
        joblib.dump(target_transformer, 'models/target_transformer.joblib')
    
    # Create visualizations
    print("\nGenerating visualizations for model analysis...")
    create_visualizations(y_test, y_pred, 'Stacked Model', 'plots_stacked')
    
    print("\nAll visualizations have been saved in the 'plots_stacked' directory")

    # Print feature importance summary with correct pairing of features and importances - to print the feature importance summary   
    print("\nFeature Importance Summary:")
    print("\nTop 5 Most Important Features (Random Forest):")
    rf_importance = list(zip(feature_cols, stacked_model.rf.feature_importances_))
    for feature, importance in sorted(rf_importance, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feature}: {importance:.4f}")
    
    print("\nTop 5 Most Important Features (XGBoost):")
    xgb_importance = list(zip(feature_cols, stacked_model.xgb.feature_importances_))
    for feature, importance in sorted(xgb_importance, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main() 