import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, PowerTransformer
import tensorflow as tf
import joblib
import os
from NN import create_base_model
from ML import RandomForestRegressor
from stacked_model import StackedRegressor

def train_and_save_models():
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
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
        
        df = pd.read_csv('data/listings.csv', dtype=dtypes)
        
        # Clean price and handle outliers
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        price_q1 = df['price'].quantile(0.25)
        price_q3 = df['price'].quantile(0.75)
        price_iqr = price_q3 - price_q1
        price_lower_bound = price_q1 - 1.5 * price_iqr
        price_upper_bound = price_q3 + 1.5 * price_iqr
        df = df[(df['price'] >= price_lower_bound) & (df['price'] <= price_upper_bound)]
        
        # Log transform price
        df['price'] = np.log1p(df['price'])
        
        # Prepare features
        selected_features = [
            'accommodates', 'bedrooms', 'bathrooms', 'beds',
            'room_type', 'property_type', 'latitude', 'longitude',
            'neighbourhood_cleansed', 'host_is_superhost',
            'number_of_reviews', 'minimum_nights',
            'review_scores_rating', 'review_scores_cleanliness',
            'review_scores_value'
        ]
        
        X = df[selected_features].copy()
        y = df['price']
        
        # Handle categorical variables
        categorical_features = ['room_type', 'property_type', 'neighbourhood_cleansed']
        encoders = {}
        
        for feature in categorical_features:
            X[feature] = X[feature].fillna('Unknown')
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            encoders[feature] = le
            joblib.dump(le, f'models/{feature}_encoder.joblib')
        
        # Convert host_is_superhost to numeric
        X['host_is_superhost'] = X['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
        
        # Fill any remaining missing values with median
        for column in X.columns:
            X[column] = X[column].fillna(X[column].median())
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'models/scaler.joblib')
        
        # Transform target variable
        pt = PowerTransformer(method='yeo-johnson')
        y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
        joblib.dump(pt, 'models/target_transformer.joblib')
        
        # Save the number of features for later use
        n_features = X.shape[1]
        with open('models/n_features.txt', 'w') as f:
            f.write(str(n_features))
        
        # Train and save Neural Network model
        print("Training Neural Network model...")
        nn_model = create_base_model(input_shape=(n_features,))
        nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/nn_model.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = nn_model.fit(
            X_scaled,
            y_transformed,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train and save Random Forest model
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(X_scaled, y_transformed)
        joblib.dump(rf_model, 'models/rf_model.joblib')
        
        # Train and save Stacked model
        print("Training Stacked model...")
        stacked_model = StackedRegressor()
        stacked_model.fit(X_scaled, y_transformed)
        joblib.dump(stacked_model, 'models/stacked_model.joblib')
        
        print("All models have been trained and saved successfully!")
        
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        raise

if __name__ == '__main__':
    train_and_save_models() 