# Airbnb Price Prediction System

This project implements a machine learning system for predicting Airbnb listing prices using multiple models:
- Neural Network
- Random Forest
- Stacked Model (Ensemble)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the models:
```bash
python train_models.py
```

3. Run the Flask application:
```bash
python app.py
```

## Project Structure

- `app.py`: Main Flask application with prediction endpoints
- `NN.py`: Neural Network model implementation
- `ML.py`: Random Forest model implementation
- `stacked_model.py`: Stacked model implementation
- `train_models.py`: Script to train and save all models
- `models/`: Directory containing trained models and encoders
- `data/`: Directory containing the dataset

## API Endpoints

### POST /api/predict
Predicts the price of an Airbnb listing using multiple models.

Request body:
```json
{
    "accommodates": 2,
    "bedrooms": 1,
    "bathrooms": 1,
    "beds": 1,
    "room_type": "Entire home/apt",
    "property_type": "Apartment",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "neighbourhood_cleansed": "Manhattan",
    "host_is_superhost": true,
    "number_of_reviews": 10,
    "minimum_nights": 2,
    "review_scores_rating": 4.5,
    "review_scores_cleanliness": 4.5,
    "review_scores_value": 4.5
}
```

Response:
```json
{
    "predictions": {
        "neural_network": 150.00,
        "random_forest": 145.00,
        "stacked_model": 148.00,
        "ensemble": 147.67
    },
    "confidence": {
        "neural_network": 0.95,
        "random_forest": 0.92,
        "stacked_model": 0.94
    }
}
```

## Models

1. Neural Network: A deep learning model with multiple layers for complex pattern recognition
2. Random Forest: An ensemble of decision trees for robust predictions
3. Stacked Model: A meta-model that combines predictions from multiple base models

## Data Preprocessing

The system includes:
- Categorical variable encoding
- Feature scaling
- Missing value handling
- Outlier removal

## Performance

The models are evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared score 