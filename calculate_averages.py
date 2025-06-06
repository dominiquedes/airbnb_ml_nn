import pandas as pd
import json

# Read the listings data
print("Reading listings data...")
df = pd.read_csv('data/listings.csv')

# Calculate averages for numerical columns
averages = {
    'accommodates': df['accommodates'].mean(),
    'bathrooms': df['bathrooms'].mean(),
    'bedrooms': df['bedrooms'].mean(),
    'beds': df['beds'].mean(),
    'minimum_nights': df['minimum_nights'].mean(),
    'latitude': df['latitude'].mean(),
    'longitude': df['longitude'].mean(),
    # Review-related columns
    'number_of_reviews': df['number_of_reviews'].mean(),
    'number_of_reviews_ltm': df['number_of_reviews_ltm'].mean(),
    'number_of_reviews_l30d': df['number_of_reviews_l30d'].mean(),
    'number_of_reviews_ly': df['number_of_reviews_ly'].mean() if 'number_of_reviews_ly' in df.columns else None,
    'reviews_per_month': df['reviews_per_month'].mean(),
    'review_scores_rating': df['review_scores_rating'].mean(),
    'review_scores_accuracy': df['review_scores_accuracy'].mean(),
    'review_scores_cleanliness': df['review_scores_cleanliness'].mean(),
    'review_scores_checkin': df['review_scores_checkin'].mean(),
    'review_scores_communication': df['review_scores_communication'].mean(),
    'review_scores_location': df['review_scores_location'].mean(),
    'review_scores_value': df['review_scores_value'].mean()
}

# Get most common values for categorical columns
averages['neighborhood'] = df['neighbourhood_cleansed'].mode().iloc[0]
averages['room_type'] = df['room_type'].mode().iloc[0]

# Round numerical values
for key in averages:
    if isinstance(averages[key], float):
        averages[key] = round(averages[key], 2)

# Save to JSON file
print("Saving averages to defaults.json...")
with open('defaults.json', 'w') as f:
    json.dump(averages, f, indent=4)

print("Averages calculated and saved:")
print(json.dumps(averages, indent=4)) 