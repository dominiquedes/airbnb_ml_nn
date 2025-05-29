import pandas as pd
import numpy as np


# Read the listings data
print("Reading listings data...")
listings_df = pd.read_csv('data/listings.csv')

# Convert IDs to strings to ensure matching
listings_df['id'] = listings_df['id'].astype(str)

# Print unique counts
print(f"\nUnique listings: {len(listings_df['id'].unique())}")

# Clean price column - remove $ and convert to float
listings_df['price'] = listings_df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Calculate number of amenities
print("\nCalculating amenities...")
listings_df['amenities_count'] = listings_df['amenities'].str.count(',') + 1

# Convert superhost to numeric (1 for superhost, 0 for not)
listings_df['is_superhost_numeric'] = listings_df['host_is_superhost'].map({'t': 1, 'f': 0})

# Calculate performance metrics
print("\nCalculating performance metrics...")
listings_df['performance_score'] = (
    (listings_df['price'] / listings_df['price'].max()) * 0.3 +  # Price component (30%)
    (listings_df['number_of_reviews'] / listings_df['number_of_reviews'].max()) * 0.3 +  # Reviews component (30%)
    (listings_df['amenities_count'] / listings_df['amenities_count'].max()) * 0.2 +  # Amenities component (20%)
    listings_df['is_superhost_numeric'] * 0.2  # Superhost component (20%)
)

# Group listings into performance tiers
print("\nGrouping listings into performance tiers...")
listings_df['performance_group'] = pd.qcut(
    listings_df['performance_score'],
    q=3,
    labels=['Bottom-performing', 'Mid-performing', 'Top-performing']
)

# Print summary statistics
print("\nPerformance Group Distribution:")
print(listings_df['performance_group'].value_counts())

print("\nAverage metrics by performance group:")
group_stats = listings_df.groupby('performance_group').agg({
    'price': ['mean', 'median'],
    'number_of_reviews': ['mean', 'median'],
    'review_scores_rating': ['mean', 'median'],
    'amenities_count': ['mean', 'median'],
    'is_superhost_numeric': 'mean'
}).round(2)

print(group_stats)

# Save the grouped listings data
print("\nSaving grouped listings data...")
listings_df = listings_df.rename(columns={'id': 'listing_id'})
listings_df.to_csv('data/grouped_listings.csv', index=False)
print("Grouped listings saved to 'grouped_listings.csv'")

# Now process reviews
print("\nProcessing reviews data...")
reviews_df = pd.read_csv('data/reviews.csv')
reviews_df['listing_id'] = reviews_df['listing_id'].astype(str)

# Create a dictionary for quick lookup of performance groups
performance_groups = dict(zip(listings_df['listing_id'], listings_df['performance_group']))

# Process reviews in chunks to handle large files efficiently
chunk_size = 100000
processed_reviews = []
total_reviews = len(reviews_df)
matched_count = 0

print("\nMatching reviews with listings performance groups...")
for i in range(0, len(reviews_df), chunk_size):
    chunk = reviews_df.iloc[i:i+chunk_size].copy()
    
    # Add performance group to each review if listing_id matches
    chunk['performance_group'] = chunk['listing_id'].map(performance_groups)
    
    # Keep track of matches
    matched_count += chunk['performance_group'].notna().sum()
    
    # Store processed chunk
    processed_reviews.append(chunk)
    
    # Print progress
    print(f"Processed {min(i + chunk_size, total_reviews)} out of {total_reviews} reviews...")

# Combine all processed chunks
filtered_reviews_df = pd.concat(processed_reviews, ignore_index=True)

# Print matching statistics
print(f"\nTotal reviews processed: {total_reviews}")
print(f"Reviews with matching listings: {matched_count}")
print(f"Reviews without matching listings: {total_reviews - matched_count}")

# Clean reviews by removing rows with NaN performance groups
print("\nCleaning reviews data...")
print(f"Reviews before cleaning: {len(filtered_reviews_df)}")
filtered_reviews_df = filtered_reviews_df.dropna(subset=['performance_group'])
print(f"Reviews after removing NaN performance groups: {len(filtered_reviews_df)}")

# Print review distribution by performance group
print("\nReview Distribution by Performance Group:")
print(filtered_reviews_df['performance_group'].value_counts())

# Save the matched reviews
print("\nSaving matched reviews data...")
filtered_reviews_df.to_csv('data/grouped_reviews.csv', index=False)
print("Matched reviews saved to 'grouped_reviews.csv'")

# Print sample of matched reviews
print("\nSample of matched reviews (first 5):")
sample_cols = ['listing_id', 'date', 'performance_group']
print(filtered_reviews_df[sample_cols].head()) 