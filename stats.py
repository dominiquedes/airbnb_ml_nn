import pandas as pd
import json
from pathlib import Path
import numpy as np

def save_stats_for_frontend():
    print("Reading data...")
    df = pd.read_csv('data/grouped_listings.csv')
    
    # Create output directory
    output_dir = Path('data/frontend')
    output_dir.mkdir(exist_ok=True)
    
    # Prepare statistics dictionary
    stats = {
        "overall": {
            "total_listings": int(len(df)),
            "unique_hosts": int(df['host_id'].nunique()),
            "neighborhoods": int(df['neighbourhood'].nunique())
        },
        
        "reviews": {
            "total_reviews": int(df['number_of_reviews'].sum()),
            "avg_reviews_per_listing": float(df['number_of_reviews'].mean()),
            "median_reviews_per_listing": float(df['number_of_reviews'].median()),
            "avg_review_score": float(df['review_scores_rating'].mean()),
            "median_review_score": float(df['review_scores_rating'].median()),
            "avg_reviews_per_month": float(df['reviews_per_month'].mean()),
            "median_reviews_per_month": float(df['reviews_per_month'].median())
        },
        
        "prices": {
            "avg_price": float(df['price'].mean()),
            "median_price": float(df['price'].median()),
            "min_price": float(df['price'].min()),
            "max_price": float(df['price'].max()),
            "price_std": float(df['price'].std())
        },
        
        "accommodation": {
            "avg_accommodates": float(df['accommodates'].mean()),
            "median_accommodates": float(df['accommodates'].median()),
            "avg_bedrooms": float(df['bedrooms'].mean()),
            "median_bedrooms": float(df['bedrooms'].median()),
            "avg_bathrooms": float(df['bathrooms'].mean()),
            "median_bathrooms": float(df['bathrooms'].median()),
            "avg_amenities": float(df['amenities_count'].mean()),
            "median_amenities": float(df['amenities_count'].median())
        },
        
        "room_types": df['room_type'].value_counts().to_dict(),
        
        "property_types": df['property_type'].value_counts().head(10).to_dict(),
        
        "performance_groups": {
            "distribution": df['performance_group'].value_counts().to_dict(),
            "percentage": (df['performance_group'].value_counts() / len(df) * 100).round(2).to_dict()
        },
        
        "group_statistics": {}
    }
    
    # Add detailed group statistics
    for group in df['performance_group'].unique():
        group_df = df[df['performance_group'] == group]
        stats["group_statistics"][group] = {
            "price": {
                "mean": float(group_df['price'].mean()),
                "median": float(group_df['price'].median()),
                "std": float(group_df['price'].std()),
                "min": float(group_df['price'].min()),
                "max": float(group_df['price'].max())
            },
            "reviews": {
                "mean": float(group_df['number_of_reviews'].mean()),
                "median": float(group_df['number_of_reviews'].median()),
                "total": int(group_df['number_of_reviews'].sum())
            },
            "scores": {
                "mean": float(group_df['review_scores_rating'].mean()),
                "median": float(group_df['review_scores_rating'].median())
            },
            "accommodation": {
                "avg_accommodates": float(group_df['accommodates'].mean()),
                "median_accommodates": float(group_df['accommodates'].median()),
                "avg_bedrooms": float(group_df['bedrooms'].mean()),
                "median_bedrooms": float(group_df['bedrooms'].median()),
                "avg_bathrooms": float(group_df['bathrooms'].mean()),
                "median_bathrooms": float(group_df['bathrooms'].median()),
                "avg_amenities": float(group_df['amenities_count'].mean()),
                "median_amenities": float(group_df['amenities_count'].median())
            },
            "superhost_percentage": float(group_df['is_superhost_numeric'].mean()),
            "top_property_types": group_df['property_type'].value_counts().head(5).to_dict()
        }
    
    # Save to JSON file
    output_path = output_dir / 'statistics.json'
    print(f"Saving statistics to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Statistics saved successfully!")
    
    # Also save a sample of the data for frontend use
    sample_data = df.sample(min(1000, len(df))).to_dict('records')
    sample_path = output_dir / 'sample_listings.json'
    
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample data saved successfully!")

if __name__ == "__main__":
    save_stats_for_frontend() 