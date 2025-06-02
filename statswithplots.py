import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_listings():
    print("Reading data...")
    df = pd.read_csv('data/grouped_listings.csv')
    
    # Create output directory for plots
    output_dir = Path('data/plots')
    output_dir.mkdir(exist_ok=True)
    
    # Overall Statistics
    print("\n=== OVERALL LISTINGS STATISTICS ===")
    print(f"Total number of listings: {len(df):,}")
    print(f"Number of unique hosts: {df['host_id'].nunique():,}")
    print(f"Number of neighborhoods: {df['neighbourhood'].nunique():,}")
    
    # Review Statistics
    print("\n=== REVIEW STATISTICS ===")
    review_stats = {
        'Total Reviews': df['number_of_reviews'].sum(),
        'Average Reviews per Listing': df['number_of_reviews'].mean(),
        'Median Reviews per Listing': df['number_of_reviews'].median(),
        'Average Review Score': df['review_scores_rating'].mean(),
        'Median Review Score': df['review_scores_rating'].median(),
        'Reviews per Month (Avg)': df['reviews_per_month'].mean(),
        'Reviews per Month (Median)': df['reviews_per_month'].median()
    }
    
    for stat, value in review_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Listing Characteristics
    print("\n=== LISTING CHARACTERISTICS ===")
    print("\nRoom Types:")
    print(df['room_type'].value_counts())
    
    print("\nProperty Types (Top 10):")
    print(df['property_type'].value_counts().head(10))
    
    print("\nAccommodation Statistics:")
    acc_stats = {
        'Average Accommodates': df['accommodates'].mean(),
        'Median Accommodates': df['accommodates'].median(),
        'Average Bedrooms': df['bedrooms'].mean(),
        'Median Bedrooms': df['bedrooms'].median(),
        'Average Bathrooms': df['bathrooms'].mean(),
        'Median Bathrooms': df['bathrooms'].median()
    }
    
    for stat, value in acc_stats.items():
        print(f"{stat}: {value:.2f}")
    
    print("\nAmenities Statistics:")
    print(f"Average number of amenities: {df['amenities_count'].mean():.2f}")
    print(f"Median number of amenities: {df['amenities_count'].median():.2f}")
    
    # Price Statistics
    print("\n=== PRICE STATISTICS ===")
    price_stats = {
        'Average Price': df['price'].mean(),
        'Median Price': df['price'].median(),
        'Minimum Price': df['price'].min(),
        'Maximum Price': df['price'].max(),
        'Price Standard Deviation': df['price'].std()
    }
    
    for stat, value in price_stats.items():
        print(f"{stat}: ${value:.2f}")
    
    # Performance Group Distribution
    print("\n=== PERFORMANCE GROUP DISTRIBUTION ===")
    group_dist = df['performance_group'].value_counts()
    print(group_dist)
    print("\nPercentage Distribution:")
    print((group_dist / len(df) * 100).round(2))
    
    print("\n" + "="*50 + "\n")
    
    # Original group-based statistics
    print("=== Basic Statistics by Performance Group ===")
    stats_by_group = df.groupby('performance_group').agg({
        'price': ['mean', 'median', 'std', 'min', 'max'],
        'number_of_reviews': ['mean', 'median', 'sum'],
        'review_scores_rating': ['mean', 'median'],
        'accommodates': ['mean', 'median'],
        'bedrooms': ['mean', 'median'],
        'bathrooms': ['mean', 'median'],
        'amenities_count': ['mean', 'median']
    }).round(2)
    
    print("\nDetailed Statistics:")
    print(stats_by_group)
    
    # Save detailed stats to CSV
    stats_by_group.to_csv('data/group_statistics.csv')
    print("\nDetailed statistics saved to 'data/group_statistics.csv'")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Price Distribution by Group
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='performance_group', y='price', data=df)
    plt.title('Price Distribution by Performance Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/plots/price_distribution.png')
    plt.close()
    
    # 2. Review Scores by Group
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='performance_group', y='review_scores_rating', data=df)
    plt.title('Review Scores by Performance Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/plots/review_scores.png')
    plt.close()
    
    # 3. Amenities Count by Group
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='performance_group', y='amenities_count', data=df)
    plt.title('Amenities Count by Performance Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/plots/amenities_count.png')
    plt.close()
    
    # 4. Room Type Distribution
    plt.figure(figsize=(12, 6))
    room_type_counts = df.groupby(['performance_group', 'room_type']).size().unstack()
    room_type_counts.plot(kind='bar', stacked=True)
    plt.title('Room Type Distribution by Performance Group')
    plt.xlabel('Performance Group')
    plt.ylabel('Count')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.savefig('data/plots/room_type_distribution.png')
    plt.close()
    
    # 5. Correlation Heatmap
    numeric_cols = ['price', 'number_of_reviews', 'review_scores_rating', 
                   'accommodates', 'bedrooms', 'bathrooms', 'amenities_count']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('data/plots/correlation_heatmap.png')
    plt.close()
    
    # Additional statistics
    print("\n=== Additional Statistics ===")
    
    # Superhost statistics
    superhost_stats = df.groupby('performance_group')['is_superhost_numeric'].mean()
    print("\nSuperhost Percentage by Group:")
    print(superhost_stats.round(3))
    
    # Property type distribution
    print("\nTop 5 Property Types by Group:")
    for group in df['performance_group'].unique():
        group_df = df[df['performance_group'] == group]
        top_properties = group_df['property_type'].value_counts().head()
        print(f"\nGroup {group}:")
        print(top_properties)
    
    print("\nAnalysis complete! Check the 'data/plots' directory for visualizations.")

if __name__ == "__main__":
    analyze_listings()
