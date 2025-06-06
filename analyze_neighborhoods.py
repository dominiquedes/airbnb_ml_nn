import pandas as pd
import json

def analyze_neighborhoods():
    # Load the data
    print("Loading data...")
    df = pd.read_csv('data/listings.csv')
    
    # Get unique neighborhoods
    neighborhoods = df['neighbourhood_cleansed'].unique()
    
    # Sort them alphabetically
    neighborhoods = sorted(neighborhoods)
    
    # Count listings per neighborhood
    neighborhood_counts = df['neighbourhood_cleansed'].value_counts()
    
    # Create a dictionary with neighborhoods and their counts
    neighborhood_data = {
        'total_neighborhoods': len(neighborhoods),
        'neighborhoods': {}
    }
    
    for neighborhood in neighborhoods:
        count = neighborhood_counts[neighborhood]
        neighborhood_data['neighborhoods'][neighborhood] = {
            'count': int(count),
            'percentage': round((count / len(df)) * 100, 2)
        }
    
    # Save to JSON file
    with open('neighborhood_analysis.json', 'w') as f:
        json.dump(neighborhood_data, f, indent=4)
    
    # Print summary
    print(f"\nTotal number of unique neighborhoods: {len(neighborhoods)}")
    print("\nNeighborhoods with their listing counts:")
    print("-" * 50)
    
    # Print in a formatted way
    for neighborhood in neighborhoods:
        count = neighborhood_counts[neighborhood]
        percentage = (count / len(df)) * 100
        print(f"• {neighborhood}:")
        print(f"  - Listings: {count}")
        print(f"  - Percentage: {percentage:.2f}%")
        print()

if __name__ == "__main__":
    analyze_neighborhoods() 