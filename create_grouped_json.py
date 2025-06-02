import pandas as pd
import json
from pathlib import Path

def create_grouped_json():
    # Read the CSV file
    print("Reading grouped listings data...")
    df = pd.read_csv('data/grouped_listings.csv')
    
    # Group by performance_group
    print("Grouping data by performance group...")
    grouped_data = {}
    
    for group, group_df in df.groupby('performance_group'):
        # Convert the group's data to a list of dictionaries
        group_listings = group_df.to_dict('records')
        grouped_data[group] = group_listings
    
    # Create output directory if it doesn't exist
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    # Save to JSON file
    output_path = output_dir / 'grouped_listings.json'
    print(f"Saving grouped data to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(grouped_data, f, indent=2)
    
    print("Done! JSON file has been created.")
    
    # Print some statistics
    print("\nStatistics:")
    for group in grouped_data:
        print(f"Group {group}: {len(grouped_data[group])} listings")

if __name__ == "__main__":
    create_grouped_json() 