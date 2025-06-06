import pandas as pd

# Read the CSV file
df = pd.read_csv('data/listings.csv')

# Get unique neighborhoods and sort them
unique_neighborhoods = sorted(df['neighbourhood_cleansed'].unique())

# Print each neighborhood
print("Unique neighborhoods in listings.csv:")
for neighborhood in unique_neighborhoods:
    print(f"'{neighborhood}'") 