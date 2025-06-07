import csv

def print_column_values(filename, column_name):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        if column_name not in reader.fieldnames:
            print(f"Error: '{column_name}' column not found.")
            print("Available columns:", reader.fieldnames)
            return

        for row in reader:
            value = row[column_name].strip()
            if value:
                print(value)

# Call it with the correct column
print_column_values('data/listings.csv', 'neighbourhood_group_cleansed')