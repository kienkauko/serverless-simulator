import os
import pandas as pd

# Directory where the CSV files are located
directory = os.path.dirname(os.path.abspath(__file__))

# Get all files that start with "profile_" and end with ".csv"
csv_files = [f for f in os.listdir(directory) if f.startswith('profile_') and f.endswith('.csv')]

# Read and combine all CSV files
all_data = []
for i, file in enumerate(csv_files):
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    all_data.append(df)

# Combine all dataframes
combined_df = pd.concat(all_data, ignore_index=True)

# Write to a new CSV file
output_file = os.path.join(directory, 'combined_profile_results.csv')
combined_df.to_csv(output_file, index=False)

print(f"Combined {len(csv_files)} files into {output_file}")