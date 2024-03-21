import pandas as pd

# Load the original CSV file
df = pd.read_csv('combined_updated.csv')

# Split the DataFrame into two DataFrames based on row counts
df_part1 = df.iloc[:43845]  # First 3200 rows
df_part2 = df.iloc[43845:]  # Remaining rows

# Write the two DataFrames to separate CSV files
df_part1.to_csv('train_updated.csv', index=False)
df_part2.to_csv('test_updated.csv', index=False)