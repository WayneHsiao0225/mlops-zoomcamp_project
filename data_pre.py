import pandas as pd

# Load your DataFrame
df= pd.read_csv('BankCustomerChurnPrediction.csv')

# Calculate the split index
split_index = len(df) // 2

# Split the DataFrame into two parts (first 50% and second 50%)
df1 = df.iloc[:split_index]
df2 = df.iloc[split_index:]

# Save the DataFrames to CSV files
df1.to_csv('split_part1.csv', index=False)
df2.to_csv('split_part2.csv', index=False)