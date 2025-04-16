import pandas as pd

# Load the dataset
file_path = 'data/Friday-DDOS.csv'
df = pd.read_csv(file_path)

# Basic info
print(" Dataset Overview")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nColumns:\n", df.columns.tolist())

# Check for nulls
print("\n Missing Values:")
print(df.isnull().sum())

# Preview of the label column
print("\n Unique Labels:")
print(df[' Label'].value_counts())