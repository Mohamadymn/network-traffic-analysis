import pandas as pd

# Load the original dataset
raw_path = 'data/Friday-DDOS.csv'
clean_path = 'data/Friday-DDOS-clean.csv'

df = pd.read_csv(raw_path)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Save cleaned version
df.to_csv(clean_path, index=False)

print(f"Cleaned dataset saved to {clean_path}")