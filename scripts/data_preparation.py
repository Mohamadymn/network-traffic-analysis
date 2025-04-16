import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv('data/Friday-DDOS-clean.csv')

# Show available labels
print("Label breakdown:")
print(df['Label'].value_counts())

# Encode labels: BENIGN = 0, others = 1
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Drop non-numeric columns if any
non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric:
    print(f"Removing non-numeric columns: {non_numeric}")
    df = df.drop(columns=non_numeric)

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaNs
df = df.dropna()

# Split features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save prepared datasets
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Data prepared and saved.")