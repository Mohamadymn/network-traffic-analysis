import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and features
model = joblib.load('models/random_forest_ddos.pkl')
X_train = pd.read_csv('data/X_train.csv')

# Get feature importances
importances = model.feature_importances_
features = X_train.columns

# Sort by importance
feature_series = pd.Series(importances, index=features).sort_values(ascending=False)

# Plot top 15
top_n = 15
plt.figure(figsize=(10, 6))
feature_series.head(top_n).plot(kind='bar')
plt.title(f'Top {top_n} Important Features for Detecting Malicious Traffic')
plt.ylabel('Importance Score')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()