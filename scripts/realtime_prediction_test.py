import pandas as pd
import joblib
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load('models/random_forest_ddos.pkl')

# Load and clean the new dataset
new_data = pd.read_csv('data/Friday-DDOS-test.csv')
new_data.columns = new_data.columns.str.strip()
new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
new_data.dropna(inplace=True)

# Label encoding: BENIGN = 0, others = 1
new_data['Label'] = new_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Load training data to ensure consistency
X_train = pd.read_csv('data/X_train.csv')
expected_columns = X_train.columns.tolist()

# Separate benign and malicious
benign = new_data[new_data['Label'] == 0].sample(n=10, random_state=42)
malicious = new_data[new_data['Label'] == 1].sample(n=20, random_state=42)

# Combine and shuffle
combined = pd.concat([benign, malicious]).sample(frac=1, random_state=42).reset_index(drop=True)
X_new = combined.drop('Label', axis=1)[expected_columns] # Aligning data columns
y_new = combined['Label']

# Collect predictions and actual labels
predictions = []
actuals = []

# Simulate real-time predictions
log_path = 'reports/realtime_prediction_log.txt'
with open(log_path, 'w') as log_file:
    log_file.write("Real-Time Prediction Log\n")
    log_file.write("========================\n\n")
    for i in range(len(X_new)):  # Simulate the 30 entries
        row = X_new.iloc[i].values.reshape(1, -1)
        actual_label = y_new.iloc[i]
        prediction = model.predict(row)[0]
        predictions.append(prediction)
        actuals.append(actual_label)
        status = "MALICIOUS" if prediction == 1 else "BENIGN"
        expected = "MALICIOUS" if actual_label == 1 else "BENIGN"
        log_entry = f"[{i+1:02}] Detected: {status} | Expected: {expected}\n"
        print(log_entry.strip())
        log_file.write(log_entry)
        time.sleep(0.3)

# Accuracy rating
    accuracy = accuracy_score(actuals, predictions)
    summary = f"\nOverall Accuracy on 30 simulated entries: {accuracy:.2%}\n"
    print(summary)
    log_file.write(summary)