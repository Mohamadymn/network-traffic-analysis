# Network Traffic Analysis & DDoS Detection (Machine Learning Project)

This project simulates a real-world scenario where machine learning is used to detect Distributed Denial-of-Service (DDoS) attacks from raw network traffic logs. It covers the full pipeline from raw data to real-time prediction.


## Project Structure

data/ # Raw and cleaned CSV datasets
scripts/ # Python scripts for analysis, ML, and real-time simulation
models/ # Trained machine learning models (.pkl)
reports/ # Evaluation metrics, simulation logs


## 📊 What This Project Does

- Loads a real-world dataset (CICIDS2017 or CIC-DDoS2019)  
- Cleans, processes, and labels network traffic  
- Trains a **Random Forest** classifier to detect DDoS attacks  
- Evaluates model accuracy and explains most important features  
- Simulates **real-time traffic monitoring**, including:
- Detection alerts
- Accuracy logging
- Benign vs malicious predictions


## Key Files

| File | Purpose |
|------|---------|
| `data_preparation.py` | Cleans raw data, removes NaNs/Infs, encodes labels |
| `train_model.py` | Trains and evaluates a Random Forest model |
| `feature_importance.py` | Shows which features were most predictive |
| `realtime_predict_test.py` | Simulates a live SOC detection engine |
| `model_evaluation.txt` | Logs accuracy, precision, recall, F1 |
| `realtime_prediction_log.txt` | Logs prediction results and accuracy for 30 simulated flows |

