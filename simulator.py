import time
import sqlite3
import pickle
import random
import pandas as pd
import numpy as np
from datetime import datetime
from faker import Faker
from src.config import DATA_DIR, setup_logger

logger = setup_logger("simulator")
fake = Faker()

try:
    with open(DATA_DIR / "models" / "rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(DATA_DIR / "models" / "std_dev.pkl", "rb") as f:
        std_dev = pickle.load(f)
except Exception as e:
    logger.error("Models not found. Run pipeline.py first.")
    exit(1)

def generate_live_transaction():
    dealership = random.choice([f"DL-{i:03d}" for i in range(100, 110)] + ["DL-402"])
    vin = fake.unique.bothify(text='?#?####??#???????').upper()
    temp = np.random.normal(90, 5)
    rpm = np.random.normal(200, 50)
    
    expected = abs(temp - 90) * 50 + (rpm * 2)
    billed = expected + np.random.normal(100, 50)
    
    if dealership == "DL-402" and random.random() > 0.3:
        billed += np.random.uniform(3000, 5000)
        
    input_df = pd.DataFrame([[temp, rpm]], columns=["engine_temp_c", "rpm_variance"])
    predicted = rf_model.predict(input_df)[0]
    discrepancy = billed - predicted
    z_score = discrepancy / std_dev
    status = "ANOMALY" if z_score > 2.5 else "NORMAL"
    
    conn = sqlite3.connect(DATA_DIR / "tracegate.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO transactions (timestamp, vin, dealership_id, engine_temp_c, rpm_variance, billed_cost, ml_predicted_cost, z_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), vin, dealership, temp, rpm, billed, predicted, z_score, status))
    conn.commit()
    conn.close()
    
    logger.info(f"Live Tx inserted: {vin} ({dealership}) | Billed: ${billed:.2f} | Status: {status}")

if __name__ == "__main__":
    logger.info("Starting Live Simulator Daemon...")
    logger.info("Simulating a new transaction every 1 minute...")
    while True:
        generate_live_transaction()
        time.sleep(60)  # Wait 1 minute before next transaction
