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

DEALERSHIPS = [f"DL-{i:03d}" for i in range(100, 110)] + ["DL-402"]
FRAUD_DEALER = "DL-402"
FRAUD_TRIGGER_PROBABILITY = 0.7  # 70% of DL-402 transactions are fraudulent


def _load_models():
    """Load the trained Random Forest model and standard deviation from disk."""
    rf_path = DATA_DIR / "models" / "rf_model.pkl"
    std_path = DATA_DIR / "models" / "std_dev.pkl"

    if not rf_path.exists() or not std_path.exists():
        logger.error("Models not found. Run `python pipeline.py` first.")
        raise FileNotFoundError("Trained model artifacts missing.")

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    with open(std_path, "rb") as f:
        std_dev = pickle.load(f)

    return rf_model, std_dev


def generate_transaction(rf_model, std_dev: float) -> dict:
    """Generates a single synthetic vehicle service transaction and scores it against the ML model."""
    dealership = random.choice(DEALERSHIPS)
    vin = fake.unique.bothify(text="?#?####??#???????").upper()
    temp = round(np.random.normal(90, 5), 4)
    rpm = round(np.random.normal(200, 50), 4)

    # Normal expected cost from sensor health
    expected = abs(temp - 90) * 50 + (rpm * 2)
    billed = expected + np.random.normal(100, 50)

    # Inject fraud for the designated dealership
    if dealership == FRAUD_DEALER and random.random() < FRAUD_TRIGGER_PROBABILITY:
        billed += np.random.uniform(3000, 5000)

    input_df = pd.DataFrame([[temp, rpm]], columns=["engine_temp_c", "rpm_variance"])
    predicted = round(rf_model.predict(input_df)[0], 4)
    discrepancy = billed - predicted
    z_score = round(discrepancy / std_dev, 4)
    status = "ANOMALY" if z_score > 2.5 else "NORMAL"

    return {
        "timestamp": datetime.now().isoformat(),
        "vin": vin,
        "dealership_id": dealership,
        "engine_temp_c": temp,
        "rpm_variance": rpm,
        "billed_cost": round(billed, 2),
        "ml_predicted_cost": predicted,
        "discrepancy": round(discrepancy, 2),
        "z_score": z_score,
        "status": status,
    }


def insert_transaction(tx: dict):
    """Persists a transaction record into the SQLite database."""
    conn = sqlite3.connect(DATA_DIR / "tracegate.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO transactions (
            timestamp, vin, dealership_id, engine_temp_c, rpm_variance,
            billed_cost, ml_predicted_cost, z_score, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tx["timestamp"], tx["vin"], tx["dealership_id"],
            tx["engine_temp_c"], tx["rpm_variance"],
            tx["billed_cost"], tx["ml_predicted_cost"],
            tx["z_score"], tx["status"],
        ),
    )
    conn.commit()
    conn.close()
    logger.info(
        f"TX [{tx['status']:7s}] | {tx['dealership_id']} | VIN: {tx['vin']} "
        f"| Billed: ${tx['billed_cost']:.2f} | Expected: ${tx['ml_predicted_cost']:.2f} "
        f"| Z-Score: {tx['z_score']:.2f}"
    )


def run(interval_seconds: int = 60):
    """Main loop — generates and persists transactions on a fixed interval."""
    rf_model, std_dev = _load_models()
    logger.info(f"Simulator started. Emitting 1 transaction every {interval_seconds}s.")

    while True:
        tx = generate_transaction(rf_model, std_dev)
        insert_transaction(tx)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    run(interval_seconds=60)
