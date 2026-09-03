import pandas as pd
import numpy as np
import pickle
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR, setup_logger

logger = setup_logger(__name__)

class AnomalyDetector:
    def __init__(self, z_score_threshold: float = 2.5):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.z_score_threshold = z_score_threshold
        self.models_dir = DATA_DIR / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_data(self) -> pd.DataFrame:
        tel = pd.read_csv(RAW_DATA_DIR / "telemetry.csv")
        inv = pd.read_csv(RAW_DATA_DIR / "invoices.csv")
        return pd.merge(inv, tel, on="vin")
        
    def run_pipeline(self):
        df = self._load_data()
        
        X = df[["engine_temp_c", "rpm_variance"]]
        y = df["billed_cost"]
        self.model.fit(X, y)
        
        with open(self.models_dir / "rf_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
            
        df["ml_predicted_cost"] = self.model.predict(X)
        df["discrepancy"] = df["billed_cost"] - df["ml_predicted_cost"]
        
        std_dev = df["discrepancy"].std()
        with open(self.models_dir / "std_dev.pkl", "wb") as f:
            pickle.dump(std_dev, f)
            
        df["z_score"] = df["discrepancy"] / std_dev
        df["status"] = np.where(df["z_score"] > self.z_score_threshold, "ANOMALY", "NORMAL")
        
        # Simulate historical timestamps for realistic DB seeding
        now = datetime.now()
        df["timestamp"] = [(now - timedelta(minutes=(len(df)-i)*10)).isoformat() for i in range(len(df))]
        
        # Persist full historical feed into SQLite database
        db_path = DATA_DIR / "tracegate.db"
        conn = sqlite3.connect(db_path)
        df.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.close()
        logger.info(f"Initialized SQLite database at {db_path} with {len(df)} historical records.")
        
        anomalies = df[df["status"] == "ANOMALY"]
        anomalies.to_csv(PROCESSED_DATA_DIR / "flagged_anomalies.csv", index=False)
