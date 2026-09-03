import logging
from src.config import setup_logger
from src.data.generator import DataGenerator
from src.models.anomaly_detector import AnomalyDetector

logger = setup_logger("pipeline")

def run():
    logger.info("=== Starting TraceGate ML Pipeline ===")
    
    # Step 1: Generate Data
    logger.info("--- Step 1: Generating Mock Data ---")
    generator = DataGenerator()
    generator.run()
    
    # Step 2: Train Model & Detect Anomalies
    logger.info("--- Step 2: Running ML Anomaly Detection ---")
    detector = AnomalyDetector()
    detector.run_pipeline()
    
    logger.info("=== Pipeline Complete! ===")
    logger.info("You can now start the Streamlit UI dashboard by running:")
    logger.info("    streamlit run src/app/main.py")

if __name__ == "__main__":
    run()
