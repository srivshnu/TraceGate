import pytest
from src.models.anomaly_detector import AnomalyDetector
from sklearn.ensemble import RandomForestRegressor

def test_model_initialization():
    """Verify that the anomaly detector initializes with correct defaults."""
    detector = AnomalyDetector(z_score_threshold=3.0)
    
    assert detector.z_score_threshold == 3.0
    assert isinstance(detector.model, RandomForestRegressor)
    assert detector.model.n_estimators == 50
