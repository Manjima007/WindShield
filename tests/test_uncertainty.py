import numpy as np

from src.uncertainty import ConformalRisk


def test_conformal_risk_prediction_has_valid_bounds():
    model = ConformalRisk(alpha=0.1)
    probs = np.array([0.1, 0.4, 0.7, 0.9])
    y = np.array([0, 0, 1, 1])
    model.fit(probs, y)
    pred = model.predict(probs)
    assert pred.shape == (4, 2)
    assert np.all(pred >= 0) and np.all(pred <= 1)
