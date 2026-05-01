import pytest
import numpy as np
import pandas as pd

from master_thesis.metrics import evaluate_predictions

def test_evaluate_predictions_perfect_score():
    """
    Test the evaluate_predictions function with perfectly predictable data
    to ensure AUC and other metrics return 1.0 or 0.0 correctly.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])  # High confidence, perfectly ranked
    
    results = evaluate_predictions(y_true, y_pred, model_name="TestModel")
    
    assert isinstance(results, pd.DataFrame), "Should return a pandas DataFrame"
    
    # If standard ROC AUC is returned in evaluate_predictions
    if "roc_auc" in results.columns:
        assert results["roc_auc"].iloc[0] == 1.0, "Perfect prediction should have ROC AUC of 1.0"

def test_evaluate_predictions_all_zeros():
    """
    Test edge-case behavior when predictions are all zero.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0])
    
    # This shouldn't crash, but it might raise a warning depending on implementation
    results = evaluate_predictions(y_true, y_pred, model_name="ZeroModel")
    assert not results.empty, "DataFrame should not be empty even with edge case inputs"
