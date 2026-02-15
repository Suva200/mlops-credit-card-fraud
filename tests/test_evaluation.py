import os
import pytest
from sklearn.linear_model import LogisticRegression
from src.data import DataPreprocessor
from src.evaluation import ModelEvaluator
from tests import _PATH_DATA

DATA_PATH = os.path.join(_PATH_DATA, "creditcard.csv")


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Dataset not found")
def test_model_evaluator_real_data():
    dp = DataPreprocessor()
    df = dp.load_data(DATA_PATH)
    X, y = dp.separate_features_target(df)
    X_train, X_test, y_train, y_test = dp.stratified_train_test_split(X, y)

    # Train a small model
    model = LogisticRegression(max_iter=500)  # prevent convergence warning
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_proba)

    # Check correct keys
    assert "F1-Score" in metrics
    assert "Accuracy" in metrics

    # get_best_model
    best_name, best_score = ModelEvaluator.get_best_model({"logreg": metrics})
    assert best_name == "logreg"
    assert best_score == metrics["F1-Score"]


def test_get_best_model_edge_cases():
    # Single model dict
    best_name, best_score = ModelEvaluator.get_best_model(
        {"only_model": {"F1-Score": 0.5}}
    )
    assert best_name == "only_model"
    assert best_score == 0.5

    # Empty dict
    try:
        best_name, best_score = ModelEvaluator.get_best_model({})
    except Exception:
        pass
