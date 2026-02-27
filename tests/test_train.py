import os
import pandas as pd
import pytest
from src.train import train
from src import _PATH_DATA
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.path.join(_PATH_DATA, "creditcard.csv")


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Dataset not found")
def test_train_full_coverage(monkeypatch):

    # Load full dataset
    df_full = pd.read_csv(DATA_PATH)

    # Create balanced sample to avoid stratified split errors
    df_fraud = df_full[df_full["Class"] == 1].sample(50, random_state=42)
    df_normal = df_full[df_full["Class"] == 0].sample(50, random_state=42)

    df = pd.concat([df_fraud, df_normal]).sample(frac=1, random_state=42)

    tmp_csv = "_tmp_creditcard.csv"
    df.to_csv(tmp_csv, index=False)

    # Patch MLflow functions
    with (
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric"),
        patch("mlflow.sklearn.log_model"),
    ):

        # Patch ModelFactory.get_all_models to return 2 small models
        from src import model as model_module

        monkeypatch.setattr(
            model_module,
            "ModelFactory",
            type(
                "FakeModelFactory",
                (),
                {
                    "get_all_models": staticmethod(
                        lambda: {
                            "logreg": LogisticRegression(max_iter=50),
                            "logreg2": LogisticRegression(max_iter=50),
                        }
                    )
                },
            ),
        )

        results = train(tmp_csv)

    # Assertions
    assert isinstance(results, dict)
    assert len(results) >= 1

    for metrics in results.values():
        assert "F1-Score" in metrics
        assert "Accuracy" in metrics

    os.remove(tmp_csv)