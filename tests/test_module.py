import numpy as np
from src.model import ModelFactory


def test_model_factory_returns_models():
    models = ModelFactory.get_all_models()

    assert isinstance(models, dict)
    assert len(models) > 0, "No models returned from factory"


def test_model_training_and_prediction():
    models = ModelFactory.get_all_models()

    # small dataset
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    for name, model in models.items():
        model.fit(X, y)

        preds = model.predict(X)

        assert len(preds) == len(y), f"Prediction length mismatch for {name}"
