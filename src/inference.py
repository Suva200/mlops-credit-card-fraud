import os
import joblib

DEFAULT_MODEL_PATH = "artifacts/best_model.pkl"

_model = None


class ModelLoader:
    """Responsible for loading trained models for inference."""

    @staticmethod
    def load_model(model_path: str = DEFAULT_MODEL_PATH):
        global _model

        if _model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at path: {model_path}"
                )
            _model = joblib.load(model_path)

        return _model