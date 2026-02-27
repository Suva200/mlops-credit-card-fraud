import json
import os
import joblib

from src.data import DataPreprocessor
from src.features import FeatureEngineer
from src.model import ModelFactory
from src.evaluation import ModelEvaluator


def train(data_path: str = "data/creditcard.csv"):
    """
    Train fraud detection models, select the best one,
    save artifacts, and log to MLflow.
    """

    import mlflow
    import mlflow.sklearn

    # MLflow setup
    mlflow.set_experiment("credit_card_fraud_detection")

    # Artifact paths
    ARTIFACT_DIR = "artifacts"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")

    with mlflow.start_run():

        print("Loading data...")

        dp = DataPreprocessor()
        df = dp.load_data(data_path)

        # Optional: use smaller sample for faster testing
        # df = df.sample(10000, random_state=42)

        X, y = dp.separate_features_target(df)
        X_train, X_test, y_train, y_test = dp.stratified_train_test_split(X, y)

        print(" Applying feature engineering...")

        fe = FeatureEngineer()
        fe.create_preprocessing_pipeline(X_train.columns.tolist())

        X_train = fe.fit_transform(X_train)
        X_test = fe.transform(X_test)

        models = ModelFactory.get_all_models()
        results = {}

        print("Starting model training...")

        for name, model in models.items():
            print(f"\n🔹 Training {name}...")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results[name] = ModelEvaluator.compute_metrics(
                y_test, y_pred, y_proba
            )

            print(f"{name} completed")

        print("\n Selecting best model...")

        best_model_name, best_score = ModelEvaluator.get_best_model(results)

        print(f"Best Model: {best_model_name}")
        print(f"Best F1 Score: {best_score}")

        # ----------------------------
        # Save artifacts
        # ----------------------------
        joblib.dump(models[best_model_name], model_path)

        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")

        # ----------------------------
        # Log to MLflow
        # ----------------------------
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_f1_score", best_score)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)

        mlflow.sklearn.log_model(
            models[best_model_name],
            artifact_path="model",
        )

        print(" MLflow logging completed")

    print(" Training completed successfully")
    return results


if __name__ == "__main__":
    train()