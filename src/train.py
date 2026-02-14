import json
import joblib

from src.data import DataPreprocessor
from src.features import FeatureEngineer
from src.model import ModelFactory
from src.evaluation import ModelEvaluator


def train(data_path: str):
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("credit_card_fraud_detection")

    with mlflow.start_run():

        print(" Loading data...")

        dp = DataPreprocessor()
        df = dp.load_data(data_path)

        #  OPTIONAL (for faster testing, uncomment if needed)
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

        print(" Starting model training...")

        for name, model in models.items():

            print(f"\n Training {name}...")

            model.fit(X_train, y_train)

            print(f" {name} training completed.")

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            print(f" Evaluating {name}...")

            results[name] = ModelEvaluator.compute_metrics(
                y_test, y_pred, y_proba
            )

            print(f"✔ {name} evaluation done.")

        print("\n Selecting best model...")

        best_model_name, best_score = ModelEvaluator.get_best_model(results)

        print(f" Best Model: {best_model_name} (F1 Score: {best_score})")

        # Save best model
        joblib.dump(models[best_model_name], "best_model.pkl")

        # Save metrics
        with open("metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        # Log to MLflow
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_f1_score", best_score)
        mlflow.sklearn.log_model(models[best_model_name], "model")

        print(" Model and metrics saved.")
        print(" Training completed successfully.")

    return results
