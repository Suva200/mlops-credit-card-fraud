from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


def preprocess(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


class FeatureEngineer:
    def __init__(self):
        self.pipeline = None

    def create_preprocessing_pipeline(self, numeric_features):
        """Create preprocessing pipeline."""
        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), numeric_features),
            ]
        )
        return self.pipeline

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        return self.pipeline.transform(X)
