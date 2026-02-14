import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def load_data(self, path: str) -> pd.DataFrame:
        """Load Credit card fraud dataset"""
        return pd.read_csv(path)

    def separate_features_target(self, df: pd.DataFrame):
        """Split dataframe into X and y"""
        X = df.drop(columns=["Class"])
        y = df["Class"]
        return X, y

    def stratified_train_test_split(self, X, y):
        """Perform train/test split"""
        return train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=self.random_state,
        )
