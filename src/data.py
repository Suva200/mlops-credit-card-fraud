import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load Credit card fraud dataset."""
    return pd.read_csv(path)