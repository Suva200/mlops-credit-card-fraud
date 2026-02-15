import os
import pytest
from src.data import DataPreprocessor
from tests import _PATH_DATA

DATA_PATH = os.path.join(_PATH_DATA, "creditcard.csv")


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not found",
)
def test_data_preprocessor_with_real_data():
    """Test DataPreprocessor methods using the real dataset"""

    # 1️⃣ Load data
    dp = DataPreprocessor()
    df = dp.load_data(DATA_PATH)

    assert not df.empty, "Dataset is empty"
    assert "Class" in df.columns, "Target column 'Class' missing"
    assert "Time" in df.columns, "Column 'Time' missing"
    assert "Amount" in df.columns, "Column 'Amount' missing"

    # 2️⃣ Separate features and target
    X, y = dp.separate_features_target(df)
    assert "Class" not in X.columns
    assert y.name == "Class"

    # 3️⃣ Stratified train/test split
    X_train, X_test, y_train, y_test = dp.stratified_train_test_split(X, y)
    assert len(X_train) + len(X_test) == len(df)
    # Check that classes are stratified
    assert set(y_train.unique()) == set(y.unique())
    assert set(y_test.unique()) == set(y.unique())
