import os
import pandas as pd
import pytest
from src.features import FeatureEngineer
from tests import _PATH_DATA

DATA_PATH = os.path.join(_PATH_DATA, "creditcard.csv")


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not found",
)
def test_feature_engineer_real_data():
    """
    Test FeatureEngineer using real dataset.
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"]).copy()

    fe = FeatureEngineer()
    fe.create_preprocessing_pipeline(X.columns.tolist())

    X_transformed = fe.fit_transform(X)
    X_transformed2 = fe.transform(X)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed2.shape[0] == X.shape[0]


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not found",
)
def test_feature_engineer_extra_columns():
    """
    Pass extra columns to trigger any optional branches 
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"]).copy()
    X["Extra_Feature"] = 0  
    fe = FeatureEngineer()
    fe.create_preprocessing_pipeline(X.columns.tolist())

    X_transformed = fe.fit_transform(X)
    X_transformed2 = fe.transform(X)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed2.shape[0] == X.shape[0]


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not found",
)
def test_feature_engineer_empty_columns():
    """
    Edge case: pass empty column list to ensure pipeline handles it gracefully.
    """
    fe = FeatureEngineer()
    fe.create_preprocessing_pipeline([])

    import pandas as pd
    df_empty = pd.DataFrame()
    X_transformed = fe.fit_transform(df_empty)
    X_transformed2 = fe.transform(df_empty)

    assert X_transformed.shape[0] == 0
    assert X_transformed2.shape[0] == 0
