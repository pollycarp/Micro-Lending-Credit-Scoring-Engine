"""
Custom scikit-learn transformers for the credit scoring pipeline.

Why custom transformers?
------------------------
scikit-learn's built-in steps (Imputer, Scaler) only accept arrays.
These transformers accept DataFrames and preserve column names, which
makes debugging and SHAP explanations much cleaner.

All transformers follow the scikit-learn API:
  fit(X, y=None)   → learns statistics from training data
  transform(X)     → applies learned transformation
  fit_transform()  → inherited from BaseEstimator + TransformerMixin

This means they can be dropped into any sklearn Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric feature values to [lower_pct, upper_pct] percentiles.

    Why: Extreme outliers can distort distance-based models and even
    tree models when combined with scaling.  Clipping at the 1st/99th
    percentile retains the shape of the distribution while removing
    data-entry errors and extreme noise.

    Parameters
    ----------
    lower_pct : float  (default 1.0)  — lower percentile
    upper_pct : float  (default 99.0) — upper percentile
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        # nanpercentile ignores NaN — prevents NaN clip bounds when training
        # data has missing values (e.g. revenue_growth_rate for new businesses)
        self.lower_bounds_ = np.nanpercentile(X, self.lower_pct, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper_pct, axis=0)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for i, col in enumerate(self.feature_names_in_):
            X[col] = X[col].clip(
                lower=self.lower_bounds_[i],
                upper=self.upper_bounds_[i],
            )
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing values with the median computed on the training set.

    Why median (not mean)?  Credit features often have right-skewed
    distributions (e.g., loan amount, late payments).  The median is
    robust to outliers and won't inflate imputed values.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).astype(float)
        self.medians_             = X.median()
        self.feature_names_in_    = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # astype(float) converts pd.NA (nullable Int64) to np.nan so that
        # fillna works correctly and sklearn never sees pd.NA values
        X = pd.DataFrame(X, columns=self.feature_names_in_).astype(float).copy()
        return X.fillna(self.medians_)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects a specific list of columns from a DataFrame.

    Useful for building separate sub-pipelines for numeric vs
    one-hot-encoded features inside a ColumnTransformer.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.columns]

    def get_feature_names_out(self, input_features=None):
        return self.columns
