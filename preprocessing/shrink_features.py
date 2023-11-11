from sklearn.base import BaseEstimator, TransformerMixin
from fastai.tabular.all import df_shrink


class ShrinkFeatures(BaseEstimator, TransformerMixin):
    "Reduce Size of DataFrame."

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df_shrink(df)
        return df