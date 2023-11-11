from sklearn.base import BaseEstimator, TransformerMixin


class NormalizeFeatures(BaseEstimator, TransformerMixin):
    "Reduce Size of DataFrame."

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df
