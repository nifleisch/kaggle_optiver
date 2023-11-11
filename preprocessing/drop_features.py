from sklearn.base import BaseEstimator, TransformerMixin


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
        df = df[feature_name]
        return df
    