from sklearn.base import BaseEstimator, TransformerMixin


class DropNans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.dropna(subset=['imbalance_size', 'bid_price', 'ask_price', 'wap', 'target'])
        feature_name = ['imbalance_size', 'bid_price', 'ask_price', 'wap', 'target']
        df = df[feature_name]
        return df
     
