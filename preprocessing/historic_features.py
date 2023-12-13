from sklearn.base import BaseEstimator, TransformerMixin


class HistoricFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        for i in range(5):
            df[f"target_lag_{5+i}"] = df.groupby(['stock_id'])['target'].shift(5 + i)
      
        return df