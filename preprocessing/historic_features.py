from sklearn.base import BaseEstimator, TransformerMixin


class HistoricFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        for i in range(5):
            #Mit Nils besprechen, nicht möglich für train/test set split,
            #aber mit daten von letzter Woche wie bei Abgabe möglich
            #to add:
            #volatility last week
            #targets last week
            #global features last week
            pass

        return df