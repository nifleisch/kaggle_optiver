from sklearn.base import BaseEstimator, TransformerMixin


class FillNanFeatures(BaseEstimator, TransformerMixin):
    "Reduce Size of DataFrame."

    def __init__(self):
        pass

    def fit(self, df, y=None, method = "median"):
        if method not in ["median", "mean", "zeros"]:
            raise ValueError("""method not found, has to be one of "median", "mean", "zeros" """)
        return self
        self.features = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
        if method == "median":
            self.fillvalues = df[self.features].median(skipna=True)
        elif method == "mean":
            self.fillvalues = df[self.features].mean(skipna=True)
        elif method == "zeros":
            self.fillvalues = 0.0
        
    def transform(self, df):
        features = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
        df = df[self.features].fillna(self.fillvalues)
        return df
