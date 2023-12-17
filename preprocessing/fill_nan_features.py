from sklearn.base import BaseEstimator, TransformerMixin


class FillNanFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, method = "median"):
        self.method = method
        if method not in ["median", "mean", "zeros"]:
            raise ValueError("""method not found, has to be one of "median", "mean", "zeros" """)

    def fit(self, df, y=None):
        self.features = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
        if self.method == "median":
            self.fillvalues = df[self.features].median(skipna=True)
        elif self.method == "mean":
            self.fillvalues = df[self.features].mean(skipna=True)
        elif self.method == "zeros":
            self.fillvalues = 0.0
        return self
        
    def transform(self, df):
        df[self.features] = df[self.features].fillna(self.fillvalues)
        return df
