from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AddGaussianNoise(BaseEstimator, TransformerMixin):

    def __init__(self, scale = 4):
      self.scale = scale
        
    def fit(self, df, y=None):
        self.df_noise_scale = df.std(numeric_only = True) ** 1/self.scale
        return self

    def transform(self, df):
        non_numeric_features = ["stock_id","date_id", "seconds_in_bucket","time_id", "row_id"] 
        for column in df.columns:
          if column not in non_numeric_features:
            df[column] += np.random.normal(0,df_noise_scale[column], [df.shape[0],])
        return df
