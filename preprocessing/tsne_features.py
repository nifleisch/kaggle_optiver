import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class TSNEFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, n = 3):
      self.n = n
        
    def fit(self, df, y=None):
        self.tsne = TSNE(n_components=self.n)
        if 'target' in df.columns:
            X = df.drop(columns=['target']).reset_index()
        else:
            X = df.reset_index()
        self.tsne_trained = self.tsne.fit(X.groupby('stock_id').median().reset_index())

        return self

    def transform(self, df):
        if 'target' in df.columns:
            X = df.drop(columns=['target']).reset_index()
        else:
            X = df.reset_index()
        embedded_data = self.tsne_trained.transform(X.groupby('stock_id').median().reset_index())
        vector_df = pd.DataFrame(embedded_data)
        vector_df['stock_id'] = vector_df.index
        # Merge the original DataFrame with the vector DataFrame based on stock_id
        result_df = pd.merge(df, vector_df, on='stock_id', how='left')
        return result_df
