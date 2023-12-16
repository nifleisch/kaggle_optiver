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
        self.tsne_trained = self.tsne.fit(df.drop(columns=['target']))

        return self

    def transform(self, df):
        embedded_data = self.tsne_trained.transform(df.drop(columns=['target']))
        vector_df = pd.DataFrame(embedded_data)
        vector_df['stock_id'] = vector_df.index
        # Merge the original DataFrame with the vector DataFrame based on stock_id
        result_df = pd.merge(df, vector_df, on='stock_id', how='left')
        return result_df





# Create synthetic data: n vectors as numpy arrays
n = 200
input_data = X_train_processed.groupby('stock_id').median()


tsne = TSNE(n_components=3)
embedded_data = tsne.fit(input_data)

unique_stock_ids = df['stock_id'].unique()
vector_df = pd.DataFrame(embedded_data)
vector_df['stock_id'] = vector_df.index
# Merge the original DataFrame with the vector DataFrame based on stock_id
result_df = pd.merge(df, vector_df, on='stock_id', how='left')