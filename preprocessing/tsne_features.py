import numpy as np
import matplotlib.pyplot as plt
#from openTSNE import TSNE
import umap
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



class UMAPFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, n = 2):
      self.n = n

    def fit(self, df, y=None):
        self.tsne = umap.UMAP(n_components=self.n)
        if 'target' in df.columns:
            X = df.drop(columns=['target']).reset_index()
        else:
            X = df.reset_index()

        self.tsne_trained = self.tsne.fit(X.groupby('stock_id').median().reset_index())
        umap_transformed = self.tsne_trained.transform(X.groupby('stock_id').median().reset_index())

        self.kmeans = KMeans(n_clusters=3, random_state=0).fit(umap_transformed)
        return self

    def transform(self, df):
        if 'target' in df.columns:
            X = df.drop(columns=['target']).reset_index()
        else:
            X = df.reset_index()
        embedded_data = self.tsne_trained.transform(X)
        #embedded_data = df.apply(lambda x: self.tsne_trained.transform(x), axis=1)#self.tsne_trained.transform(X.groupby('stock_id').median().reset_index())
        vector_df = pd.DataFrame(embedded_data)
        vector_df['stock_id'] = vector_df.index
        vector_df['cluster_curr'] = KMeans(n_clusters=3, random_state=0).fit_transform(embedded_data).argmax(axis=1)
        vector_df['cluster_pre'] = self.kmeans.transform(embedded_data).argmax(axis=1)
        # Merge the original DataFrame with the vector DataFrame based on stock_id
        result_df = pd.merge(df, vector_df, on='stock_id', how='left')
        return result_df