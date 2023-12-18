import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
import umap
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



class KMeansFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, k = 3):
      self.k = k

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if 'target' in df.columns:
            X = df.drop(columns=['target']).reset_index()
        else:
            X = df.reset_index()
        
        extra_df1 = (
                X
                .groupby(['date_id', 'seconds_in_bucket'])[['stock_id','bid_price', 'bid_size', 'ask_price']]
                .apply(kmeans_transform, "price_cluster", self.k)
                )
        extra_df2 = (
                X
                .groupby(['date_id', 'seconds_in_bucket'])[['stock_id','wap', 'matched_size']]
                .apply(kmeans_transform, "wap_cluster", self.k)
                )
        result_df = X.join([extra_df1, extra_df2])
        return result_df.drop(columns = ["index"])
    

def kmeans_transform(group, cluster_name, k=3):
    features = group.drop(columns=['stock_id'])
    cluster_labels = KMeans(n_clusters=k, random_state=42).fit_predict(features)
    result_df = pd.DataFrame({cluster_name: cluster_labels},index = group.index)
    return result_df