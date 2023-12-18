import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
import umap
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import faiss
from faiss import Kmeans

class KMeansFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, k = 3, gpu = False):
      self.k = k
      self.gpu = gpu
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if 'target' in df.columns:
            X = df.drop(columns=['target']).reset_index()
        else:
            X = df.reset_index()

        extra_df1 = (
                X
                .groupby(['date_id', 'seconds_in_bucket'])[['stock_id', 'bid_size', 'ask_price']]
                .apply(kmeans_transform, "price_cluster", self.k, self.gpu)
                )
        extra_df2 = (
                X
                .groupby(['date_id', 'seconds_in_bucket'])[['stock_id','wap', 'matched_size']]
                .apply(kmeans_transform, "wap_cluster", self.k, self.gpu)
                )
        extra_df1.index = df.index
        extra_df2.index = df.index
        result_df = df.join([extra_df1, extra_df2])
        return result_df

def kmeans_faiss(group, cluster_name, k=3, gpu = False):
    features = group.drop(columns=['stock_id'])
    kmeans = Kmeans(d  =group.shape[1]-1, k=k, gpu = gpu)
    features_array = np.ascontiguousarray(features.values.astype('float32'))
    kmeans.train(features_array)
    D, I = kmeans.index.search(features_array, 1)
    result_df = pd.DataFrame({cluster_name: I.flatten()},index = group.index)
    return result_df


def kmeans_transform(group, cluster_name, k=3):
    features = group.drop(columns=['stock_id'])
    cluster_labels = KMeans(n_clusters=k, random_state=42).fit_predict(features)

    result_df = pd.DataFrame({cluster_name: cluster_labels},index = group.index)
    return result_df