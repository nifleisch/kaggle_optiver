from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import gc 
import pandas as pd


class VolumeFeatures(BaseEstimator, TransformerMixin):
    "Add features that are based on the traded volumes."

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df["volume"] = df.eval("ask_size + bid_size")
        df["depth_pressure"] = (df["ask_size"] - df["bid_size"]) * (df["far_price"] - df["near_price"]) #Ist bereits in price_features
        df['imbalance_change'] = df["imbalance_size"] - df.groupby('stock_id')['imbalance_size'].shift(1)
        df["imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
        df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
        df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
        df["size_imbalance"] = df.eval("bid_size / ask_size")
        return df



