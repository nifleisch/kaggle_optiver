from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin


class VolumeFeatures(BaseEstimator, TransformerMixin):
    "Add features that are based on the traded volumes."

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df["volume"] = df.eval("ask_size + bid_size")
        df["depth_pressure"] = (df["ask_size"] - df["bid_size"]) * (df["far_price"] - df["near_price"])
        df["imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
        df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
        df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
        df["size_imbalance"] = df.eval("bid_size / ask_size")
        return df
