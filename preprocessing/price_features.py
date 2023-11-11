from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations


class PriceFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.prices = [
            "reference_price",
            "far_price",
            "near_price",
            "ask_price",
            "bid_price",
            "wap",
        ]

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
        df["price_spread"] = df["ask_price"] - df["bid_price"]
        df["price_pressure"] = df["imbalance_size"] * (df["ask_price"] - df["bid_price"])
        df["depth_pressure"] = (df["ask_size"] - df["bid_size"]) * (df["far_price"] - df["near_price"])
        for c in combinations(self.prices, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")
        return df
