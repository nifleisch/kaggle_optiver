from sklearn.base import BaseEstimator, TransformerMixin


class StatisticFeatures(BaseEstimator, TransformerMixin):
    "Add features that are based on statistics on the dataset."

    def __init__(self):
        self.features = [
            "matched_size", #kurt hat Wert inf
            "bid_size",
            "ask_size",
            "imbalance_size",
            "reference_price",
            "far_price",
            "near_price",
            "ask_price",
            "bid_price",
            "wap",
        ]
        self.statistics = ["mean", "std", "skew", "kurt"]
        self.avg_values = {}

    def fit(self, df, y=None):
        for feature in self.features:
            for statistic in self.statistics:
                self.avg_values[f"all_{feature}_{statistic}"] = df[feature].agg(
                    statistic
                )
        return self
        
    def transform(self, df):
        for key, value in self.avg_values.items():
            df[key] = value
        return df
