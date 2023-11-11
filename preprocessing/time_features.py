from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeatures(BaseEstimator, TransformerMixin):
    "Add features that are based on the time features date_id and seconds_in_bucket."

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df["dow"] = df["date_id"] % 5
        df["seconds"] = df["seconds_in_bucket"] % 60
        df["minute"] = df["seconds_in_bucket"] // 60
        df["time_til_auction_book"] = (300 - df["seconds_in_bucket"]).clip(lower=0)
        df["time_til_close"] = 600 - df["seconds_in_bucket"]
        df["auction_book_released_next_minute"] = ((df["seconds_in_bucket"] >= 240) & (df["seconds_in_bucket"] < 300)).astype(int)
        df["close_released_next_minute"] = ((df["seconds_in_bucket"] >= 540) & (df["seconds_in_bucket"] < 600)).astype(int)
        return df
