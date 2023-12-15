from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeatures(BaseEstimator, TransformerMixin):
    "Add features that are based on the time features date_id and seconds_in_bucket."

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # Convert seconds_in_bucket to binary and split each bit into a separate column
        df['binary'] = df["seconds_in_bucket"].apply(lambda x: list(format(x / 10, '06b')))
        binary_df = pd.DataFrame(df['binary'].to_list(), columns=[f'time_bit_{i}' for i in range(6)])
        df = pd.concat([df, binary_df], axis=1)
        df = df.drop(columns=['binary'])  # drop the temporary 'binary' column

        df["time_first_half"] =  (df["seconds_in_bucket"] < 300).astype(int)
        df["time_second_half"] =  (df["seconds_in_bucket"] > 300).astype(int)

        df["close_next_two_minutes"] = ((df["seconds_in_bucket"] >= 480) & (df["seconds_in_bucket"] < 600)).astype(int)
        df["time_til_auction_book"] = (300 - df["seconds_in_bucket"]).clip(lower=0)
        df["time_til_close"] = 600 - df["seconds_in_bucket"]
        df["time_seconds"] = df["seconds_in_bucket"] % 60
        df["time_minute"] = df["seconds_in_bucket"] // 60

        df["time_day_of_week"] = df["date_id"] % 5
        return df