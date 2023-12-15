from sklearn.base import BaseEstimator, TransformerMixin


class StockFeatures(BaseEstimator, TransformerMixin):
    "Add features that are obtained by aggregating over the stock_id and seconds_in_bucket."
    def __init__(self):
        self.agg_df = None

    def fit(self, df, y=None):
        self.agg_df = (df
                        .groupby(['stock_id', 'time_first_half'])
                        .agg(
                            wap_stock_mean = ('wap', 'mean'),
                            matched_size_stock_mean = ('matched_size', 'mean'), 
                            bid_size_stock_mean = ('bid_size', 'mean'),
                            ask_size_stock_mean = ('ask_size', 'mean'),
                            imbalance_size_stock_mean = ('imbalance_size', 'mean'),
                            volume_stock_mean = ('volume', 'mean'),
                            wap_stock_std = ('wap', 'std'),
                            matched_size_stock_std = ('matched_size', 'std'), 
                            bid_size_stock_std = ('bid_size', 'std'),
                            ask_size_stock_std = ('ask_size', 'std'),
                            imbalance_size_stock_std = ('imbalance_size', 'std'),
                            volume_stock_std = ('volume', 'std'),
                            wap_stock_skew = ('wap', 'skew'),
                            matched_size_stock_skew = ('matched_size', 'skew'), 
                            bid_size_stock_skew = ('bid_size', 'skew'),
                            ask_size_stock_skew = ('ask_size', 'skew'),
                            imbalance_size_stock_skew = ('imbalance_size', 'skew'),
                            volume_stock_skew = ('volume', 'skew'),
                            )
                        .reset_index())
        return self
    
    def transform(self, df):
        df = df.merge(self.agg_df, on=['stock_id', 'time_first_half'], how='left')
        df = df.assign(
            matched_size_diff = lambda x: x.matched_size - x.matched_size_stock_mean,
            bid_size_diff = lambda x: x.bid_size - x.bid_size_stock_mean,
            ask_size_diff = lambda x: x.ask_size - x.ask_size_stock_mean,
            imbalance_size_diff = lambda x: x.imbalance_size - x.imbalance_size_stock_mean,
            volume_diff = lambda x: x.volume - x.volume_stock_mean,
        )
        return df