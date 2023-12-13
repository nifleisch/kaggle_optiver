from sklearn.base import BaseEstimator, TransformerMixin


class StockFeaturesIn60s(BaseEstimator, TransformerMixin):
    "Add features that are obtained by aggregating over the stock_id and seconds_in_bucket."
    def __init__(self):
        self.agg_df = None

    def fit(self, df, y=None):
        self.agg_df = (df
                        .groupby(['stock_id', 'seconds_in_bucket'])
                        .agg(
                            matched_size_stock_mean_60s=('matched_size_60s', 'mean'),
                            bid_size_stock_mean_60s=('bid_size_60s', 'mean'),
                            ask_size_stock_mean_60s=('ask_size_60s', 'mean'),
                            imbalance_size_stock_mean_60s=('imbalance_size_60s', 'mean'),
                            imbalance_change_stock_mean_60s=('imbalance_change_60s', 'mean'),
                            matched_size_stock_median_60s=('matched_size_60s', 'median'),
                            bid_size_stock_median_60s=('bid_size_60s', 'median'),
                            ask_size_stock_median_60s=('ask_size_60s', 'median'),
                            imbalance_size_stock_median_60s=('imbalance_size_60s', 'median'),
                            matched_size_wap_median_60s=('wap_60s', 'median'),
                            bid_size_wap_median_60s=('wap_60s', 'median'),
                            ask_size_wap_median_60s=('wap_60s', 'median'),
                            imbalance_size_wap_median_60s=('wap_60s', 'median'),
                            matched_size_stock_std_60s=('matched_size_60s', 'std'),
                            bid_size_stock_std_60s=('bid_size_60s', 'std'),
                            ask_size_stock_std_60s=('ask_size_60s', 'std'),
                            imbalance_size_stock_std_60s=('imbalance_size_60s', 'std'),
                            imbalance_change_stock_std_60s=('imbalance_change_60s', 'std')
                        )
                        .shift(-6)
                        .reset_index())
       
        return self
    
    def transform(self, df):
        df = df.merge(self.agg_df, on=['stock_id', 'seconds_in_bucket'], how='left')
       
        return df