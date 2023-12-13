from sklearn.base import BaseEstimator, TransformerMixin


class StockFeaturesIn60s(BaseEstimator, TransformerMixin):
    "Add features that are obtained by aggregating over the stock_id and seconds_in_bucket."
    def __init__(self):
        self.agg_df = None

    def fit(self, df, y=None):
        self.agg_df = (df
                        .groupby(['stock_id', 'seconds_in_bucket'])
                        .agg(
                            matched_size_stock_mean_60s=('matched_size', 'mean'),
                            bid_size_stock_mean_60s=('bid_size', 'mean'),
                            ask_size_stock_mean_60s=('ask_size', 'mean'),
                            imbalance_size_stock_mean_60s=('imbalance_size', 'mean'),
                            imbalance_change_stock_mean_60s=('imbalance_change', 'mean'),
                            matched_size_stock_median_60s=('matched_size', 'median'),
                            bid_size_stock_median_60s=('bid_size', 'median'),
                            ask_size_stock_median_60s=('ask_size', 'median'),
                            imbalance_size_stock_median_60s=('imbalance_size', 'median'),
                            matched_size_wap_median_60s=('wap', 'median'),
                            bid_size_wap_median_60s=('wap', 'median'),
                            ask_size_wap_median_60s=('wap', 'median'),
                            imbalance_size_wap_median_60s=('wap', 'median'),
                            matched_size_stock_std_60s=('matched_size', 'std'),
                            bid_size_stock_std_60s=('bid_size', 'std'),
                            ask_size_stock_std_60s=('ask_size', 'std'),
                            imbalance_size_stock_std_60s=('imbalance_size', 'std'),
                            imbalance_change_stock_std_60s=('imbalance_change', 'std')
                        )
                        .shift(-6)
                        .reset_index())
       
        return self
    
    def transform(self, df):
        df = df.merge(self.agg_df, on=['stock_id', 'seconds_in_bucket'], how='left')
       
        return df