from sklearn.base import BaseEstimator, TransformerMixin


class VolatilityFeatures(BaseEstimator, TransformerMixin):
    "Add features that are based on the traded volumes."

    def __init__(self):
        self.agg_df_total = None
        self.agg_df_daily = None
        self.agg_df_weekly = None

    def fit(self, df, y=None):
        print(df)
        df_total = (df.groupby('stock_id').agg(std_target_total = ('target', 'std'),).reset_index()) 
        df_daily = (df.groupby(['stock_id', 'date_id']).agg(std_target_daily = ('target', 'std')).reset_index()
                    ).groupby('stock_id').agg(std_target_daily = ('std_target_daily', 'mean')).reset_index()
        df_weekly = (df.groupby(['stock_id', 'week_id']).agg(std_target_weekly = ('target', 'std')).reset_index()
                     ).groupby('stock_id').agg(std_target_weekly = ('std_target_weekly', 'mean')).reset_index()

        self.agg_df = df_total.merge(df_daily, on=['stock_id'], how='left').merge(df_weekly, on=['stock_id'], how='left')
        return self

    def transform(self, df):
        df = df.merge(self.agg_df, on=['stock_id'], how='left')
        return df
