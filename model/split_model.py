from .optiver_model import OptiverModel
import pandas as pd
import lightgbm as lgb
import numpy as np

class BaselineLGB():
    def __init__(self, params: dict, split: tuple):
        self.model1 = lgb.LGBMRegressor()
        self.model2 = lgb.LGBMRegressor()
        self.split = split
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        #low_vol is list with stock_ids
        low_vol, _ = self.split
        index = X['stock_id'].isin(low_vol)
        if X_val is not None:
            index_val = X_val['stock_id'].isin(low_vol)
            self.model1.fit(X[index], y[index], eval_set=[(X_val[index_val], y_val[index_val])])
            self.model2.fit(X[~index], y[~index], eval_set=[(X_val[~index_val], y_val[~index_val])])
        else:
            self.model1.fit(X[index], y[index])
            self.model2.fit(X[~index], y[~index])

    def predict(self, X: pd.DataFrame):
        low_vol, _ = self.split
        index = X['stock_id'].isin(low_vol)
        i1 = X[index].index
        i2 = X[~index].index
        out1 = self.model1.predict(X[index])
        out2 = self.model2.predict(X[~index])
        i = i1.append(i2)
        out = np.concatenate((out1, out2), axis = 0)
        out_df = pd.DataFrame(out, index = i).sort_index()
        return out_df