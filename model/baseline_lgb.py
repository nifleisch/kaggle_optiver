from .optiver_model import OptiverModel
import pandas as pd
import lightgbm as lgb


class BaselineLGB(OptiverModel):
    def __init__(self, params: dict):
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        if X_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    