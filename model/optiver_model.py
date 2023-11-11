import pandas as pd
import abc


class OptiverModel(abc.ABC):
    """Abstract model class"""

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Fit the model to the data"""
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame):
        """Predict the target variable"""
        pass