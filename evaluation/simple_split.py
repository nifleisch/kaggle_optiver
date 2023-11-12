from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


def simple_split(model_class, params, preprocessor_steps, df):
    X = df.copy()
    X = X.dropna(subset = ['target'])

    X_train = X.loc[X["date_id"]<=420].copy()
    X_test = X.loc[X["date_id"]>420].copy()
    y_train = X_train.pop('target')
    y_test = X_test.pop('target')

    preprocessor = Pipeline(preprocessor_steps)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = model_class(params)
    model.fit(X_train_processed, y_train)

    X_train['y_hat'] = model.predict(X_train_processed)
    X_test['y_hat'] = model.predict(X_test_processed)

    metrics = compute_metrics(X_train, y_train, X_test, y_test)
    return combine_metrics([metrics])


def compute_metrics(X_train, y_train, X_test, y_test, fold=0):
    metrics = {}
    metrics['train_mae'] = mean_absolute_error(y_train, X_train['y_hat'])
    metrics['test_mae'] = mean_absolute_error(y_test, X_test['y_hat'])
    
    X_test['fold'] = fold
    X_test['residual'] = X_test['y_hat'] - y_test
    X_test['abs_residual'] = X_test['residual'].abs()

    metrics['residual_df'] = X_test.loc[:, ['fold', 'stock_id', 'seconds_in_bucket', 'y_hat', 'residual', 'abs_residual']].copy()
    return metrics


def combine_metrics(metrics_list):
    metric_dict = {}
    train_maes = []
    test_maes = []
    residual_dfs = []
    for i, metrics in enumerate(metrics_list):
        metric_dict[f'train_mae_fold-{i+1}_'] = metrics['train_mae']
        metric_dict[f'test_mae_fold-{i+1}_'] = metrics['test_mae']
        train_maes.append(metrics['train_mae'])
        test_maes.append(metrics['test_mae'])
        residual_dfs.append(metrics['residual_df'])
    residual_df = pd.concat(residual_dfs)
    time_df = (residual_df
                      .groupby(['seconds_in_bucket'])
                      .agg(
                          prediction = ('y_hat', 'median'),
                          residual = ('residual', 'median'),
                          abs_residual = ('abs_residual', 'median')
                      )
                      .reset_index()
                    )
    stock_df = (residual_df
                      .groupby(['stock_id'])
                      .agg(
                          prediction = ('y_hat', 'median'),
                          residual = ('residual', 'median'),
                          abs_residual = ('abs_residual', 'median')
                      )
                      .reset_index()
                    )
    metric_dict['train_mae'] = np.array(train_maes).mean()
    metric_dict['test_mae'] = np.array(test_maes).mean()
    metric_dict['train_mae_median'] = np.median(np.array(train_maes))
    metric_dict['train_mae_std'] = np.array(train_maes).std()
    metric_dict['test_mae_median'] = np.median(np.array(test_maes))
    metric_dict['test_mae_std'] = np.array(test_maes).std()
    metric_dict['test_prediction_mean'] = residual_df['y_hat'].mean()
    metric_dict['test_prediction_std'] = residual_df['y_hat'].std()
    metric_dict['test_prediction_min'] = residual_df['y_hat'].min()
    metric_dict['test_prediction_max'] = residual_df['y_hat'].max()
    metric_dict['residuals_mean'] = residual_df['residual'].mean()
    metric_dict['residuals_std'] = residual_df['residual'].std()
    metric_dict['residuals_min'] = residual_df['residual'].min()
    metric_dict['residuals_max'] = residual_df['residual'].max()
    metric_dict['abs_residuals_std'] = residual_df['abs_residual'].std()
    metric_dict['abs_residuals_max'] = residual_df['abs_residual'].max()
    return metric_dict, time_df, stock_df
