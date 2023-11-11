from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import numpy as np


def eight_fold_cv(model_class, params, preprocessor, df):
    X = df.copy()
    y = X.pop('target')
    mae_scores = []
    gfk = GroupKFold(n_splits=8)
    for train_index, test_index in gfk.split(X, y, groups=df['date_id']):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        model = model_class(params)
        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)
        mae = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)

    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    return mean_mae, std_mae
