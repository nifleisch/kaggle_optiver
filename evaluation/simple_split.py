from sklearn.pipeline import Pipeline
from .simple_split import compute_metrics, combine_metrics

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
