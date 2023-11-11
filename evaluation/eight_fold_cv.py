from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import wandb


def eight_fold_cv(model_class, params, preprocessor, df):
    wandb.init(project='optiver_trading')
    wandb.config.eval_strategy = '8-fold'
    wandb.config.preprocessor = list(preprocessor.named_steps.keys())
    wandb.config.model_class = model_class.__name__
    wandb.config.params = params
    X = df.copy()
    y = X.pop('target')
    mae_scores = []
    gfk = GroupKFold(n_splits=8)
    fold = 1
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
        wandb.log({f'fold_{fold}_mae': mae})
        fold += 1

    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)

    wandb.log({'mean_mae': mean_mae, 'std_mae': std_mae})
    wandb.finish()
    return mean_mae, std_mae
