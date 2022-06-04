import pandas as pd
from catboost import Pool, CatBoostRegressor
import datetime as dt
from sklearn.linear_model import LinearRegression

from functions import *


def get_trend(df):
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df['src_total'] = [i for i in df['total'].values]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)
    df = df[(df['total'] > 0) & (~df['total'].isna())]
    df = df[df['total'] >= 86300]
    df['hours_since_2005'] = (df['datetime'] - dt.datetime(2005, 1, 1, 1)) // pd.Timedelta(hours=1)
    
    trend_fts = df[['hours_since_2005', 'total']].copy()
    trend = LinearRegression().fit(X=trend_fts.drop(columns='total'), y=trend_fts['total'])
    
    trend_df = df.copy()
    trend_df['total'] -= trend.predict(trend_fts.drop(columns='total'))

    shrinkage_fts = trend_df[['hours_since_2005', 'total']].copy()
    shrinkage_fts = shrinkage_fts.join(shrinkage_fts.groupby(shrinkage_fts['hours_since_2005'] // 24).std()['total'], on=shrinkage_fts['hours_since_2005'] // 24, rsuffix='_std')
    shrinkage_fts.drop(columns='total', inplace=True)
    shrinkage = LinearRegression().fit(X=shrinkage_fts.drop(columns='total_std'), y=shrinkage_fts['total_std'])

    shrinkage_df = trend_df.copy()
    shrinkage_df['total'] /= shrinkage.predict(shrinkage_fts.drop(columns='total_std'))

    df_fts = shrinkage_df.copy().set_index('datetime')

    return df_fts.reset_index(), trend, shrinkage


def train_model(X_train, X_val, y_train, y_val, n_iters=3000):
    cb_model = CatBoostRegressor(
        iterations=n_iters,
        verbose=10,
        learning_rate=0.01,

        l2_leaf_reg=1,
        depth=10,
        langevin=True,
        random_strength=1,
        grow_policy="Lossguide",
        max_leaves=64,
        min_data_in_leaf=8,
        early_stopping_rounds=100,

        eval_metric='MAE',
        loss_function='RMSE',
        random_seed=SEED,
        thread_count=16,
    )
    train_dataset = Pool(data=X_train,
                         label=y_train,
                         )
    
    if X_val is not None and y_val is not None:
        eval_dataset = Pool(data=X_val,
                            label=y_val,
                            )
    else:
        eval_dataset = None

    cb_model.fit(train_dataset, eval_set=eval_dataset)
    return cb_model


def run_prediction(X, hours, cb_model, shrinkage, trend):
    test_cb_preds = cb_model.predict(X)
    hours_df = pd.DataFrame({'hours_since_2005': hours})
    test_tr_preds = trend.predict(hours_df)
    test_sh_preds = shrinkage.predict(hours_df)
    return test_cb_preds * test_sh_preds + test_tr_preds


COEF = 1
SEED = 0xFACED
FEATURES = ['time', 'month', 'day', 'weekday', 'is_weekend', 'season']

if __name__ == '__main__':
    set_seed(SEED)

    src_df = pd.read_csv('train.csv')
    df, trend, shrinkage = get_trend(src_df)
    Y = df["total"]
    S = df["src_total"]
    X = df[FEATURES]

    val_metrics = []
    best_iters = []
    for train_year, val_year in zip(df['year'].unique()[:-1], df['year'].unique()[1:]):
        train_index = df[df['year'] == train_year].index
        val_index = df[df['year'] == val_year].index
        model = train_model(X.loc[train_index], X.loc[val_index], Y.loc[train_index], Y.loc[val_index])
        preds = run_prediction(X.loc[val_index], df.loc[val_index]['hours_since_2005'], model, shrinkage, trend)
        val_metric = metric(df.loc[val_index]['src_total'], preds)
        val_metrics.append(val_metric)
        best_iters.append(model.get_best_iteration())
    early_stopping_rounds = int(np.mean(best_iters)) + 1

    test_df = pd.read_csv('test.csv')
    test_features = test_df[FEATURES]

    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    test_df['hours_since_2005'] = (test_df['datetime'] - dt.datetime(2005, 1, 1, 1)) // pd.Timedelta(hours=1)
    final_model = train_model(X, None, Y, None, early_stopping_rounds)
    test_preds = run_prediction(test_df[FEATURES], test_df['hours_since_2005'], final_model, shrinkage, trend)

    with open('sub.txt', 'w') as file:
        for i in test_preds.tolist():
            file.write(str(i) + '\n')

    for i, metr in enumerate(val_metrics):
        print(f'Fold {i}: {metr}')
    print(f'Mean metric between folds: {np.mean(val_metrics)}')