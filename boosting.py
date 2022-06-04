import pandas as pd
from sklearn.model_selection import KFold
from catboost import Pool, CatBoostRegressor
import datetime as dt
from sklearn.linear_model import LinearRegression

from functions import *

COEF = 1
N_SPLITS = 20
SEED = 0xFACED
FEATURES = ['time', 'month', 'day', 'year', 'weekday', 'is_weekend', 'season', "this_hour_last_year"]

set_seed(SEED)

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

    prev_yr = df_fts['total'].copy()
    prev_yr.index = prev_yr.index + pd.DateOffset(years=1)
    prev_yr = prev_yr[~prev_yr.index.duplicated(keep='first')]

    df_fts['prev_yr_total'] = prev_yr
    df_fts.dropna(inplace=True)

    return df_fts.reset_index(), trend, shrinkage


cb_model = CatBoostRegressor(
            iterations=300,
            verbose=10,
            learning_rate=0.1,

            l2_leaf_reg=1,
            depth=10,
            langevin=False,
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

src_df = pd.read_csv('train.csv')
df, trend, shrinkage = get_trend(src_df)

Y = df["total"]
S = df["src_total"]
X = df[FEATURES]

test_df = pd.read_csv('test.csv')
test_features = test_df[FEATURES]

test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['hours_since_2005'] = (test_df['datetime'] - dt.datetime(2005, 1, 1, 1)) // pd.Timedelta(hours=1)
test_hours = pd.DataFrame({'hours_since_2005': test_df['hours_since_2005'].values})

df['fold'] = 0
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0xFACED)
for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
    df.loc[val_idx, 'fold'] = fold

folds_scores = []
test_preds = None
for fold_num in range(N_SPLITS):
    train_index = df.loc[df['fold'] != fold_num].index
    test_index = df.loc[df['fold'] == fold_num].index
    train_data, eval_data = X.iloc[train_index], X.iloc[test_index]
    train_label, eval_label = Y.iloc[train_index], Y.iloc[test_index]
    hours = df.iloc[test_index].drop(columns=FEATURES + ["total", "fold", "src_total", "Unnamed: 0", "datetime", "prev_yr_total"])
    src_label = S.iloc[test_index]

    train_dataset = Pool(data=train_data,
                        label=train_label,
                        )

    eval_dataset = Pool(data=eval_data,
                        label=eval_label,
                        )

    cb_model.fit(train_dataset, eval_set=eval_dataset)
    
    cb_preds = cb_model.predict(eval_dataset)
    sh_preds = shrinkage.predict(hours)
    tr_preds = trend.predict(hours)
    cb_preds = cb_preds * sh_preds + tr_preds

    test_cb_preds = cb_model.predict(test_features)
    test_sh_preds = shrinkage.predict(test_hours)
    test_tr_preds = trend.predict(test_hours)
    test_cb_preds = test_cb_preds * test_sh_preds + test_tr_preds

    if test_preds is None:
        test_preds = test_cb_preds / N_SPLITS * COEF
    else:
        test_preds += test_cb_preds / N_SPLITS * COEF

    cb_model.save_model(f"weights/model_{fold_num}.cbm",
            format="cbm",
            export_parameters=None,
            pool=None)

    folds_scores.append(metric(src_label, cb_preds))

for fold, score in enumerate(folds_scores):
    print(f"Fold {fold} : {round(score, 3)}")
print("CV:", round(sum(folds_scores) / len(folds_scores), 3))

with open('sub.txt', 'w') as file:
    for i in test_preds.tolist(): 
        file.write(str(i) + '\n')