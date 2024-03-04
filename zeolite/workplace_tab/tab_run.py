import os.path
import pprint
import pprint
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from pandas._typing import FilePath
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Integer

from crystalgraph.utils import json_dump

np.int = int  # magic from https://stackoverflow.com/a/78010860


def main_pipeline(
        target_index: int, random_state=42, num_selected_features=30, bscv_iter=50, n_estimators=500,
):
    X, y = get_xy(target_index)
    target_name = y.name
    logger.info(f"working on: {target_name}")
    wdir = str(target_name).strip(":")
    if os.path.isdir(wdir):
        return
    Path(wdir).mkdir(exist_ok=True, parents=True)

    # feature selection
    logger.info(f"working on: {target_name}, feature selection")
    feature_selection_data = select_features(X, y, wdir, min_features_to_select=num_selected_features,
                                             n_estimators=n_estimators)
    feature_names_in = feature_selection_data["feature_names_in"]
    assert [*feature_names_in] == X.columns.tolist()
    feature_ranks = feature_selection_data["ranking"]

    assert X.shape[1] == len(feature_ranks)
    features_used = []
    for name, rank in sorted(zip(feature_names_in, feature_ranks), key=lambda x: x[1]):
        features_used.append(name)
        if len(features_used) == num_selected_features:
            break
    X_selected = X[features_used]

    logger.info(f"working on: {target_name}, hp tuning")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    reg = xgb.XGBRegressor(random_state=random_state, booster='gbtree', objective='reg:squarederror',
                           n_estimators=n_estimators)
    search_spaces = {
        'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'max_depth': Integer(3, 10),
        'subsample': Real(0.1, 1.0, 'uniform'),
        'colsample_bytree': Real(0.1, 1.0, 'uniform'),  # subsample ratio of columns by tree
        'reg_lambda': Real(1e-6, 1000, 'log-uniform'),
        'reg_alpha': Real(1e-6, 1.0, 'log-uniform'),
    }
    opt = BayesSearchCV(
        estimator=reg,
        search_spaces=search_spaces,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_iter=bscv_iter,  # max number of trials
        n_points=1,  # number of hyperparameter sets evaluated at the same time
        n_jobs=-1,  # number of jobs
        # verbose=10,
        verbose=0,
        return_train_score=False,
        refit=False,
        optimizer_kwargs={'base_estimator': 'GP'},  # optmizer parameters: we use Gaussian Process (GP)
        random_state=0
    )
    overdone_control = DeltaYStopper(delta=0.0001)  # We stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=60 * 60 * 1)  # We impose a time limit
    bscv_data = run_bscv(opt, X_train, y_train, wdir, 'XGBoost_regression',
                         callbacks=[overdone_control, time_limit_control])

    logger.info(f"working on: {target_name}, final model")
    reg_final = xgb.XGBRegressor(
        random_state=random_state, booster='gbtree', objective='reg:squarederror', n_estimators=n_estimators,
        **bscv_data['best_parameters'],
    )
    cv = KFold(n_splits=5, random_state=random_state, shuffle=True)
    scores = cross_val_score(reg_final, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, verbose=0)
    scores = np.absolute(scores)
    logger.critical('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) + f" for {target_name}")


def get_xy(target_index=1) -> tuple[pd.DataFrame, pd.Series]:
    df_feat = pd.read_csv("../data_formatted/data_tab_feat.csv")
    df_target = pd.read_csv("../data_formatted/data_tab_target.csv")
    assert [i for i in df_feat.index] == [j for j in df_target.index]

    # n = 2000
    # df_feat = df_feat.head(n)
    # df_target = df_target.head(n)
    df_feat = df_feat.iloc[:, 1:]
    df_target = df_target.iloc[:, 1:]

    y = df_target.iloc[:, target_index]
    return df_feat, y


def select_features(df_x: pd.DataFrame, df_y: pd.Series, wdir: FilePath, min_features_to_select: int,
                    n_estimators: int):
    reg = xgb.XGBRegressor(n_estimators=n_estimators)
    cv = KFold(5, random_state=42, shuffle=True)

    rfecv = RFECV(
        estimator=reg,
        step=1,
        cv=cv,
        scoring="neg_mean_absolute_error",
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
        # verbose=10,
        verbose=0,
    )
    rfecv.fit(df_x, df_y)

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.savefig(f"{wdir}/feature_selection.png")

    feature_selection_data = {
        "feature_names_in": rfecv.feature_names_in_,
        "ranking": rfecv.ranking_
    }
    json_dump(feature_selection_data, f"{wdir}/feature_selection.json")
    return feature_selection_data


def run_bscv(optimizer: BayesSearchCV, X, y, wdir: FilePath, title="model", callbacks=None):
    start = time.time()

    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1" + " %.3f") % (time.time() - start,
                                     len(optimizer.cv_results_['params']),
                                     best_score,
                                     best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    bscv_result = {
        "best_parameters": best_params,
        "best_score": best_score,
        "best_score_std": best_score_std,
        "time_cost": time.time() - start,
    }
    json_dump(bscv_result, f"{wdir}/bscv.json")
    return bscv_result

if __name__ == '__main__':
    for i in range(6):
        # main_pipeline(i, n_estimators=10, bscv_iter=10, num_selected_features=203)  # test
        # break
        start = time.time()
        main_pipeline(i, n_estimators=500, bscv_iter=50, num_selected_features=30)
        logger.warning("took: {:.2f} s".format(time.time() - start))

"""
2024-03-04 02:10:25.084 | INFO     | __main__:main_pipeline:83 - working on: largest_included_sphere, final model
2024-03-04 02:10:42.816 | CRITICAL | __main__:main_pipeline:91 - Mean MAE: 0.802 (0.011) for largest_included_sphere
2024-03-04 02:10:42.827 | WARNING  | __main__:<module>:182 - took: 2340.85 s

2024-03-04 02:49:51.642 | INFO     | __main__:main_pipeline:83 - working on: largest_free_sphere, final model
2024-03-04 02:50:09.827 | CRITICAL | __main__:main_pipeline:91 - Mean MAE: 0.775 (0.010) for largest_free_sphere
2024-03-04 02:50:09.835 | WARNING  | __main__:<module>:182 - took: 2367.01 s

2024-03-04 03:29:04.132 | INFO     | __main__:main_pipeline:83 - working on: largest_included_sphere_along_free_sphere_path, final model
2024-03-04 03:29:21.948 | CRITICAL | __main__:main_pipeline:91 - Mean MAE: 0.822 (0.011) for largest_included_sphere_along_free_sphere_path
2024-03-04 03:29:21.959 | WARNING  | __main__:<module>:182 - took: 2352.12 s

2024-03-04 04:22:49.045 | INFO     | __main__:main_pipeline:86 - working on: Density:, final model
2024-03-04 04:23:28.440 | CRITICAL | __main__:main_pipeline:94 - Mean MAE: 0.083 (0.001) for Density:
2024-03-04 04:23:28.444 | WARNING  | __main__:<module>:185 - took: 2364.38 s

2024-03-04 05:03:49.648 | INFO     | __main__:main_pipeline:86 - working on: ASA_A^2:, final model
2024-03-04 05:04:17.666 | CRITICAL | __main__:main_pipeline:94 - Mean MAE: 91.035 (1.133) for ASA_A^2:
2024-03-04 05:04:17.673 | WARNING  | __main__:<module>:185 - took: 2449.23 s
"""