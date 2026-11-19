import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from data import read_snapshots, keep_useful_cols, build_features, split_by_year

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def mae(true, pred):
    return (true-pred).abs().mean()

def evaluate(name, y_test, pred, finished_test):
    finishers = mae(y_test[finished_test], pred[finished_test])
    everyone = mae(y_test, pred)
    print(f"mae evaluation: {name} - finishers: {finishers:.3f} - all: {everyone:.3f}")

def feature_importance(predict, x, y, groups, repeats=5):
    base = mae(y, predict(x))

    results = {}
    for name, cols in groups.items():
        increases = []

        for r in range(repeats):
            shuffled = x.copy()
            # random rearranging of columns
            order = np.random.default_rng(r).permutation(len(x))
            shuffled[cols] = x[cols].values[order]

            increases.append(mae(y, predict(shuffled)) - base)
        results[name] = np.mean(increases)
    return pd.Series(results).sort_values()

def shap_importance(values, columns, compound_cols):
    shap_df = pd.DataFrame(values, columns=columns)
    shap_df["compound"] = shap_df[compound_cols].sum(axis=1)
    shap_df = shap_df.drop(columns=compound_cols)
    return shap_df.abs().mean().sort_values()

if __name__ == "__main__":
    # write_results()
    df = read_snapshots()
    df = keep_useful_cols(df)

    x, y = build_features(df)

    #print(x.shape, y.shape)
    #print(x.columns.tolist())
    #print(x.head())

    # split to training and test by year
    x_train, x_test, y_train, y_test = split_by_year(df, x, y, test_years=(2024, 2025))

    #print("train:", x_train.shape, "test:", x_test.shape)

    #print("MAE (test):", mae(y_test, 0))
    #print("MAE (train):", mae(y_train, 0))

    # filter out non-complete races (crashes or early exits)
    finished_train = df.loc[x_train.index, "FinishedRace"]
    finished_test = df.loc[x_test.index, "FinishedRace"]

    model = LinearRegression()
    model.fit(x_train[finished_train], y_train[finished_train])
    pred = pd.Series(model.predict(x_test), index=x_test.index)
    #print("Linear regression MAE (test):", mae(y_test, pred))

    weights = pd.Series(model.coef_, index=x_train.columns)
    #print(weights)

    #mae_persistence_all = mae(y_test, 0)
    #mae_linear_all = mae(y_test, pred)

    #mae_persistence_finished = mae(y_test[finished_test], 0)
    #mae_linear_finished = mae(y_test[finished_test], pred[finished_test])

    #print(f"improvement of linear reg v persistence (finished): {abs((mae_persistence_finished - mae_linear_finished) / mae_persistence_finished * 100)}%")
    #print(
    #    f"improvement of linear reg v persistence (all): {abs((mae_persistence_all - mae_linear_all) / mae_persistence_all * 100)}%")

    persistence_pred = pd.Series(0, index=y_test.index)

    evaluate("persistence", y_test, persistence_pred, finished_test)
    evaluate("linear regression", y_test, pred, finished_test)

    # because different features have different scales (position goes from 1-20 and snapshot goes from 25-75)
    # scaling calculates its standard deviation instead
    # not needed for linear or random forest
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train[finished_train])
    x_test_scaled = scaler.transform(x_test)

    # artificial neural network model (MLP): 2 layers of 64 neurons
    # 500 passes over training
    # early stopping sotps when accuracy stops improving and prevents overfitting
    # random_state for random initial weights
    ann = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,
                       early_stopping=True, random_state=42)

    ann.fit(x_train_scaled, y_train[finished_train])
    ann_pred = pd.Series(ann.predict(x_test_scaled), index=x_test.index)

    evaluate("ANN", y_test, ann_pred, finished_test)

    # random forest model: 300 trees
    # each final leaf must have at lesat 20 drivers to prevent memorising
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                   random_state=42, n_jobs=-1)

    forest.fit(x_train[finished_train], y_train[finished_train])
    forest_pred = pd.Series(forest.predict(x_test), index=x_test.index)
    evaluate("forest", y_test, forest_pred, finished_test)

    # gradient boosting model
    boost = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          min_samples_leaf=20, random_state=42)
    boost.fit(x_train[finished_train], y_train[finished_train])
    boost_pred = pd.Series(boost.predict(x_test), index=x_test.index)
    evaluate("gradient boosting", y_test, boost_pred, finished_test)

    # for each compound
    compound_cols = [c for c in x.columns if c.startswith("compound_")]
    groups = {"pos": ["pos"], "GridPosition": ["GridPosition"],
              "life": ["life"],
              "PitStopsSoFar": ["PitStopsSoFar"], "SnapshotPct": ["SnapshotPct"],
              "compound": compound_cols}

    test_finishers = x_test[finished_test]
    y_finishers = y_test[finished_test]

    # permutation feature importance
    print()
    print("### FEATURE IMPORTANCE ###")
    linear_imp = feature_importance(model.predict, test_finishers, y_finishers, groups)
    ann_imp = feature_importance(lambda d: ann.predict(scaler.transform(d)),
                                 test_finishers, y_finishers, groups)
    forest_imp = feature_importance(forest.predict, test_finishers, y_finishers, groups)
    boost_imp = feature_importance(boost.predict, test_finishers, y_finishers, groups)

    print("Linear:\n", linear_imp)
    print()
    print("ANN:\n", ann_imp)
    print()
    print("Forest:\n", forest_imp)
    print()
    print("Gradient boosting:\n", boost_imp)

    print("### SHAP IMPORTANCE ###")
    sample = test_finishers.sample(500, random_state=42)

    linear_shap = shap.LinearExplainer(model, x_train[finished_train]).shap_values(sample)
    print("Linear SHAP:\n", shap_importance(linear_shap, sample.columns, compound_cols))
    print()

    # only explain the first 200 rows
    background = shap.sample(x_train[finished_train], 100, random_state=42)
    ann_explainer = shap.KernelExplainer(lambda d: ann.predict(scaler.transform(pd.DataFrame(d, columns=x_train.columns))), background)
    ann_shap = ann_explainer.shap_values(sample.iloc[:300], nsamples=300)

    print("ANN SHAP:\n", shap_importance(ann_shap, sample.columns, compound_cols))
    print()
    forest_shap = shap.TreeExplainer(forest).shap_values(sample)
    print("Forest SHAP:\n", shap_importance(forest_shap, sample.columns, compound_cols))

    boost_shap = shap.TreeExplainer(boost).shap_values(sample)
    print("Gradient boosting SHAP:\n", shap_importance(boost_shap, sample.columns, compound_cols))