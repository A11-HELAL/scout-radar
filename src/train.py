"""Preprocessing, the models, the leaderboard, the ablation, the importances."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.evaluate import evaluate


def build_preprocessor(features=None):
    features = list(features or config.FEATURES)
    numeric = [f for f in features if f in config.NUMERIC_FEATURES]
    categorical = [f for f in features if f in config.CATEGORICAL_FEATURES]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=10, sparse_output=False
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def build_models():
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=config.RANDOM_STATE,
        ),
    }


def train_all(train, test, features=None, models=None):
    features = list(features or config.FEATURES)
    models = models or build_models()

    X_train, y_train = train[features], train[config.TARGET]
    X_test, y_test = test[features], test[config.TARGET]

    rows, fitted, preds = [], {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", build_preprocessor(features)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)

        rows.append(evaluate(y_test, y_hat, label=name))
        fitted[name] = pipe
        preds[name] = y_hat

    leaderboard = pd.DataFrame(rows).sort_values("MAE_log").reset_index(drop=True)
    return leaderboard, fitted, preds


def ablation(train, test, model_name="gradient_boosting", groups=None):
    groups = groups if groups is not None else config.ABLATION_GROUPS

    rows = []
    board, _, _ = train_all(
        train, test, config.FEATURES, {model_name: build_models()[model_name]}
    )
    rows.append(board.assign(dropped="nothing (all features)"))

    for label, drop in groups.items():
        kept = [f for f in config.FEATURES if f not in drop]
        board, _, _ = train_all(
            train, test, kept, {model_name: build_models()[model_name]}
        )
        rows.append(board.assign(dropped=f"without {label}"))

    out = pd.concat(rows, ignore_index=True)
    return out[["dropped", "MAE_log", "MedAPE_pct", "Spearman", "n"]]


def feature_importance(fitted_pipeline, test, features=None, n_repeats=5):
    features = list(features or config.FEATURES)
    result = permutation_importance(
        fitted_pipeline,
        test[features],
        test[config.TARGET],
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": features,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
