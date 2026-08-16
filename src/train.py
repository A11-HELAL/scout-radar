"""Preprocessing + the model zoo.

Everything is wrapped in a scikit-learn Pipeline on purpose: the imputer and
the scaler are fitted on the TRAIN snapshot only, so no information from 2024
can leak backwards into the 2023 model through a median or a mean.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.evaluate import evaluate


def build_preprocessor(features=None):
    """Impute + scale the numbers, impute + one-hot the categories."""
    features = list(features or config.FEATURES)
    numeric = [f for f in config.NUMERIC_FEATURES if f in features]
    categorical = [f for f in config.CATEGORICAL_FEATURES if f in features]

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" so a sub_position that only shows up in 2024
        # cannot crash the model; min_frequency folds rare labels together.
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])


def build_models():
    """Four models, deliberately in increasing order of cleverness.

    baseline_mean exists so we can prove the clever models beat "predict the
    average". A project without a baseline has no story.
    """
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3,
            n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=config.RANDOM_STATE,
        ),
    }


def train_all(train, test, features=None, models=None):
    """Fit every model on the train snapshot and score it on the test snapshot.

    Returns (leaderboard, fitted_pipelines, best_model_name).
    """
    features = list(features or config.FEATURES)
    models = models or build_models()

    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", build_preprocessor(features)),
            # clone() so re-running the cell never reuses an already-fitted model
            ("model", clone(estimator)),
        ])
        pipe.fit(train[features], train[config.TARGET])
        predictions = pipe.predict(test[features])
        rows.append(evaluate(test[config.TARGET], predictions, label=name))
        fitted[name] = pipe

    leaderboard = pd.DataFrame(rows).set_index("model").sort_values("MAE_log")
    return leaderboard, fitted, leaderboard.index[0]


def ablation(train, test, drop=None, model_name="gradient_boosting"):
    """Retrain the winner without the features we are not fully sure about.

    If the score barely moves, those features were not doing the work and the
    result does not depend on them - which is exactly what we want to be able
    to say out loud in the defence.
    """
    drop = list(drop or config.ANACHRONISTIC_FEATURES)
    kept = [f for f in config.FEATURES if f not in drop]
    only_winner = {model_name: build_models()[model_name]}

    with_all, _, _ = train_all(train, test, config.FEATURES, only_winner)
    without, _, _ = train_all(train, test, kept, only_winner)

    out = pd.concat([
        with_all.assign(features="all features"),
        without.assign(features=f"without {len(drop)} suspects"),
    ])
    return out[["features", "MAE_log", "MedAPE_eur", "Spearman", "n"]]


def feature_importance(pipe, top=15):
    """Importances straight out of the fitted tree model, with readable names."""
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.Series(dtype=float)
    names = pipe.named_steps["prep"].get_feature_names_out()
    return (
        pd.Series(model.feature_importances_, index=names)
        .sort_values(ascending=False)
        .head(top)
    )
