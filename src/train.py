"""Preprocessing, the models, the leaderboard, the ablation, the importances.

Everything the model touches is fitted INSIDE a Pipeline, so the imputer and
the scaler learn from the training snapshot only. Scaling before the split is
the quietest form of leakage there is.
"""

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
    """Numbers: median-impute then scale. Categories: fill then one-hot."""
    features = list(features or config.FEATURES)
    numeric = [f for f in features if f in config.NUMERIC_FEATURES]
    categorical = [f for f in features if f in config.CATEGORICAL_FEATURES]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a club or nationality that only exists in the
        # 2024 snapshot must not crash prediction.
        # min_frequency=10: one-hotting a category with 3 players just gives the
        # model 3 rows to memorise.
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=10, sparse_output=False
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def build_models():
    """Four models, from stupid to strong.

    baseline_mean is not filler: it predicts the average value for everybody, so
    it tells us how much the real models actually add. If a RandomForest cannot
    beat it, we have no result to present. ridge is the linear reference - if a
    straight line is nearly as good, the trees are not earning their keep.
    """
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
    """Fit every model on the train snapshot, score it on the test snapshot.

    Returns (leaderboard, fitted_pipelines, predictions). The leaderboard is
    sorted by MAE in log space, so row 0 is the winner.
    """
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
    """Retrain the winner with one group of features removed at a time.

    Three groups only (config.ABLATION_GROUPS) because each one costs a full
    training run. The row that matters in the defence is "without present-day
    club info": it says out loud how much of the result leans on columns that
    know a little about the future.
    """
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
    """Permutation importance: shuffle one column and see how much worse it gets.

    Measured on the TEST snapshot on purpose. The question is "does this column
    help price players the model has never seen", not "how often did the trees
    split on it".
    """
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
