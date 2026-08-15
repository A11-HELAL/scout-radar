"""Five models - from dumbest to smartest."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.evaluate import evaluate
from src.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor():
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def get_models():
    return {
        "0_baseline": DummyRegressor(strategy="median"),
        "1_linear": LinearRegression(),
        "2_ridge": Ridge(alpha=1.0),
        "3_random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "4_gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def run_all(train, test):
    """Train all five and return the metrics table plus fitted pipelines."""
    X_train, y_train = train[ALL_FEATURES], train["y_log"]
    X_test, y_test = test[ALL_FEATURES], test["y_log"]

    rows, fitted = [], {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        scores = evaluate(y_test, pipe.predict(X_test))
        scores["model"] = name
        rows.append(scores)
        fitted[name] = pipe
        print(name, scores)

    results = pd.DataFrame(rows).set_index("model").sort_values("MAE")
    return results, fitted
