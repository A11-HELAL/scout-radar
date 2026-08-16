"""Explain one player's prediction with a SHAP waterfall. Optional extra.

Why this is a separate script and not part of run_all.py: `shap` is a heavy
dependency that breaks on version bumps, and the pipeline must never depend on
it. Nothing in src/ or tests/ imports this file.

    pip install shap
    python notebooks/shap_demo.py

What you get: one PNG showing, for a single player, which features pushed the
model's predicted value up and which pushed it down. That is the picture that
answers "why is he on the list?" in the discussion.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_dataset
from src.train import build_models, build_preprocessor

# Mohamed Salah, our sanity-check player. If he is not in the snapshot (he may
# fail the age filter), we fall back to whoever played the most minutes.
PLAYER_ID = 28003
MODEL_NAME = "gradient_boosting"


def main():
    try:
        import shap
    except ImportError:
        raise SystemExit(
            "This script needs shap, which the pipeline deliberately does not "
            "require:\n    pip install shap"
        )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"building the {config.SNAPSHOT_TRAIN} snapshot ...")
    train = build_dataset(config.SNAPSHOT_TRAIN)
    features = train[config.FEATURES]

    print(f"fitting {MODEL_NAME} ...")
    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", build_models()[MODEL_NAME]),
    ])
    pipeline.fit(features, train[config.TARGET])

    # Slice the pipeline instead of naming steps: everything except the last
    # step is the preprocessing, the last step is the model itself.
    matrix = pipeline[:-1].transform(features)
    if hasattr(matrix, "toarray"):        # one-hot encoding returns sparse
        matrix = matrix.toarray()
    try:
        names = list(pipeline[:-1].get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(matrix.shape[1])]
    matrix = pd.DataFrame(matrix, columns=names, index=train.index)

    if (train["player_id"] == PLAYER_ID).any():
        position = int(np.flatnonzero((train["player_id"] == PLAYER_ID).to_numpy())[0])
    else:
        position = int(np.argmax(train["minutes"].to_numpy()))
        print(f"player {PLAYER_ID} is not in this snapshot, using a stand-in")

    row = train.iloc[position]
    print(
        f"explaining {row['name']} ({row['club_name']}), "
        f"market EUR {row['market_value_in_eur']:,.0f}"
    )

    explanation = shap.TreeExplainer(pipeline[-1])(matrix)

    shap.plots.waterfall(explanation[position], max_display=14, show=False)
    plt.title(f"Why the model priced {row['name']} where it did")
    plt.tight_layout()

    out_path = Path(config.FIGURES) / "shap_player.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"saved -> {out_path}")
    print(
        "read it as log-value: a bar of +0.30 means that feature multiplied the "
        "predicted value by about exp(0.30) = 1.35"
    )


if __name__ == "__main__":
    main()
