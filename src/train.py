# train.py
import os
import json
from pathlib import Path
import yaml
import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV

import pickle

DEFAULT_CONFIG = {
    "data": {
        "train_path": "data/train.csv",
        "target_col": "diagnosis"
    },
    "model": {
        "type": "logreg",  # "logreg" or "random_forest"
        "logreg": {
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "C_grid": [0.1, 1.0, 10.0]
        },
        "random_forest": {
            "class_weight": "balanced",
            "n_estimators_grid": [200, 400],
            "max_depth_grid": [None, 6, 10],
            "min_samples_leaf_grid": [1, 2, 4]
        }
    },
    "split": {
        "random_state": 42
    },
    "cv": {
        "n_splits": 5,
        "scoring": ["roc_auc", "accuracy"],
        "refit": "roc_auc",
        "n_jobs": -1,
        "verbose": 1
    },
    "artifacts": {
        "model_path": "models/model.pkl",
        "metadata_path": "models/metadata.json"
    }
}

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print(f"config.yaml not found. Using defaults: {DEFAULT_CONFIG}")
        return DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # shallow merge
    out = DEFAULT_CONFIG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v
    return out

def build_estimator(cfg, numeric_features):
    mc = cfg["model"]
    model_type = mc["type"]

    # Preprocessing for numeric columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )

    if model_type == "logreg":
        base_clf = LogisticRegression(
            penalty=mc["logreg"]["penalty"],
            solver=mc["logreg"]["solver"],
            class_weight=mc["logreg"]["class_weight"],
            max_iter=2000
        )
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clf", base_clf)
        ])
        param_grid = {
            "clf__C": mc["logreg"].get("C_grid", [0.1, 1.0, 10.0])
        }

    elif model_type == "random_forest":
        base_clf = RandomForestClassifier(
            class_weight=mc["random_forest"]["class_weight"],
            random_state=cfg["split"]["random_state"],
            n_jobs=-1
        )
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clf", base_clf)
        ])
        param_grid = {
            "clf__n_estimators": mc["random_forest"].get("n_estimators_grid", [200, 400]),
            "clf__max_depth": mc["random_forest"].get("max_depth_grid", [None, 6, 10]),
            "clf__min_samples_leaf": mc["random_forest"].get("min_samples_leaf_grid", [1, 2, 4])
        }
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")

    return pipe, param_grid, model_type

def main():
    cfg = load_config()
    dcfg = cfg["data"]
    tpath = Path(dcfg["train_path"])
    target_col = dcfg["target_col"]

    if not tpath.exists():
        raise FileNotFoundError(f"Train set not found at {tpath.resolve()}. Run preprocess.py first.")

    print(f"Loading train data from {tpath}...")
    df = pd.read_csv(tpath)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in train set.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    print(f"Using {len(numeric_features)} numeric features.")

    pipe, param_grid, model_type = build_estimator(cfg, numeric_features)

    cvcfg = cfg["cv"]
    vcfg = cvcfg["n_splits"]
    cv = StratifiedKFold(
        n_splits=vcfg,
        shuffle=True,
        random_state=cfg["split"]["random_state"]
    )
    scoring = cvcfg.get("scoring", ["roc_auc", "accuracy"])
    if isinstance(scoring, list):
        scoring_dict = {s: s for s in scoring}
    else:
        scoring_dict = {scoring: scoring}
    refit_metric = cvcfg.get("refit", "roc_auc")

    print(f"Starting GridSearchCV over {len(list(param_grid.values())[0])}x... combos (depends on model).")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring_dict,
        refit=refit_metric,
        n_jobs=cvcfg.get("n_jobs", -1),
        verbose=cvcfg.get("verbose", 1),
        return_train_score=False
    )

    grid.fit(X, y)
    best_estimator = grid.best_estimator_
    best_params = grid.best_params_

    # Extract CV scores for the refit metric and accuracy if present
    best_idx = grid.best_index_
    mean_scores = {}
    for metric in scoring_dict.keys():
        key = f"mean_test_{metric}"
        if key in grid.cv_results_:
            mean_scores[metric] = float(grid.cv_results_[key][best_idx])

    print(f"Best params: {best_params}")
    print(f"Best CV scores: {mean_scores}")

    # Save artifacts
    apaths = cfg["artifacts"]
    model_path = Path(apaths["model_path"])
    metadata_path = Path(apaths["metadata_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_estimator, model_path)
    print(f"Saved best model to {model_path}")
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_estimator, f)

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_type": model_type,
        "best_params": best_params,
        "cv_scores": mean_scores,
        "n_features": len(numeric_features),
        "random_state": cfg["split"]["random_state"],
        "cv_n_splits": cvcfg["n_splits"]
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to {metadata_path}")



if __name__ == "__main__":
    main()