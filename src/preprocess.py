# preprocess.py
import os
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_CONFIG = {
    "data": {
        "raw_path": "data/raw.csv",
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
        "target_col": "diagnosis",
        "id_col": "id",
        "csv_sep": ",",  # change to "\t" if your file is TSV
    },
    "split": {
        "test_size": 0.2,
        "random_state": 42
    }
}

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print(f"config.yaml not found. Using defaults: {DEFAULT_CONFIG}")
        return DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # merge shallow defaults
    out = DEFAULT_CONFIG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v
    return out

def main():
    cfg = load_config()
    dcfg = cfg["data"]
    scfg = cfg["split"]

    raw_path = Path(dcfg["raw_path"])
    train_path = Path(dcfg["train_path"])
    test_path = Path(dcfg["test_path"])
    target_col = dcfg["target_col"]
    id_col = dcfg.get("id_col")
    sep = dcfg.get("csv_sep", ",")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path.resolve()}")

    # Ensure folders exist
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {raw_path} with sep='{sep}'...")
    df = pd.read_csv(raw_path, sep=sep)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    # Drop ID column if present
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])

    # Map diagnosis: M->1, B->0 if it looks categorical
    if df[target_col].dtype == "object":
        mapping = {"M": 1, "B": 0}
        if set(df[target_col].unique()) <= set(mapping.keys()):
            df[target_col] = df[target_col].map(mapping)
        else:
            # If different labels, try factorize as fallback
            df[target_col] = pd.factorize(df[target_col])[0]

    # Basic sanity check: target must be binary 0/1
    unique_targets = sorted(df[target_col].dropna().unique().tolist())
    print(f"Target unique values after mapping: {unique_targets}")
    if not set(unique_targets).issubset({0, 1}):
        print("Warning: Target is not strictly binary 0/1 after mapping.")

    # Split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=scfg["test_size"],
        random_state=scfg["random_state"],
        stratify=y
    )

    # Save to CSV
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train set to {train_path} ({train_df.shape[0]} rows, {train_df.shape[1]} cols)")
    print(f"Saved test set to {test_path} ({test_df.shape[0]} rows, {test_df.shape[1]} cols)")

if __name__ == "__main__":
    main()