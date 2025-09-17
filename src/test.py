import os
import json
from pathlib import Path
import yaml
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

DEFAULT_CONFIG = {
    "data": {
        "test_path": "data/test.csv",
        "target_col": "diagnosis"
    },
    "artifacts": {
        "model_path": "models/model.joblib",
        "metadata_path": "models/metadata.json",
        "training_history_path": "training_history.json"  # Added path for training history
    },
    "experiments": {
        "results_csv": "experiments/results.csv"
    }
}

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print(f"config.yaml not found. Using defaults: {DEFAULT_CONFIG}")
        return DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    out = DEFAULT_CONFIG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v
    return out

def safe_predict_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # Scale to [0,1] with a simple min-max on decision scores if needed
        import numpy as np
        scores = model.decision_function(X)
        # Fallback normalization
        min_s, max_s = scores.min(), scores.max()
        proba = (scores - min_s) / (max_s - min_s + 1e-12)
    else:
        # No probability—fall back to predictions (ROC AUC may be less informative)
        proba = None
    preds = model.predict(X)
    return preds, proba

def main():
    cfg = load_config()
    dcfg = cfg["data"]
    apaths = cfg["artifacts"]
    expcfg = cfg["experiments"]

    test_path = Path(dcfg["test_path"])
    target_col = dcfg["target_col"]
    model_path = Path(apaths["model_path"])
    metadata_path = Path(apaths["metadata_path"])
    training_history_path = Path(apaths["training_history_path"])  # Get training history path
    results_csv = Path(expcfg["results_csv"])

    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found at {test_path.resolve()}. Run preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path.resolve()}. Run train.py first.")

    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in test set.")

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col].astype(int)

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    preds, proba = safe_predict_proba_or_score(model, X_test)

    # Calculate final test metrics
    test_accuracy = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds, average="macro")
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")

    # Load training history
    training_history = []
    if training_history_path.exists():
        try:
            with open(training_history_path, "r") as f:
                training_history = json.load(f)
            print(f"Loaded training history from {training_history_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {training_history_path}. Skipping training history.")
    else:
        print(f"Training history file not found at {training_history_path}. DVC plots for training history will be empty.")

    # Prepare metrics for metrics.json
    metrics = {
        "accuracy": test_accuracy,
        "f1": test_f1,
        "training_history": training_history  # Include the full history for plotting
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)  # Use indent for readability
    print("Metrics and training history saved to metrics.json")

    # ✅ Save training history separately for DVC plots
    Path("plots").mkdir(exist_ok=True)
    with open("plots/training.json", "w") as f:
        json.dump(training_history, f, indent=2)
    print("Training history saved to plots/training.json")

    if proba is not None:
        roc = roc_auc_score(y_test, proba)
    else:
        roc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # Load training metadata if available
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Prepare results row
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_type": metadata.get("model_type", ""),
        "best_params": json.dumps(metadata.get("best_params", {})),
        "cv_roc_auc_mean": metadata.get("cv_scores", {}).get("roc_auc", ""),
        "cv_accuracy_mean": metadata.get("cv_scores", {}).get("accuracy", ""),
        "test_accuracy": test_accuracy,  # Use the calculated test_accuracy
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": test_f1,  # Use the calculated test_f1
        "test_roc_auc": roc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": int(len(y_test))
    }

    # Append (create file with header if not exists)
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(results_csv, index=False)  # Changed to to_csv to match initial load
    else:
        pd.DataFrame([row]).to_csv(results_csv, index=False)

    print(f"Appended results to {results_csv}")
    print("Test metrics:")
    print({k: v for k, v in row.items() if k.startswith("test_")})

if __name__ == "__main__":
    main()
