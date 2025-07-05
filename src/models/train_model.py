"""Train a small model reusing the main pipeline utilities."""
from pathlib import Path
import argparse

import pandas as pd

from ..abt.build_abt import build_abt
from ..models.linear_model import train_linear
from ..evaluation import evaluate_predictions
from ..utils.schema_guard import hash_schema, save_with_schema


MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a smoke-test model")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    abt_paths = build_abt("daily")
    df = pd.read_csv(abt_paths[args.ticker], index_col=0, parse_dates=True)
    df = df.head(args.rows)
    df["target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    X = df.drop(columns=["Close", "target"], errors="ignore")
    y = df["target"]

    model = train_linear(X, y)

    schema_hash = hash_schema(X)
    model_path = MODEL_DIR / f"{args.ticker}_linreg_{schema_hash}.joblib"
    save_with_schema(model, model_path, list(X.columns), schema_hash)

    preds = model.predict(X)
    metrics = evaluate_predictions(y, preds)
    print(model_path)
    print(metrics)


if __name__ == "__main__":
    main()
