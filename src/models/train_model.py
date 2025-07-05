"""Minimal training script storing schema metadata."""
from pathlib import Path
import argparse

import pandas as pd
from sklearn.linear_model import LinearRegression

from ..utils.schema_guard import hash_schema, save_with_schema


MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dummy model")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = args.rows
    X = pd.DataFrame({f"feat_{i}": range(rows) for i in range(3)})
    y = pd.Series(range(rows))

    model = LinearRegression()
    model.fit(X, y)

    schema_hash = hash_schema(X)
    model_path = MODEL_DIR / f"{args.ticker}_linreg_{schema_hash}.joblib"
    save_with_schema(model, model_path, list(X.columns), schema_hash)
    print(model_path)


if __name__ == "__main__":
    main()
