"""Minimal prediction script with schema validation."""
from pathlib import Path
import argparse
import sys
import subprocess
import pandas as pd

from ..utils.schema_guard import load_with_schema, validate_schema

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy prediction")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--auto-retrain", action="store_true")
    args = parser.parse_args()

    matches = sorted(MODEL_DIR.glob(f"{args.ticker}_linreg_*.joblib"))
    if not matches:
        print(f"No model found for {args.ticker}", file=sys.stderr)
        sys.exit(1)
    model_path = matches[-1]
    model, feature_list, schema_hash = load_with_schema(model_path)

    X_live = pd.DataFrame({name: range(args.rows) for name in feature_list})
    try:
        validate_schema(feature_list, X_live, schema_hash)
    except SystemExit:
        if args.auto_retrain:
            ret = subprocess.run([
                sys.executable,
                "-m",
                "src.models.train_model",
                "--ticker",
                args.ticker,
                "--rows",
                str(args.rows),
                "--smoke",
            ])
            if ret.returncode != 0:
                sys.exit(98)
            matches = sorted(MODEL_DIR.glob(f"{args.ticker}_linreg_*.joblib"))
            model_path = matches[-1]
            model, feature_list, schema_hash = load_with_schema(model_path)
            validate_schema(feature_list, X_live, schema_hash)
        else:
            raise

    X_live = X_live[feature_list]
    preds = model.predict(X_live)
    print(",".join(str(p) for p in preds))


if __name__ == "__main__":
    main()
