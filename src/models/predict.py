"""Run a quick prediction using the main pipeline helpers."""
import argparse
import sys
import subprocess
import pandas as pd

from ..abt.build_abt import build_abt
from ..predict import load_models, run_predictions, MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy prediction")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--auto-retrain", action="store_true")
    args = parser.parse_args()

    abt_paths = build_abt("daily")
    df = pd.read_csv(abt_paths[args.ticker], index_col=0, parse_dates=True)
    df = df.head(args.rows)

    models = load_models(MODEL_DIR)
    try:
        run_predictions(models, {args.ticker: df})
    except SystemExit:
        if args.auto_retrain:
            subprocess.run(
                [sys.executable, "-m", "src.models.train_model", "--ticker", args.ticker],
                check=True,
            )
            models = load_models(MODEL_DIR)
            run_predictions(models, {args.ticker: df})
        else:
            raise


if __name__ == "__main__":
    main()
