"""End-to-end pipeline runner"""
import logging
import yaml
from pathlib import Path
from abt.build_abt import build_abt
from models.lstm_model import train_lstm_model
from portfolio.optimize import optimize_portfolio
from notify.notifier import send_notification

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    abt_path = build_abt(config)
    model_path = train_lstm_model(abt_path, config)
    weights_path = optimize_portfolio(abt_path, config)
    send_notification(
        f"Pipeline completed. Model: {model_path}, Weights: {weights_path}")


if __name__ == "__main__":
    main()
