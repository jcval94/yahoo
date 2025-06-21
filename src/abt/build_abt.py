"""Build daily and weekly ABTs enriched with technical indicators"""
import pandas as pd, yfinance as yf, ta, yaml, os
from pathlib import Path

# Load configuration from the project root
with open(os.path.join(os.path.dirname(__file__), '../../config.yaml')) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), '../../data'))
DATA_DIR.mkdir(exist_ok=True, parents=True)

# ... (insert full code from abt_builder_plus2.py here) ...
