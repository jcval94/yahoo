import hashlib
import json
import sys
import logging
from pathlib import Path
from typing import Iterable, Tuple, Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

def hash_schema(df: pd.DataFrame) -> str:
    """Return short SHA-1 hash of DataFrame column names."""
    joined = tuple(df.columns)
    digest = hashlib.sha1(str(joined).encode()).hexdigest()
    return digest[:10]


def save_with_schema(model: Any, path: str | Path, feature_list: Iterable[str], schema_hash: str) -> None:
    """Save a model with schema metadata."""
    p = Path(path)
    joblib.dump(model, p)
    meta = {"features": list(feature_list), "schema_hash": schema_hash}
    p.with_suffix(".json").write_text(json.dumps(meta))


def load_with_schema(path: str | Path) -> Tuple[Any, list[str], str]:
    """Load a model and its schema metadata."""
    p = Path(path)
    model = joblib.load(p)
    meta = json.loads(p.with_suffix(".json").read_text())
    return model, meta.get("features", []), meta.get("schema_hash", "")


def validate_schema(feature_list: Iterable[str], df: pd.DataFrame, schema_hash: str) -> None:
    """Validate columns against stored schema, exit 99 on mismatch."""
    subset = df[list(feature_list)]
    live_hash = hash_schema(subset)
    if live_hash != schema_hash:
        exp_cols = len(feature_list)
        recv_cols = df.shape[1]
        logger.error("\u2716 SCHEMA MISMATCH \u2716")
        logger.error("- Esperado: %s columnas (hash %s)", exp_cols, schema_hash)
        logger.error("- Recibido: %s columnas (hash %s)", recv_cols, live_hash)
        logger.error("Sugerencia: re-entrena con build_abt actual.")
        raise SystemExit(99)

