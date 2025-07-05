import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple, Any

import joblib
import pandas as pd


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
        print("\u2716 SCHEMA MISMATCH \u2716", file=sys.stderr)
        print(f"- Esperado: {exp_cols} columnas (hash {schema_hash})", file=sys.stderr)
        print(f"- Recibido: {recv_cols} columnas (hash {live_hash})", file=sys.stderr)
        print("Sugerencia: re-entrena con build_abt actual.", file=sys.stderr)
        sys.exit(99)
