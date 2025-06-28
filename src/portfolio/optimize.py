"""Portfolio optimization utilities and weekly recommendation logic."""
import logging
import time
from typing import Dict, Sequence

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    yaml = None
from pathlib import Path

logger = logging.getLogger(__name__)


def _mean(values: Sequence[float]) -> float:
    """Return the average of a sequence of numbers."""
    return sum(values) / len(values)


def _cov_matrix(returns: Dict[str, Sequence[float]]):
    """Return covariance matrix and mean returns for a returns dictionary."""
    tickers = list(returns)
    n_obs = len(next(iter(returns.values())))
    means = [_mean(returns[t]) for t in tickers]
    cov = [[0.0 for _ in tickers] for _ in tickers]

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            s = 0.0
            for k in range(n_obs):
                s += (returns[ti][k] - means[i]) * (returns[tj][k] - means[j])
            cov[i][j] = s / n_obs
    return cov, means


def _invert(matrix):
    """Return inverse of a square matrix using Gauss-Jordan elimination."""
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = aug[i][i]
        if pivot == 0:
            for r in range(i + 1, n):
                if aug[r][i] != 0:
                    aug[i], aug[r] = aug[r], aug[i]
                    pivot = aug[i][i]
                    break
        if pivot == 0:
            raise ValueError("Matrix is singular")
        factor = aug[i][i]
        for c in range(2 * n):
            aug[i][c] /= factor
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for c in range(2 * n):
                    aug[r][c] -= factor * aug[i][c]
    inv = [[aug[i][j + n] for j in range(n)] for i in range(n)]
    return inv


def optimize_portfolio(data: Dict[str, Sequence[float]], **kwargs) -> Dict[str, float]:
    """Compute Markowitz mean-variance optimal weights.

    Parameters
    ----------
    data:
        Dictionary mapping tickers to lists of historical returns.

    Returns
    -------
    Dict[str, float]
        Dictionary of optimal weights that sum to 1.
    """

    start = time.perf_counter()
    logger.info("Starting portfolio optimization")
    try:
        if not data:
            logger.warning("No data supplied for optimization")
            return {}

        config_path = kwargs.get(
            "config_path", Path(__file__).resolve().parents[1] / "config.yaml"
        )
        rf_rate = 0.0
        try:
            with open(config_path) as cfg_file:
                if yaml is not None:
                    config = yaml.safe_load(cfg_file)
                    rf_rate = config.get("risk_free_rate", 0.0)
                else:
                    for line in cfg_file:
                        if line.strip().startswith("risk_free_rate"):
                            rf_rate = float(line.split(":", 1)[1].strip())
                            break
        except FileNotFoundError:
            logger.warning("Config file %s not found", config_path)

        cov, means = _cov_matrix(data)
        inv_cov = _invert(cov)
        excess = [m - rf_rate for m in means]
        raw = [sum(inv_cov[i][j] * excess[j] for j in range(len(excess))) for i in range(len(excess))]
        total = sum(raw)
        if total == 0:
            w = [1.0 / len(raw) for _ in raw]
        else:
            w = [r / total for r in raw]
        weights = {t: round(wi, 3) for t, wi in zip(data.keys(), w)}
        logger.info("Optimal weights: %s", weights)
    except Exception:
        logger.exception("Error during portfolio optimization")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Optimization finished in %.2f seconds", duration)
    return weights
