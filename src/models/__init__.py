try:  # optional dependency
    from .lstm_model import train_lstm
except ImportError:  # pragma: no cover - optional tensorflow/keras
    train_lstm = None
from .rf_model import train_rf
from .xgb_model import train_xgb
from .linear_model import train_linear
from .lightgbm_model import train_lgbm
from .arima_model import train_arima

__all__ = [
    "train_rf",
    "train_xgb",
    "train_linear",
    "train_lgbm",
    "train_arima",
]
if train_lstm is not None:
    __all__.insert(0, "train_lstm")
