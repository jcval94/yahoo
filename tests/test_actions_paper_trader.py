import pandas as pd

from src.actions.paper_trader import (
    _evaluate_strategies,
    _commission,
    _execution_price,
    _max_drawdown,
    TradingConfig,
)


def test_evaluate_strategies_returns_five_signals():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "model": ["linreg", "rf", "xgb", "lgbm"],
            "actual": [100.0, 100.0, 100.0, 100.0],
            "pred": [102.0, 101.0, 99.0, 103.0],
        }
    )
    model_scores = {
        ("AAA", "linreg"): 1.1,
        ("AAA", "rf"): 1.4,
        ("AAA", "xgb"): 1.7,
        ("AAA", "lgbm"): 1.2,
    }
    ranked = {"AAA": ["linreg", "lgbm", "rf", "xgb"]}
    stability = {("AAA", "lgbm"): 0.9}

    out = _evaluate_strategies(df, model_scores, ranked, stability)

    assert out["best_model"] == "linreg"
    assert set(out["signals"].keys()) == {
        "winner_take_all",
        "top3_ensemble",
        "consensus_vote",
        "risk_adjusted_edge",
        "stability_filter",
    }
    assert isinstance(out["score"], float)


def test_cost_model_buy_sell_are_directional():
    cfg = TradingConfig()
    buy_px = _execution_price(100.0, cfg, "BUY")
    sell_px = _execution_price(100.0, cfg, "SELL")

    assert buy_px > 100.0
    assert sell_px < 100.0

    buy_fee = _commission(10_000.0, cfg, "BUY")
    sell_fee = _commission(10_000.0, cfg, "SELL")
    assert sell_fee > buy_fee


def test_max_drawdown_works_for_simple_curve():
    curve = [100.0, 110.0, 90.0, 95.0]
    dd = _max_drawdown(curve)

    assert round(dd, 4) == 0.1818
