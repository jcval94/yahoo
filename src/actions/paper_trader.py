"""Paper trading engine driven by model predictions.

This module simulates realistic daily buy/sell actions with fees and slippage,
keeps five strategy signals in competition, and persists a detailed trade ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..utils import load_config

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"
CONFIG = load_config(CONFIG_PATH)
RESULTS_DIR = ROOT_DIR / "results"
ACTIONS_DIR = RESULTS_DIR / "actions"
ACTIONS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_NAMES = [
    "winner_take_all",
    "top3_ensemble",
    "consensus_vote",
    "risk_adjusted_edge",
    "stability_filter",
]


@dataclass
class TradingConfig:
    initial_budget: float = 10_000.0
    max_position_pct: float = 0.12
    min_trade_usd: float = 300.0
    buy_threshold: float = 2.8
    sell_threshold: float = -1.8
    holding_days: int = 2
    commission_fixed: float = 0.35
    commission_bps: float = 4.0
    sec_fee_bps: float = 0.25
    spread_bps: float = 8.0
    slippage_bps: float = 10.0
    max_tickers_per_day: int = 3


def _business_day(ts: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    return ts + pd.offsets.BDay(n)


def _to_date(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).normalize()


def load_trading_config() -> TradingConfig:
    raw = CONFIG.get("actions", {})
    return TradingConfig(
        initial_budget=float(raw.get("initial_budget", 10_000.0)),
        max_position_pct=float(raw.get("max_position_pct", 0.12)),
        min_trade_usd=float(raw.get("min_trade_usd", 300.0)),
        buy_threshold=float(raw.get("buy_threshold", 2.8)),
        sell_threshold=float(raw.get("sell_threshold", -1.8)),
        holding_days=int(raw.get("holding_days", 2)),
        commission_fixed=float(raw.get("commission_fixed", 0.35)),
        commission_bps=float(raw.get("commission_bps", 4.0)),
        sec_fee_bps=float(raw.get("sec_fee_bps", 0.25)),
        spread_bps=float(raw.get("spread_bps", 8.0)),
        slippage_bps=float(raw.get("slippage_bps", 10.0)),
        max_tickers_per_day=int(raw.get("max_tickers_per_day", 3)),
    )


def _load_prediction_files() -> pd.DataFrame:
    pred_dir = RESULTS_DIR / "predicts"
    files = sorted(pred_dir.glob("*_daily_predictions.csv"))
    if not files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            logger.exception("Cannot read prediction file %s", path)
            continue
        if df.empty or "ticker" not in df.columns:
            continue
        run_date = _to_date(path.stem.split("_")[0])
        work = df.copy()
        work["run_date"] = run_date
        if "Predicted" in work.columns:
            work["predicted_date"] = pd.to_datetime(work["Predicted"], errors="coerce").dt.normalize()
        else:
            work["predicted_date"] = run_date
        work["actual"] = pd.to_numeric(work.get("actual"), errors="coerce")
        work["pred"] = pd.to_numeric(work.get("pred"), errors="coerce")
        work["delta"] = work["pred"] - work["actual"]
        work["delta_pct"] = np.where(work["actual"] != 0, work["delta"] / work["actual"], 0.0)
        frames.append(work[["ticker", "model", "actual", "pred", "delta", "delta_pct", "run_date", "predicted_date"]])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["ticker", "model", "actual", "pred", "predicted_date"])


def _load_latest_model_scores() -> tuple[dict[tuple[str, str], float], dict[str, list[str]]]:
    metrics_files = sorted((RESULTS_DIR / "metrics").glob("metrics_daily_*.csv"))
    if not metrics_files:
        return {}, {}
    df = pd.read_csv(metrics_files[-1])
    if df.empty:
        return {}, {}
    work = df.copy()
    work = work[work.get("dataset", "") == "test"] if "dataset" in work.columns else work
    work["MAE"] = pd.to_numeric(work.get("MAE"), errors="coerce")
    work = work.dropna(subset=["model", "MAE"])

    score_map: dict[tuple[str, str], float] = {}
    ranked: dict[str, list[str]] = {}

    for _, row in work.iterrows():
        model_key = str(row["model"])
        if "_" not in model_key:
            continue
        ticker, algo = model_key.split("_", 1)
        score_map[(ticker, algo)] = float(row["MAE"])

    by_ticker: dict[str, list[tuple[str, float]]] = {}
    for (ticker, algo), mae in score_map.items():
        by_ticker.setdefault(ticker, []).append((algo, mae))
    for ticker, vals in by_ticker.items():
        vals.sort(key=lambda x: x[1])
        ranked[ticker] = [algo for algo, _ in vals]

    return score_map, ranked


def _load_stability_scores() -> dict[tuple[str, str], float]:
    files = sorted((RESULTS_DIR / "edge_metrics").glob("edge_metrics_*.csv"))[-20:]
    if not files:
        return {}
    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            logger.exception("Cannot parse edge metrics %s", p)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    if df.empty or not {"ticker", "model", "MAE"}.issubset(df.columns):
        return {}
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    stab = (
        df.dropna(subset=["ticker", "model", "MAE"])
        .groupby(["ticker", "model"], as_index=False)["MAE"]
        .mean()
    )
    return {(r["ticker"], r["model"]): float(r["MAE"]) for _, r in stab.iterrows()}


def _sign(x: float, eps: float = 1e-6) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _evaluate_strategies(
    ticker_df: pd.DataFrame,
    model_scores: dict[tuple[str, str], float],
    ranked_models: dict[str, list[str]],
    stability_scores: dict[tuple[str, str], float],
) -> dict:
    ticker = str(ticker_df["ticker"].iloc[0])
    actual = float(ticker_df["actual"].iloc[0])

    by_model = {str(r["model"]): float(r["pred"]) for _, r in ticker_df.iterrows()}
    ranked = ranked_models.get(ticker, sorted(by_model.keys()))
    ranked_present = [m for m in ranked if m in by_model]
    if not ranked_present:
        ranked_present = sorted(by_model.keys())

    # Strategy 1: winner_take_all (best MAE model wins)
    best_model = ranked_present[0]
    s1 = _sign(by_model[best_model] - actual)

    # Strategy 2: top3 ensemble
    top3 = ranked_present[:3] if len(ranked_present) >= 3 else ranked_present
    s2 = _sign(float(np.mean([by_model[m] for m in top3])) - actual) if top3 else 0

    # Strategy 3: broad consensus vote
    votes = [_sign(v - actual) for v in by_model.values()]
    s3 = _sign(sum(votes)) if votes else 0

    # Strategy 4: risk-adjusted edge from MAE-normalized prediction edge
    best_edge = 0.0
    best_edge_signal = 0
    for model, pred in by_model.items():
        mae = model_scores.get((ticker, model), np.nan)
        if np.isnan(mae) or mae <= 0:
            mae = max(abs(actual) * 0.01, 1e-6)
        edge = (pred - actual) / mae
        if abs(edge) > abs(best_edge):
            best_edge = edge
            best_edge_signal = _sign(edge)
    s4 = best_edge_signal if abs(best_edge) >= 0.25 else 0

    # Strategy 5: stability leader (lowest recent edge MAE)
    stability_rank = sorted(
        by_model.keys(),
        key=lambda m: stability_scores.get((ticker, m), model_scores.get((ticker, m), 9999.0)),
    )
    stable_model = stability_rank[0]
    stable_move = (by_model[stable_model] - actual) / max(abs(actual), 1e-6)
    s5 = _sign(stable_move) if abs(stable_move) >= 0.002 else 0

    signals = {
        "winner_take_all": s1,
        "top3_ensemble": s2,
        "consensus_vote": s3,
        "risk_adjusted_edge": s4,
        "stability_filter": s5,
    }

    weights = {
        "winner_take_all": 1.2,
        "top3_ensemble": 1.1,
        "consensus_vote": 0.8,
        "risk_adjusted_edge": 1.0,
        "stability_filter": 0.9,
    }
    score = float(sum(signals[k] * w for k, w in weights.items()))

    return {
        "ticker": ticker,
        "actual": actual,
        "signals": signals,
        "score": score,
        "best_model": best_model,
        "top3_models": top3,
        "models_considered": sorted(by_model.keys()),
        "pred_avg": float(np.mean(list(by_model.values()))),
    }


def _commission(notional: float, cfg: TradingConfig, side: str) -> float:
    base = cfg.commission_fixed + notional * (cfg.commission_bps / 10_000)
    if side == "SELL":
        base += notional * (cfg.sec_fee_bps / 10_000)
    return round(base, 4)


def _execution_price(px: float, cfg: TradingConfig, side: str) -> float:
    total_bps = (cfg.spread_bps + cfg.slippage_bps) / 10_000
    if side == "BUY":
        return px * (1 + total_bps)
    return px * (1 - total_bps)


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    running_max = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        running_max = max(running_max, value)
        if running_max <= 0:
            continue
        drawdown = (running_max - value) / running_max
        max_dd = max(max_dd, drawdown)
    return float(max_dd)


def backtest_strategies(*, lookback_days: int = 15, dry_run: bool = False) -> dict[str, Path | None]:
    """Backtest each individual strategy for the latest ``lookback_days`` business days."""
    cfg = load_trading_config()
    preds = _load_prediction_files()
    if preds.empty:
        logger.warning("No predictions available for strategy backtest")
        return {"summary": None, "daily": None, "trades": None}

    model_scores, ranked_models = _load_latest_model_scores()
    stability_scores = _load_stability_scores()

    all_days = sorted(pd.Timestamp(d).normalize() for d in preds["predicted_date"].dropna().unique())
    trading_days = all_days[-lookback_days:]
    if not trading_days:
        logger.warning("No trading days available for strategy backtest")
        return {"summary": None, "daily": None, "trades": None}

    close_price_by_day: dict[pd.Timestamp, dict[str, float]] = {}
    for _, row in preds.iterrows():
        close_day = pd.Timestamp(row["predicted_date"]).normalize() - pd.offsets.BDay(1)
        close_price_by_day.setdefault(close_day, {})[str(row["ticker"])] = float(row["actual"])

    summary_rows: list[dict] = []
    daily_rows: list[dict] = []
    trade_rows: list[dict] = []

    for strategy_name in STRATEGY_NAMES:
        cash = cfg.initial_budget
        open_positions: dict[str, dict] = {}
        equity_curve: list[float] = []

        strat_realized_pnl = 0.0
        strat_fees = 0.0
        strat_buys = 0
        strat_sells = 0
        strat_wins = 0

        for day in trading_days:
            day_preds = preds[preds["predicted_date"] == day]

            day_realized = 0.0
            day_fees = 0.0
            day_buys = 0
            day_sells = 0

            for ticker, pos in list(open_positions.items()):
                if pd.Timestamp(pos["exit_due"]).normalize() > day:
                    continue
                close_px = close_price_by_day.get(day, {}).get(ticker)
                if close_px is None:
                    continue
                exec_sell = _execution_price(close_px, cfg, "SELL")
                notional = exec_sell * pos["shares"]
                fee = _commission(notional, cfg, "SELL")
                gross_pnl = (exec_sell - pos["entry_price_exec"]) * pos["shares"]
                net_pnl = gross_pnl - fee
                cash += notional - fee

                day_sells += 1
                day_realized += net_pnl
                day_fees += fee
                if net_pnl > 0:
                    strat_wins += 1

                trade_rows.append(
                    {
                        "strategy": strategy_name,
                        "event": "SELL",
                        "date": day.date().isoformat(),
                        "ticker": ticker,
                        "shares": pos["shares"],
                        "signal": pos["entry_signal"],
                        "signal_price": round(close_px, 6),
                        "execution_price": round(exec_sell, 6),
                        "notional": round(notional, 4),
                        "commission": fee,
                        "gross_pnl": round(gross_pnl, 4),
                        "net_pnl": round(net_pnl, 4),
                        "cash_after": round(cash, 4),
                        "close_reason": "time_exit",
                    }
                )
                del open_positions[ticker]

            decisions: list[dict] = []
            for _, grp in day_preds.groupby("ticker"):
                decisions.append(_evaluate_strategies(grp, model_scores, ranked_models, stability_scores))
            decisions.sort(key=lambda d: abs(d["score"]), reverse=True)
            decisions = decisions[: cfg.max_tickers_per_day]

            for decision in decisions:
                ticker = decision["ticker"]
                signal = int(decision["signals"].get(strategy_name, 0))
                signal_px = float(decision["actual"])

                if ticker in open_positions and signal < 0:
                    pos = open_positions[ticker]
                    exec_sell = _execution_price(signal_px, cfg, "SELL")
                    notional = exec_sell * pos["shares"]
                    fee = _commission(notional, cfg, "SELL")
                    gross_pnl = (exec_sell - pos["entry_price_exec"]) * pos["shares"]
                    net_pnl = gross_pnl - fee
                    cash += notional - fee

                    day_sells += 1
                    day_realized += net_pnl
                    day_fees += fee
                    if net_pnl > 0:
                        strat_wins += 1

                    trade_rows.append(
                        {
                            "strategy": strategy_name,
                            "event": "SELL",
                            "date": day.date().isoformat(),
                            "ticker": ticker,
                            "shares": pos["shares"],
                            "signal": signal,
                            "signal_price": round(signal_px, 6),
                            "execution_price": round(exec_sell, 6),
                            "notional": round(notional, 4),
                            "commission": fee,
                            "gross_pnl": round(gross_pnl, 4),
                            "net_pnl": round(net_pnl, 4),
                            "cash_after": round(cash, 4),
                            "close_reason": "bearish_signal",
                        }
                    )
                    del open_positions[ticker]
                    continue

                if ticker in open_positions:
                    continue
                if signal <= 0:
                    continue

                target_notional = min(cash * cfg.max_position_pct, cfg.initial_budget * cfg.max_position_pct)
                if target_notional < cfg.min_trade_usd:
                    continue

                exec_buy = _execution_price(signal_px, cfg, "BUY")
                shares = int(target_notional // exec_buy)
                if shares <= 0:
                    continue
                notional = shares * exec_buy
                fee = _commission(notional, cfg, "BUY")
                total_cost = notional + fee
                if total_cost > cash:
                    continue

                cash -= total_cost
                day_buys += 1
                day_fees += fee

                exit_due = _business_day(day, cfg.holding_days)
                open_positions[ticker] = {
                    "entry_date": day.date().isoformat(),
                    "exit_due": exit_due.date().isoformat(),
                    "entry_price_exec": exec_buy,
                    "shares": shares,
                    "entry_signal": signal,
                }

                trade_rows.append(
                    {
                        "strategy": strategy_name,
                        "event": "BUY",
                        "date": day.date().isoformat(),
                        "ticker": ticker,
                        "shares": shares,
                        "signal": signal,
                        "signal_price": round(signal_px, 6),
                        "execution_price": round(exec_buy, 6),
                        "notional": round(notional, 4),
                        "commission": fee,
                        "gross_pnl": 0.0,
                        "net_pnl": -fee,
                        "cash_after": round(cash, 4),
                        "close_reason": "",
                    }
                )

            mtm_value = 0.0
            for ticker, pos in open_positions.items():
                px = close_price_by_day.get(day, {}).get(ticker, pos["entry_price_exec"])
                mtm_value += pos["shares"] * px

            equity = cash + mtm_value
            equity_curve.append(equity)

            daily_rows.append(
                {
                    "strategy": strategy_name,
                    "date": day.date().isoformat(),
                    "buys": day_buys,
                    "sells": day_sells,
                    "open_positions": len(open_positions),
                    "realized_pnl": round(day_realized, 4),
                    "fees_paid": round(day_fees, 4),
                    "cash": round(cash, 4),
                    "portfolio_mtm": round(mtm_value, 4),
                    "equity": round(equity, 4),
                }
            )

            strat_buys += day_buys
            strat_sells += day_sells
            strat_realized_pnl += day_realized
            strat_fees += day_fees

        closed_trades = max(strat_sells, 1)
        final_equity = equity_curve[-1] if equity_curve else cfg.initial_budget
        summary_rows.append(
            {
                "strategy": strategy_name,
                "lookback_days": lookback_days,
                "start_date": trading_days[0].date().isoformat(),
                "end_date": trading_days[-1].date().isoformat(),
                "buys": strat_buys,
                "sells": strat_sells,
                "fees_paid": round(strat_fees, 4),
                "realized_pnl": round(strat_realized_pnl, 4),
                "final_equity": round(final_equity, 4),
                "total_return_pct": round(((final_equity / cfg.initial_budget) - 1) * 100, 4),
                "max_drawdown_pct": round(_max_drawdown(equity_curve) * 100, 4),
                "win_rate_pct": round((strat_wins / closed_trades) * 100, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("total_return_pct", ascending=False)
    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trade_rows)

    summary_file = ACTIONS_DIR / f"strategy_backtest_{lookback_days}d_summary.csv"
    daily_file = ACTIONS_DIR / f"strategy_backtest_{lookback_days}d_daily.csv"
    trades_file = ACTIONS_DIR / f"strategy_backtest_{lookback_days}d_trades.csv"

    if not dry_run:
        summary_df.to_csv(summary_file, index=False)
        daily_df.to_csv(daily_file, index=False)
        trades_df.to_csv(trades_file, index=False)
        logger.info(
            "Saved strategy backtest outputs: %s, %s, %s",
            summary_file,
            daily_file,
            trades_file,
        )

    return {"summary": summary_file, "daily": daily_file, "trades": trades_file}


def run_actions(*, dry_run: bool = False) -> dict[str, Path | None]:
    cfg = load_trading_config()
    preds = _load_prediction_files()
    if preds.empty:
        logger.warning("No predictions available for paper trading actions")
        return {"trades": None, "daily": None, "positions": None}

    model_scores, ranked_models = _load_latest_model_scores()
    stability_scores = _load_stability_scores()

    trading_days = sorted(preds["predicted_date"].dropna().unique())

    close_price_by_day: dict[pd.Timestamp, dict[str, float]] = {}
    for _, row in preds.iterrows():
        close_day = pd.Timestamp(row["predicted_date"]).normalize() - pd.offsets.BDay(1)
        close_price_by_day.setdefault(close_day, {})[str(row["ticker"])] = float(row["actual"])

    cash = cfg.initial_budget
    open_positions: dict[str, dict] = {}
    trade_rows: list[dict] = []
    daily_rows: list[dict] = []

    for day in trading_days:
        day = pd.Timestamp(day).normalize()
        day_preds = preds[preds["predicted_date"] == day]

        closed_today = 0
        realized_pnl = 0.0
        total_fees = 0.0

        # Exit matured positions first.
        for ticker, pos in list(open_positions.items()):
            if pd.Timestamp(pos["exit_due"]).normalize() > day:
                continue
            close_px = close_price_by_day.get(day, {}).get(ticker)
            if close_px is None:
                continue
            exec_sell = _execution_price(close_px, cfg, "SELL")
            notional = exec_sell * pos["shares"]
            fee = _commission(notional, cfg, "SELL")
            gross_pnl = (exec_sell - pos["entry_price_exec"]) * pos["shares"]
            net_pnl = gross_pnl - fee
            cash += notional - fee

            closed_today += 1
            realized_pnl += net_pnl
            total_fees += fee

            trade_rows.append(
                {
                    "event": "SELL",
                    "date": day.date().isoformat(),
                    "ticker": ticker,
                    "strategy_score": pos["strategy_score"],
                    "strategy_signals": json.dumps(pos["strategy_signals"], sort_keys=True),
                    "best_model": pos["best_model"],
                    "entry_date": pos["entry_date"],
                    "exit_date": day.date().isoformat(),
                    "holding_days": int(pos["holding_days"]),
                    "shares": pos["shares"],
                    "signal_price": round(close_px, 6),
                    "execution_price": round(exec_sell, 6),
                    "notional": round(notional, 4),
                    "commission": fee,
                    "gross_pnl": round(gross_pnl, 4),
                    "net_pnl": round(net_pnl, 4),
                    "cash_after": round(cash, 4),
                    "position_after": 0,
                    "close_reason": "time_exit",
                }
            )
            del open_positions[ticker]

        decisions: list[dict] = []
        for ticker, grp in day_preds.groupby("ticker"):
            decisions.append(_evaluate_strategies(grp, model_scores, ranked_models, stability_scores))

        decisions.sort(key=lambda d: abs(d["score"]), reverse=True)
        decisions = decisions[: cfg.max_tickers_per_day]

        buys_today = 0
        sells_today = 0

        for decision in decisions:
            ticker = decision["ticker"]
            score = float(decision["score"])
            signal_px = float(decision["actual"])

            if ticker in open_positions and score <= cfg.sell_threshold:
                # Early sell can be triggered by a strong bearish composite score.
                exec_sell = _execution_price(signal_px, cfg, "SELL")
                pos = open_positions[ticker]
                notional = exec_sell * pos["shares"]
                fee = _commission(notional, cfg, "SELL")
                gross_pnl = (exec_sell - pos["entry_price_exec"]) * pos["shares"]
                net_pnl = gross_pnl - fee
                cash += notional - fee

                sells_today += 1
                realized_pnl += net_pnl
                total_fees += fee

                trade_rows.append(
                    {
                        "event": "SELL",
                        "date": day.date().isoformat(),
                        "ticker": ticker,
                        "strategy_score": score,
                        "strategy_signals": json.dumps(decision["signals"], sort_keys=True),
                        "best_model": decision["best_model"],
                        "entry_date": pos["entry_date"],
                        "exit_date": day.date().isoformat(),
                        "holding_days": int((day - pd.Timestamp(pos["entry_date"])).days),
                        "shares": pos["shares"],
                        "signal_price": round(signal_px, 6),
                        "execution_price": round(exec_sell, 6),
                        "notional": round(notional, 4),
                        "commission": fee,
                        "gross_pnl": round(gross_pnl, 4),
                        "net_pnl": round(net_pnl, 4),
                        "cash_after": round(cash, 4),
                        "position_after": 0,
                        "close_reason": "bearish_signal",
                    }
                )
                del open_positions[ticker]
                continue

            if ticker in open_positions:
                continue

            if score < cfg.buy_threshold:
                continue

            target_notional = min(cash * cfg.max_position_pct, cfg.initial_budget * cfg.max_position_pct)
            if target_notional < cfg.min_trade_usd:
                continue

            exec_buy = _execution_price(signal_px, cfg, "BUY")
            shares = int(target_notional // exec_buy)
            if shares <= 0:
                continue
            notional = shares * exec_buy
            fee = _commission(notional, cfg, "BUY")
            total_cost = notional + fee
            if total_cost > cash:
                continue

            cash -= total_cost
            buys_today += 1
            total_fees += fee

            exit_due = _business_day(day, cfg.holding_days)
            open_positions[ticker] = {
                "entry_date": day.date().isoformat(),
                "exit_due": exit_due.date().isoformat(),
                "holding_days": cfg.holding_days,
                "entry_price_exec": exec_buy,
                "shares": shares,
                "strategy_score": score,
                "strategy_signals": decision["signals"],
                "best_model": decision["best_model"],
            }

            trade_rows.append(
                {
                    "event": "BUY",
                    "date": day.date().isoformat(),
                    "ticker": ticker,
                    "strategy_score": score,
                    "strategy_signals": json.dumps(decision["signals"], sort_keys=True),
                    "best_model": decision["best_model"],
                    "entry_date": day.date().isoformat(),
                    "exit_date": exit_due.date().isoformat(),
                    "holding_days": cfg.holding_days,
                    "shares": shares,
                    "signal_price": round(signal_px, 6),
                    "execution_price": round(exec_buy, 6),
                    "notional": round(notional, 4),
                    "commission": fee,
                    "gross_pnl": 0.0,
                    "net_pnl": -fee,
                    "cash_after": round(cash, 4),
                    "position_after": shares,
                    "close_reason": "",
                }
            )

        mtm_value = 0.0
        for ticker, pos in open_positions.items():
            px = close_price_by_day.get(day, {}).get(ticker, pos["entry_price_exec"])
            mtm_value += pos["shares"] * px

        daily_rows.append(
            {
                "date": day.date().isoformat(),
                "buys": buys_today,
                "sells": sells_today,
                "scheduled_exits": closed_today,
                "open_positions": len(open_positions),
                "realized_pnl": round(realized_pnl, 4),
                "fees_paid": round(total_fees, 4),
                "cash": round(cash, 4),
                "portfolio_mtm": round(mtm_value, 4),
                "equity": round(cash + mtm_value, 4),
            }
        )

    positions_rows = []
    for ticker, pos in open_positions.items():
        positions_rows.append(
            {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_due": pos["exit_due"],
                "shares": pos["shares"],
                "entry_price_exec": round(pos["entry_price_exec"], 6),
                "strategy_score": pos["strategy_score"],
                "strategy_signals": json.dumps(pos["strategy_signals"], sort_keys=True),
                "best_model": pos["best_model"],
            }
        )

    trades_df = pd.DataFrame(trade_rows)
    daily_df = pd.DataFrame(daily_rows)
    positions_df = pd.DataFrame(positions_rows)

    trades_file = ACTIONS_DIR / "paper_trades.csv"
    daily_file = ACTIONS_DIR / "daily_activity.csv"
    positions_file = ACTIONS_DIR / "open_positions.csv"

    if not dry_run:
        trades_df.to_csv(trades_file, index=False)
        daily_df.to_csv(daily_file, index=False)
        positions_df.to_csv(positions_file, index=False)
        logger.info("Saved Actions outputs: %s, %s, %s", trades_file, daily_file, positions_file)

    return {"trades": trades_file, "daily": daily_file, "positions": positions_file}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper trading Actions module")
    parser.add_argument("--dry-run", action="store_true", help="Run calculations without writing output CSV files")
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=15,
        help="Number of latest trading days used to backtest each strategy",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_actions(dry_run=args.dry_run)
    backtest_strategies(lookback_days=args.backtest_days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
