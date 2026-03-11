import json
import pandas as pd
import src.visualization as viz
def test_prepare_pipeline_health_writes_summary(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    feature_dir = root / "features"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, feature_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_date = "2026-03-06"
    pd.DataFrame(
        [{"ticker": "AAA", "actual": 100.0, "pred": 101.0, "Predicted": run_date}]
    ).to_csv(pred_dir / f"{run_date}_daily_predictions.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(feature_dir / f"features_daily_{run_date}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(metrics_dir / f"metrics_daily_{run_date}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(edge_dir / f"edge_metrics_{run_date}.csv", index=False)
    (viz_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "FEATURE_DIR", feature_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    out = viz.prepare_pipeline_health()
    assert out is not None
    assert list(out.columns) == [
        "run_date",
        "duration_minutes",
        "success_pct",
        "successful_steps",
        "total_steps",
        "fallback_offline",
        "status",
    ]
    assert out.loc[0, "run_date"] == run_date
    assert out.loc[0, "successful_steps"] >= 4
    assert (viz_dir / "pipeline_health.csv").exists()
def test_prepare_last_run_report_writes_json(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_date = "2026-03-06"
    pd.DataFrame(
        [{"ticker": "AAA", "actual": 100.0, "pred": 101.0, "Predicted": run_date}]
    ).to_csv(pred_dir / f"{run_date}_daily_predictions.csv", index=False)
    pd.DataFrame(
        [{"model": "AAA_rf", "dataset": "test", "MAE": 1.1, "RMSE": 1.3, "MAPE": 2.0, "R2": 0.85}]
    ).to_csv(metrics_dir / f"metrics_daily_{run_date}.csv", index=False)
    pd.DataFrame(
        [{"ticker": "AAA", "model": "rf", "MAE": 1.0}]
    ).to_csv(edge_dir / f"edge_metrics_{run_date}.csv", index=False)
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    health_df = pd.DataFrame(
        [{
            "run_date": run_date,
            "duration_minutes": 5.5,
            "success_pct": 100.0,
            "successful_steps": 5,
            "total_steps": 5,
            "fallback_offline": "No detectado",
            "status": "SALUDABLE",
        }]
    )
    action_df = pd.DataFrame(
        [{
            "date": run_date,
            "ticker": "AAA",
            "action": "BUY",
            "strategy_score": 4.2,
            "ret_1d": "1.00%",
            "ret_5d": "2.00%",
            "ret_20d": "3.00%",
            "result_5d": "Acierto",
        }]
    )
    strategy_df = pd.DataFrame(
        [{"strategy": "winner_take_all", "ending_equity": 11000, "return_pct": 10.0, "win_rate": 55.0, "max_drawdown": 0.1}]
    )
    report = viz.prepare_last_run_report(health_df, action_df, strategy_df)
    assert report is not None
    report_path = viz_dir / "last_run_report.json"
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["run_date"] == run_date
    assert loaded["summary"]["recommended_tickers"] == 1
    assert loaded["summary"]["actions"]["BUY"] == 1
    assert loaded["summary"]["edge_coverage"]["by_model"][0]["model"] == "rf"
def test_prepare_pipeline_health_uses_previous_available_artifacts(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    feature_dir = root / "features"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, feature_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    pred_run = "2026-03-07"
    prev_run = "2026-03-06"
    pd.DataFrame([{"ticker": "AAA", "actual": 100.0, "pred": 101.0, "Predicted": pred_run}]).to_csv(
        pred_dir / f"{pred_run}_daily_predictions.csv", index=False
    )
    pd.DataFrame([{"x": 1}]).to_csv(feature_dir / f"features_daily_{prev_run}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(metrics_dir / f"metrics_daily_{prev_run}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(edge_dir / f"edge_metrics_{prev_run}.csv", index=False)
    (viz_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "FEATURE_DIR", feature_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    out = viz.prepare_pipeline_health()
    assert out is not None
    assert out.loc[0, "run_date"] == pred_run
    assert out.loc[0, "successful_steps"] == 5
    assert out.loc[0, "success_pct"] == 100.0
def test_prepare_last_run_report_falls_back_to_latest_actions(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_date = "2026-03-07"
    prev_date = "2026-03-06"
    pd.DataFrame([{"ticker": "AAA", "actual": 100.0, "pred": 101.0, "Predicted": run_date}]).to_csv(
        pred_dir / f"{run_date}_daily_predictions.csv", index=False
    )
    pd.DataFrame([{"model": "AAA_rf", "dataset": "test", "MAE": 1.1}]).to_csv(
        metrics_dir / f"metrics_daily_{prev_date}.csv", index=False
    )
    pd.DataFrame([{"ticker": "AAA", "model": "rf", "MAE": 1.0}]).to_csv(
        edge_dir / f"edge_metrics_{prev_date}.csv", index=False
    )
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    health_df = pd.DataFrame([
        {
            "run_date": run_date,
            "duration_minutes": 5.5,
            "success_pct": 100.0,
            "successful_steps": 5,
            "total_steps": 5,
            "fallback_offline": "No detectado",
            "status": "SALUDABLE",
        }
    ])
    action_df = pd.DataFrame([
        {
            "date": prev_date,
            "ticker": "AAA",
            "action": "BUY",
            "strategy_score": 4.2,
            "ret_1d": "1.00%",
            "ret_5d": "2.00%",
            "ret_20d": "3.00%",
            "result_5d": "Acierto",
        }
    ])
    report = viz.prepare_last_run_report(health_df, action_df, pd.DataFrame())
    assert report is not None
    assert report["summary"]["recommended_tickers"] == 1
    assert report["summary"]["actions"]["BUY"] == 1
    assert len(report["top_recommendations"]) == 1
    assert report["top_recommendations"][0]["ticker"] == "AAA"
    assert "by_model" in report["summary"]["edge_coverage"]
def test_prepare_last_run_report_uses_edge_metrics_when_metrics_are_missing(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_date = "2026-03-07"
    pd.DataFrame([{"ticker": "AAA", "actual": 100.0, "pred": 101.0, "Predicted": run_date}]).to_csv(
        pred_dir / f"{run_date}_daily_predictions.csv", index=False
    )
    pd.DataFrame([{"foo": 1}]).to_csv(metrics_dir / f"metrics_daily_{run_date}.csv", index=False)
    pd.DataFrame([
        {"ticker": "AAA", "model": "rf", "MAE": 1.0, "RMSE": 1.2, "MAPE": 2.0, "R2": 0.8},
        {"ticker": "BBB", "model": "rf", "MAE": 1.2, "RMSE": 1.4, "MAPE": 2.2, "R2": 0.7},
        {"ticker": "CCC", "model": "xgb", "MAE": 0.8, "RMSE": 1.0, "MAPE": 1.8, "R2": 0.85},
    ]).to_csv(edge_dir / f"edge_metrics_{run_date}.csv", index=False)
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    report = viz.prepare_last_run_report(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert report is not None
    assert len(report["model_metrics"]) == 2
    assert report["model_metrics"][0]["model"] == "xgb"
    assert report["model_metrics"][0]["MAE"] == 0.8
def test_prepare_action_recommendations_uses_run_date_for_output_date(monkeypatch, tmp_path):
    run_date = pd.Timestamp("2026-03-11")
    predicted_date = pd.Timestamp("2026-03-12")
    preds = pd.DataFrame([
        {
            "ticker": "AAA",
            "model": "rf",
            "actual": 100.0,
            "pred": 103.0,
            "run_date": run_date,
            "predicted_date": predicted_date,
        },
        {
            "ticker": "AAA",
            "model": "xgb",
            "actual": 100.0,
            "pred": 102.0,
            "run_date": run_date,
            "predicted_date": predicted_date,
        },
    ])
    class _Cfg:
        buy_threshold = 2.8
        sell_threshold = -1.8
    def _fake_eval(*args, **kwargs):
        return {"ticker": "AAA", "best_model": "rf", "score": 4.0, "actual": 100.0}
    import src.actions.paper_trader as trader
    monkeypatch.setattr(trader, "_load_prediction_files", lambda: preds)
    monkeypatch.setattr(trader, "load_trading_config", lambda: _Cfg())
    monkeypatch.setattr(trader, "_load_latest_model_scores", lambda: ({}, {}))
    monkeypatch.setattr(trader, "_load_stability_scores", lambda: {})
    monkeypatch.setattr(trader, "_evaluate_strategies", _fake_eval)
    monkeypatch.setattr(viz, "VIZ_DIR", tmp_path)
    out = viz.prepare_action_recommendations()
    assert out is not None
    assert not out.empty
    assert out.iloc[0]["date"] == "2026-03-11"
def test_prepare_pipeline_health_ignores_latest_empty_prediction_file(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    feature_dir = root / "features"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, feature_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    valid_run = "2026-03-06"
    empty_run = "2026-03-11"
    pd.DataFrame([{"ticker": "AAA", "model": "rf", "actual": 100.0, "pred": 101.0, "Predicted": valid_run}]).to_csv(
        pred_dir / f"{valid_run}_daily_predictions.csv", index=False
    )
    (pred_dir / f"{empty_run}_daily_predictions.csv").write_text("\n", encoding="utf-8")
    pd.DataFrame([{"x": 1}]).to_csv(feature_dir / f"features_daily_{valid_run}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(metrics_dir / f"metrics_daily_{valid_run}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(edge_dir / f"edge_metrics_{valid_run}.csv", index=False)
    (viz_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "FEATURE_DIR", feature_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    out = viz.prepare_pipeline_health()
    assert out is not None
    assert out.loc[0, "run_date"] == valid_run
def test_prepare_last_run_report_ignores_latest_empty_prediction_file(tmp_path, monkeypatch):
    root = tmp_path
    pred_dir = root / "predicts"
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [pred_dir, metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    valid_run = "2026-03-06"
    empty_run = "2026-03-11"
    pd.DataFrame([{"ticker": "AAA", "model": "rf", "actual": 100.0, "pred": 101.0, "Predicted": valid_run}]).to_csv(
        pred_dir / f"{valid_run}_daily_predictions.csv", index=False
    )
    (pred_dir / f"{empty_run}_daily_predictions.csv").write_text("\n", encoding="utf-8")
    pd.DataFrame([{"model": "AAA_rf", "dataset": "test", "MAE": 1.1, "RMSE": 1.3, "MAPE": 2.0, "R2": 0.85}]).to_csv(
        metrics_dir / f"metrics_daily_{valid_run}.csv", index=False
    )
    pd.DataFrame([{"ticker": "AAA", "model": "rf", "MAE": 1.0}]).to_csv(
        edge_dir / f"edge_metrics_{valid_run}.csv", index=False
    )
    monkeypatch.setattr(viz, "PRED_DIR", pred_dir)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    report = viz.prepare_last_run_report(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert report is not None
    assert report["run_date"] == valid_run
    assert report["artifacts"]["predictions_file"] == f"{valid_run}_daily_predictions.csv"
def test_prepare_edge_metrics_reads_from_edge_metrics_dir(tmp_path, monkeypatch):
    root = tmp_path
    metrics_dir = root / "metrics"
    edge_dir = root / "edge_metrics"
    viz_dir = root / "viz"
    for directory in [metrics_dir, edge_dir, viz_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_date = "2026-03-06"
    pd.DataFrame([
        {
            "ticker": "AAA",
            "model": "linreg",
            "pred": 101.0,
            "real": 100.0,
            "pred_inc": 0.01,
            "Predicted": run_date,
            "MAE": 1.0,
            "MAPE": 0.01,
            "RMSE": 1.0,
            "direction": True,
        }
    ]).to_csv(edge_dir / f"edge_metrics_{run_date}.csv", index=False)
    # Archivo señuelo en metrics para asegurar que no se usa esa carpeta.
    pd.DataFrame([
        {
            "ticker": "ZZZ",
            "model": "rf",
            "pred": 1,
            "real": 1,
            "pred_inc": 0,
            "Predicted": run_date,
            "MAE": 0,
            "MAPE": 0,
            "RMSE": 0,
            "direction": False,
        }
    ]).to_csv(metrics_dir / f"edge_metrics_{run_date}.csv", index=False)
    monkeypatch.setattr(viz, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(viz, "EDGE_METRICS_DIR", edge_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    out = viz.prepare_edge_metrics(max_files=5)
    assert out is not None
    assert not out.empty
    assert set(out["ticker"]) == {"AAA"}
def test_prepare_strategy_performance_fills_missing_strategies(tmp_path, monkeypatch):
    root = tmp_path
    actions_dir = root / "actions"
    viz_dir = root / "viz"
    actions_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"strategy": "winner_take_all", "ending_equity": 11000, "return_pct": 10.0, "win_rate": 55.0, "max_drawdown": 0.10},
        {"strategy": "equal_weight", "ending_equity": 10800, "return_pct": 8.0, "win_rate": 53.0, "max_drawdown": 0.12},
        {"strategy": "risk_parity", "ending_equity": 10700, "return_pct": 7.0, "win_rate": 52.0, "max_drawdown": 0.13},
        {"strategy": "momentum_tilt", "ending_equity": 10600, "return_pct": 6.0, "win_rate": 51.0, "max_drawdown": 0.14},
    ]).to_csv(actions_dir / "strategy_backtest_5d_summary.csv", index=False)
    monkeypatch.setattr(viz, "ACTIONS_DIR", actions_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    out = viz.prepare_strategy_performance()
    assert out is not None
    assert list(out.columns) == ["strategy", "ending_equity", "return_pct", "win_rate", "max_drawdown", "initial_budget", "max_position_pct", "min_trade_usd", "holding_days"]
    assert len(out) == 5
    assert "winner_take_all" in set(out["strategy"])
    assert "top3_ensemble" in set(out["strategy"])
    assert float(out["initial_budget"].max()) > 0
    assert float(out["max_position_pct"].max()) > 0
def test_prepare_strategy_performance_returns_default_when_missing_file(tmp_path, monkeypatch):
    root = tmp_path
    actions_dir = root / "actions"
    viz_dir = root / "viz"
    actions_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(viz, "ACTIONS_DIR", actions_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    out = viz.prepare_strategy_performance()
    assert out is not None
    assert len(out) == 5
    assert float(out["return_pct"].sum()) == 0.0
def test_prepare_strategy_performance_accepts_backtest_alias_columns(tmp_path, monkeypatch):
    root = tmp_path
    actions_dir = root / "actions"
    viz_dir = root / "viz"
    actions_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"strategy": "winner_take_all", "final_equity": 11100, "total_return_pct": 11.0, "win_rate_pct": 60.0, "max_drawdown_pct": 5.0},
    ]).to_csv(actions_dir / "strategy_backtest_5d_summary.csv", index=False)
    monkeypatch.setattr(viz, "ACTIONS_DIR", actions_dir)
    monkeypatch.setattr(viz, "VIZ_DIR", viz_dir)
    out = viz.prepare_strategy_performance()
    assert out is not None
    row = out[out["strategy"] == "winner_take_all"].iloc[0]
    assert float(row["ending_equity"]) == 11100.0
    assert float(row["return_pct"]) == 11.0
    assert float(row["win_rate"]) == 60.0
    assert float(row["max_drawdown"]) == 0.05
def test_prepare_strategy_performance_details_creates_dedicated_folder(tmp_path, monkeypatch):
    root = tmp_path
    strategy_details_dir = root / "viz" / "strategy_performance"
    strategy_df = pd.DataFrame([
        {"strategy": "winner_take_all", "initial_budget": 10000, "ending_equity": 11000, "return_pct": 10.0}
    ])
    action_df = pd.DataFrame([
        {"date": "2026-03-06", "ticker": "AAA", "action": "BUY", "strategy_score": 4.1},
        {"date": "2026-03-06", "ticker": "BBB", "action": "SELL", "strategy_score": 3.7},
    ])
    monkeypatch.setattr(viz, "STRATEGY_DETAILS_DIR", strategy_details_dir)
    viz.prepare_strategy_performance_details(strategy_df, action_df)
    budget_file = strategy_details_dir / "budget_and_action.csv"
    history_file = strategy_details_dir / "action_history.csv"
    assert budget_file.exists()
    assert history_file.exists()
    budget = pd.read_csv(budget_file)
    assert "latest_action" in budget.columns
    assert budget.loc[0, "latest_action_date"] == "2026-03-06"
    history = pd.read_csv(history_file)
    assert list(history.columns) == ["date", "buy_count", "sell_count", "hold_count", "tickers", "avg_strategy_score"]
    assert int(history.loc[0, "buy_count"]) == 1
