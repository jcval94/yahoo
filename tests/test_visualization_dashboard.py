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
