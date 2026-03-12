import json
import os
from datetime import datetime, timedelta, timezone

from src.monitoring.dashboard_freshness import validate_dashboard_freshness


def test_validate_dashboard_freshness_passes_for_recent_files(tmp_path):
    health = tmp_path / "pipeline_health.csv"
    manifest = tmp_path / "manifest.json"
    last_run_report = tmp_path / "last_run_report.json"

    health.write_text("run_date,status\n2026-03-12,SALUDABLE\n", encoding="utf-8")
    manifest.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    last_run_report.write_text(
        json.dumps({"run_date": datetime.now(timezone.utc).date().isoformat()}),
        encoding="utf-8",
    )

    ok, errors = validate_dashboard_freshness(
        health_file=health,
        manifest_file=manifest,
        last_run_report_file=last_run_report,
        max_age_hours=36,
        max_run_lag_business_days=2,
    )

    assert ok is True
    assert errors == []


def test_validate_dashboard_freshness_fails_for_stale_files(tmp_path):
    health = tmp_path / "pipeline_health.csv"
    manifest = tmp_path / "manifest.json"
    last_run_report = tmp_path / "last_run_report.json"

    health.write_text("run_date,status\n2026-03-10,CRITICO\n", encoding="utf-8")
    manifest.write_text(
        json.dumps({"generated_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}),
        encoding="utf-8",
    )
    last_run_report.write_text(
        json.dumps({"run_date": "2026-03-03"}),
        encoding="utf-8",
    )

    stale = datetime.now().timestamp() - (72 * 60 * 60)
    health.touch()
    os.utime(health, (stale, stale))

    ok, errors = validate_dashboard_freshness(
        health_file=health,
        manifest_file=manifest,
        last_run_report_file=last_run_report,
        max_age_hours=36,
        max_run_lag_business_days=2,
    )

    assert ok is False
    assert len(errors) == 3
