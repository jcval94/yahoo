import json
from datetime import datetime, timedelta, timezone

from src.monitoring.dashboard_freshness import validate_dashboard_freshness


def test_validate_dashboard_freshness_passes_for_recent_files(tmp_path):
    health = tmp_path / "pipeline_health.csv"
    manifest = tmp_path / "manifest.json"

    health.write_text("run_date,status\n2026-03-12,SALUDABLE\n", encoding="utf-8")
    manifest.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )

    ok, errors = validate_dashboard_freshness(
        health_file=health,
        manifest_file=manifest,
        max_age_hours=36,
    )

    assert ok is True
    assert errors == []


def test_validate_dashboard_freshness_fails_for_stale_files(tmp_path):
    health = tmp_path / "pipeline_health.csv"
    manifest = tmp_path / "manifest.json"

    health.write_text("run_date,status\n2026-03-10,CRITICO\n", encoding="utf-8")
    manifest.write_text(
        json.dumps({"generated_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}),
        encoding="utf-8",
    )

    stale = datetime.now().timestamp() - (72 * 60 * 60)
    health.touch()
    import os

    os.utime(health, (stale, stale))

    ok, errors = validate_dashboard_freshness(
        health_file=health,
        manifest_file=manifest,
        max_age_hours=36,
    )

    assert ok is False
    assert len(errors) == 2


def test_validate_dashboard_freshness_accepts_compact_generated_at(tmp_path):
    health = tmp_path / "pipeline_health.csv"
    manifest = tmp_path / "manifest.json"

    health.write_text("run_date,status\n2026-03-12,SALUDABLE\n", encoding="utf-8")
    manifest.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}),
        encoding="utf-8",
    )

    ok, errors = validate_dashboard_freshness(
        health_file=health,
        manifest_file=manifest,
        max_age_hours=36,
    )

    assert ok is True
    assert errors == []


def test_validate_dashboard_freshness_handles_invalid_generated_at(tmp_path):
    health = tmp_path / "pipeline_health.csv"
    manifest = tmp_path / "manifest.json"

    health.write_text("run_date,status\n2026-03-12,SALUDABLE\n", encoding="utf-8")
    manifest.write_text(
        json.dumps({"generated_at": "not-a-date"}),
        encoding="utf-8",
    )

    ok, errors = validate_dashboard_freshness(
        health_file=health,
        manifest_file=manifest,
        max_age_hours=36,
    )

    assert ok is False
    assert errors == [
        f"Manifest generated_at is stale or invalid: {manifest} (max_age_hours=36)"
    ]
