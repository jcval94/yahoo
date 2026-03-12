"""Validate dashboard health artifacts are refreshed daily."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEALTH_FILE = REPO_ROOT / "docs" / "viz" / "pipeline_health.csv"
DEFAULT_MANIFEST_FILE = REPO_ROOT / "docs" / "viz" / "manifest.json"
DEFAULT_LAST_RUN_REPORT_FILE = REPO_ROOT / "docs" / "viz" / "last_run_report.json"


def _artifact_is_fresh(path: Path, max_age_hours: int, now: datetime) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return now - mtime <= timedelta(hours=max_age_hours)


def _parse_dashboard_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None

    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(normalized, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _manifest_generated_at_is_fresh(path: Path, max_age_hours: int, now: datetime) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        return False

    parsed = _parse_dashboard_timestamp(generated_at)
    if parsed is None:
        return False
    return now - parsed <= timedelta(hours=max_age_hours)


def _business_day_lag(start_date: datetime.date, end_date: datetime.date) -> int:
    if start_date >= end_date:
        return 0

    lag = 0
    current = start_date
    while current < end_date:
        current += timedelta(days=1)
        if current.weekday() < 5:
            lag += 1
    return lag


def _run_date_is_fresh(path: Path, max_lag_business_days: int, now: datetime) -> bool:
    if not path.exists():
        return False

    payload = json.loads(path.read_text(encoding="utf-8"))
    run_date = payload.get("run_date")
    if not isinstance(run_date, str) or not run_date.strip():
        return False

    parsed_run_date = datetime.strptime(run_date, "%Y-%m-%d").date()
    lag = _business_day_lag(parsed_run_date, now.date())
    return lag <= max_lag_business_days


def validate_dashboard_freshness(
    health_file: Path = DEFAULT_HEALTH_FILE,
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    last_run_report_file: Path = DEFAULT_LAST_RUN_REPORT_FILE,
    max_age_hours: int = 36,
    max_run_lag_business_days: int = 2,
) -> tuple[bool, list[str]]:
    now = datetime.now(timezone.utc)
    errors: list[str] = []

    if not _artifact_is_fresh(health_file, max_age_hours=max_age_hours, now=now):
        errors.append(
            f"Health panel file is stale or missing: {health_file} (max_age_hours={max_age_hours})"
        )

    if not _manifest_generated_at_is_fresh(manifest_file, max_age_hours=max_age_hours, now=now):
        errors.append(
            f"Manifest generated_at is stale or invalid: {manifest_file} (max_age_hours={max_age_hours})"
        )

    if not _run_date_is_fresh(
        last_run_report_file,
        max_lag_business_days=max_run_lag_business_days,
        now=now,
    ):
        errors.append(
            "Last run report run_date is stale, missing, or invalid: "
            f"{last_run_report_file} (max_run_lag_business_days={max_run_lag_business_days})"
        )

    return (not errors), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate daily freshness of dashboard health artifacts.")
    parser.add_argument("--health-file", type=Path, default=DEFAULT_HEALTH_FILE)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--last-run-report-file", type=Path, default=DEFAULT_LAST_RUN_REPORT_FILE)
    parser.add_argument("--max-age-hours", type=int, default=36)
    parser.add_argument("--max-run-lag-business-days", type=int, default=2)
    args = parser.parse_args()

    ok, errors = validate_dashboard_freshness(
        health_file=args.health_file,
        manifest_file=args.manifest_file,
        last_run_report_file=args.last_run_report_file,
        max_age_hours=args.max_age_hours,
        max_run_lag_business_days=args.max_run_lag_business_days,
    )
    if ok:
        print("Dashboard freshness check passed.")
        return 0

    for error in errors:
        print(error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
