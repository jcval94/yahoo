"""Validate dashboard health artifacts are refreshed daily."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEALTH_FILE = REPO_ROOT / "docs" / "viz" / "pipeline_health.csv"
DEFAULT_MANIFEST_FILE = REPO_ROOT / "docs" / "viz" / "manifest.json"


def _artifact_is_fresh(path: Path, max_age_hours: int, now: datetime) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return now - mtime <= timedelta(hours=max_age_hours)


def _manifest_generated_at_is_fresh(path: Path, max_age_hours: int, now: datetime) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        return False
    parsed = _parse_generated_at(generated_at.strip())
    if parsed is None:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return now - parsed.astimezone(timezone.utc) <= timedelta(hours=max_age_hours)


def _parse_generated_at(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass

    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def validate_dashboard_freshness(
    health_file: Path = DEFAULT_HEALTH_FILE,
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    max_age_hours: int = 36,
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

    return (not errors), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate daily freshness of dashboard health artifacts.")
    parser.add_argument("--health-file", type=Path, default=DEFAULT_HEALTH_FILE)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--max-age-hours", type=int, default=36)
    args = parser.parse_args()

    ok, errors = validate_dashboard_freshness(
        health_file=args.health_file,
        manifest_file=args.manifest_file,
        max_age_hours=args.max_age_hours,
    )
    if ok:
        print("Dashboard freshness check passed.")
        return 0

    for error in errors:
        print(error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
