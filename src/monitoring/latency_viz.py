"""Generate a simple latency time-series visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
LATENCY_DIR = ROOT_DIR / "results" / "latency"
RESULTS_VIZ_DIR = ROOT_DIR / "results" / "viz"
DOCS_VIZ_DIR = ROOT_DIR / "docs" / "viz"


def _iter_latency_files() -> list[Path]:
    return sorted(LATENCY_DIR.glob("latency_*.csv"))


def build_latency_timeseries() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for csv_file in _iter_latency_files():
        df = pd.read_csv(csv_file)
        if df.empty or "latency_minutes" not in df.columns:
            continue

        values = pd.to_numeric(df["latency_minutes"], errors="coerce").dropna()
        if values.empty:
            continue

        date_text = csv_file.stem.replace("latency_", "")
        rows.append(
            {
                "date": date_text,
                "mean_latency_minutes": round(float(values.mean()), 2),
                "p95_latency_minutes": round(float(values.quantile(0.95)), 2),
            }
        )

    output = pd.DataFrame(rows, columns=["date", "mean_latency_minutes", "p95_latency_minutes"])
    if not output.empty:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output = output.dropna(subset=["date"]).sort_values("date")
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    return output


def _write_placeholder(svg_path: Path) -> None:
    svg_path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='300'>"
        "<rect width='100%' height='100%' fill='#f4f6f8'/>"
        "<text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'"
        " font-size='20' fill='#667085'>Sin datos de latencia aún</text></svg>",
        encoding="utf-8",
    )


def save_outputs() -> tuple[Path, Path]:
    RESULTS_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = build_latency_timeseries()
    csv_path = RESULTS_VIZ_DIR / "latency_monitor.csv"
    svg_path = DOCS_VIZ_DIR / "latency_monitor.svg"
    summary_df.to_csv(csv_path, index=False)

    if summary_df.empty:
        _write_placeholder(svg_path)
        return csv_path, svg_path

    chart_df = summary_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chart_df["date"], chart_df["mean_latency_minutes"], marker="o", label="Latencia media")
    ax.plot(chart_df["date"], chart_df["p95_latency_minutes"], marker="o", label="Latencia p95")
    ax.set_title("Latencia intradía (minutos)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Minutos")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    return csv_path, svg_path


if __name__ == "__main__":
    save_outputs()
