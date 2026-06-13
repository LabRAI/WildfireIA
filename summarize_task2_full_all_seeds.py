#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(".")
ROOT = BASE_DIR / "experiments" / "containment_time" / "full"
RESULTS_DIR = BASE_DIR / "results"

METRICS = [
    "val_mae_hours",
    "test_mae_hours",
    "val_rmse_hours",
    "test_rmse_hours",
    "val_median_ae_hours",
    "test_median_ae_hours",
    "val_log_mae",
    "test_log_mae",
    "val_log_rmse",
    "test_log_rmse",
    "val_r2",
    "test_r2",
    "val_spearman",
    "test_spearman",
    "val_pearson",
    "test_pearson",
]


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(ROOT.glob("*/*/*_seed*/metrics.json")):
        m = read_json(metrics_path)
        row = {
            "task": m.get("task"),
            "experiment_type": m.get("experiment_type"),
            "representation": m.get("representation"),
            "model_name": m.get("model_name"),
            "seed": m.get("seed"),
            "best_epoch": m.get("best_epoch"),
            "runtime_seconds": m.get("runtime_seconds"),
            "output_dir": str(metrics_path.parent),
        }
        for metric in METRICS:
            row[metric] = m.get(metric)
        rows.append(row)

    raw = pd.DataFrame(rows)
    raw_path = RESULTS_DIR / "containment_time_full_all_seeds_raw.csv"
    raw.to_csv(raw_path, index=False)

    if raw.empty:
        summary = pd.DataFrame()
    else:
        grouped = raw.groupby(["representation", "model_name"], dropna=False)
        parts = []
        for (representation, model_name), g in grouped:
            row = {
                "representation": representation,
                "model_name": model_name,
                "num_seeds_completed": int(g["seed"].nunique()),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = g[metric].mean()
                row[f"{metric}_std"] = g[metric].std()
            row["best_epoch_mean"] = g["best_epoch"].mean()
            row["runtime_seconds_mean"] = g["runtime_seconds"].mean()
            parts.append(row)
        summary = pd.DataFrame(parts)
        if "test_mae_hours_mean" in summary.columns:
            summary = summary.sort_values("test_mae_hours_mean", ascending=True)

    summary_path = RESULTS_DIR / "containment_time_full_all_seeds_mean_std.csv"
    md_path = RESULTS_DIR / "containment_time_full_all_seeds_mean_std.md"
    summary.to_csv(summary_path, index=False)
    summary.to_markdown(md_path, index=False)

    print(f"Read {len(raw)} metrics files from {ROOT}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
