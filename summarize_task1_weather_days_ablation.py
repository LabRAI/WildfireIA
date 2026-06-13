#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(".")
ROOT = BASE_DIR / "experiments" / "ia_failure" / "ablation" / "weather_days"
RESULTS_DIR = BASE_DIR / "results"

METRICS = [
    "val_auprc",
    "test_auprc",
    "val_auroc",
    "test_auroc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_brier",
    "test_ece",
    "test_precision_at_5",
    "test_recall_at_5",
]


MODEL_LABELS = {
    "xgboost": "XGBoost",
    "transformer": "Transformer",
    "swin_unet": "Swin-UNet",
    "swinlstm": "SwinLSTM",
}


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def fmt(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}±{std * 100:.1f}"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(ROOT.glob("*/*/*_seed*/metrics.json")):
        m = read_json(metrics_path)
        row = {
            "task": m.get("task"),
            "experiment_type": m.get("experiment_type"),
            "ablation_name": "weather_days",
            "representation": m.get("representation"),
            "model_name": m.get("model_name"),
            "model_label": MODEL_LABELS.get(m.get("model_name"), m.get("model_name")),
            "weather_days": m.get("weather_days"),
            "input_protocol": m.get("input_protocol"),
            "seed": m.get("seed"),
            "best_epoch": m.get("best_epoch"),
            "runtime_seconds": m.get("runtime_seconds"),
            "output_dir": str(metrics_path.parent),
        }
        for metric in METRICS:
            row[metric] = m.get(metric)
        rows.append(row)

    raw = pd.DataFrame(rows)
    raw_path = RESULTS_DIR / "ia_failure_weather_days_ablation_5seeds_raw.csv"
    raw.to_csv(raw_path, index=False)

    if raw.empty:
        summary = pd.DataFrame()
        table = pd.DataFrame()
    else:
        grouped = raw.groupby(["weather_days", "representation", "model_name", "model_label"], dropna=False)
        parts = []
        for keys, g in grouped:
            weather_days, representation, model_name, model_label = keys
            row = {
                "weather_days": int(weather_days),
                "representation": representation,
                "model_name": model_name,
                "model_label": model_label,
                "num_seeds_completed": int(g["seed"].nunique()),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = g[metric].mean()
                row[f"{metric}_std"] = g[metric].std()
            row["best_epoch_mean"] = g["best_epoch"].mean()
            row["runtime_seconds_mean"] = g["runtime_seconds"].mean()
            parts.append(row)
        summary = pd.DataFrame(parts).sort_values(["representation", "model_name", "weather_days"])

        models = ["XGBoost", "Transformer", "Swin-UNet", "SwinLSTM"]
        table_rows = []
        for model in models:
            row = {"Selected model": model}
            for day in [1, 2, 3, 4, 5]:
                r = summary[(summary["model_label"] == model) & (summary["weather_days"] == day)]
                if r.empty:
                    row[f"{day} day"] = "TBD"
                else:
                    rr = r.iloc[0]
                    row[f"{day} day"] = fmt(rr["test_auprc_mean"], rr["test_auprc_std"])
            table_rows.append(row)
        table = pd.DataFrame(table_rows)

    summary_path = RESULTS_DIR / "ia_failure_weather_days_ablation_5seeds_summary.csv"
    md_path = RESULTS_DIR / "ia_failure_weather_days_ablation_5seeds_summary.md"
    table_csv = RESULTS_DIR / "ia_failure_weather_days_ablation_auprc_paper_table.csv"
    table_md = RESULTS_DIR / "ia_failure_weather_days_ablation_auprc_paper_table.md"
    summary.to_csv(summary_path, index=False)
    summary.to_markdown(md_path, index=False)
    table.to_csv(table_csv, index=False)
    table.to_markdown(table_md, index=False)

    print(f"Read {len(raw)} metrics files from {ROOT}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {table_csv}")
    print(f"Wrote {table_md}")


if __name__ == "__main__":
    main()
