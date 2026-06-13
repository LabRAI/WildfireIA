from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(".")
EXP_DIR = BASE_DIR / "experiments" / "ia_failure" / "full"
RESULTS_DIR = BASE_DIR / "results"
RAW_OUT = RESULTS_DIR / "ia_failure_full_all_seeds_raw.csv"
AGG_OUT = RESULTS_DIR / "ia_failure_full_all_seeds_mean_std.csv"
MD_OUT = RESULTS_DIR / "ia_failure_full_all_seeds_mean_std.md"


METRIC_COLUMNS = [
    "val_auprc",
    "test_auprc",
    "val_auroc",
    "test_auroc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_iou",
    "test_brier",
    "test_ece",
    "test_precision_at_1",
    "test_recall_at_1",
    "test_precision_at_5",
    "test_recall_at_5",
    "test_precision_at_10",
    "test_recall_at_10",
    "best_epoch",
    "runtime_seconds",
]


def read_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    row = {
        "task": payload.get("task", "ia_failure"),
        "experiment_type": payload.get("experiment_type", "full"),
        "representation": payload.get("representation", path.parents[2].name if len(path.parents) > 2 else None),
        "model_name": payload.get("model_name", path.parent.name.split("_seed")[0]),
        "seed": payload.get("seed"),
        "output_dir": payload.get("output_dir", str(path.parent)),
    }
    # Backfill from path: full/{representation}/weather5_all/{model}_seed{seed}/metrics.json
    try:
        row["representation"] = path.parents[2].name
    except Exception:
        pass
    for col in METRIC_COLUMNS:
        row[col] = payload.get(col)
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_paths = sorted(EXP_DIR.glob("*/*/*_seed*/metrics.json"))
    rows = [read_metrics(path) for path in metrics_paths]
    columns = [
        "task",
        "experiment_type",
        "representation",
        "model_name",
        "seed",
        *METRIC_COLUMNS,
        "output_dir",
    ]
    raw = pd.DataFrame(rows, columns=columns)
    raw.to_csv(RAW_OUT, index=False)

    if raw.empty:
        agg = pd.DataFrame(columns=["representation", "model_name", "num_seeds_completed"])
    else:
        numeric_metrics = [col for col in METRIC_COLUMNS if col in raw.columns]
        grouped = raw.groupby(["representation", "model_name"], dropna=False)
        count = grouped["seed"].nunique().rename("num_seeds_completed")
        means = grouped[numeric_metrics].mean(numeric_only=True).add_suffix("_mean")
        stds = grouped[numeric_metrics].std(numeric_only=True).add_suffix("_std")
        agg = pd.concat([count, means, stds], axis=1).reset_index()
        if "test_auprc_mean" in agg.columns:
            agg = agg.sort_values("test_auprc_mean", ascending=False, na_position="last")
    agg.to_csv(AGG_OUT, index=False)
    MD_OUT.write_text(agg.to_markdown(index=False) + "\n", encoding="utf-8")

    print(f"Read {len(raw)} metrics files from {EXP_DIR}")
    print(f"Wrote {RAW_OUT}")
    print(f"Wrote {AGG_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
