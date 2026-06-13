from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(".")
EXP_DIR = BASE_DIR / "experiments" / "ia_failure" / "ablation" / "fpafod_plus_source"
RESULTS_DIR = BASE_DIR / "results"
RAW_OUT = RESULTS_DIR / "ia_failure_fpafod_plus_source_ablation_5seeds_raw.csv"
SUMMARY_OUT = RESULTS_DIR / "ia_failure_fpafod_plus_source_ablation_5seeds_summary.csv"
SUMMARY_MD_OUT = RESULTS_DIR / "ia_failure_fpafod_plus_source_ablation_5seeds_summary.md"
PAPER_TABLE_OUT = RESULTS_DIR / "ia_failure_fpafod_plus_source_ablation_auprc_paper_table.csv"
PAPER_TABLE_MD_OUT = RESULTS_DIR / "ia_failure_fpafod_plus_source_ablation_auprc_paper_table.md"

PROTOCOL_LABELS = {
    "metadata_vegetation": "Vegetation",
    "metadata_fuel": "Fuel",
    "metadata_topography": "Topography",
    "metadata_access": "Access",
    "metadata_human": "Population",
}

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "transformer": "Transformer",
    "swin_unet": "Swin-UNet",
    "swinlstm": "SwinLSTM",
}

METRIC_COLUMNS = [
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
    "best_epoch",
    "runtime_seconds",
]


def protocol_from_weather_dir(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("weather5_"):
            return part.replace("weather5_", "", 1)
    return None


def read_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    protocol = payload.get("input_protocol") or protocol_from_weather_dir(path)
    model_name = payload.get("model_name", path.parent.name.split("_seed")[0])
    row = {
        "task": payload.get("task", "ia_failure"),
        "experiment_type": payload.get("experiment_type", "ablation"),
        "ablation_name": payload.get("ablation_name", "fpafod_plus_source"),
        "protocol": protocol,
        "source_label": PROTOCOL_LABELS.get(protocol, protocol),
        "representation": payload.get("representation"),
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "seed": payload.get("seed"),
        "output_dir": payload.get("output_dir", str(path.parent)),
    }
    try:
        row["representation"] = path.parents[2].name
    except Exception:
        pass
    for col in METRIC_COLUMNS:
        row[col] = payload.get(col)
    return row


def fmt_percent(mean: float, std: float) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean * 100:.1f}±{std * 100:.1f}"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_paths = sorted(EXP_DIR.glob("*/*/*_seed*/metrics.json"))
    rows = [read_metrics(path) for path in metrics_paths]
    raw_columns = [
        "task",
        "experiment_type",
        "ablation_name",
        "protocol",
        "source_label",
        "representation",
        "model_name",
        "model_label",
        "seed",
        *METRIC_COLUMNS,
        "output_dir",
    ]
    raw = pd.DataFrame(rows, columns=raw_columns)
    raw.to_csv(RAW_OUT, index=False)

    if raw.empty:
        summary = pd.DataFrame(columns=["protocol", "representation", "model_name", "num_seeds_completed"])
    else:
        grouped = raw.groupby(["protocol", "source_label", "representation", "model_name", "model_label"], dropna=False)
        count = grouped["seed"].nunique().rename("num_seeds_completed")
        means = grouped[METRIC_COLUMNS].mean(numeric_only=True).add_suffix("_mean")
        stds = grouped[METRIC_COLUMNS].std(numeric_only=True).add_suffix("_std")
        summary = pd.concat([count, means, stds], axis=1).reset_index()
        summary = summary.sort_values(["representation", "model_name", "protocol"])
    summary.to_csv(SUMMARY_OUT, index=False)
    SUMMARY_MD_OUT.write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")

    if summary.empty:
        paper = pd.DataFrame()
    else:
        keep_order = [
            ("tabular", "xgboost"),
            ("temporal", "transformer"),
            ("spatial", "swin_unet"),
            ("spatiotemporal", "swinlstm"),
        ]
        protocol_order = [
            "metadata_vegetation",
            "metadata_fuel",
            "metadata_topography",
            "metadata_access",
            "metadata_human",
        ]
        paper_rows = []
        for representation, model_name in keep_order:
            row = {"Selected model": MODEL_LABELS[model_name]}
            subset = summary[(summary["representation"] == representation) & (summary["model_name"] == model_name)]
            for protocol in protocol_order:
                protocol_row = subset[subset["protocol"] == protocol]
                if protocol_row.empty:
                    row[PROTOCOL_LABELS[protocol]] = ""
                else:
                    r = protocol_row.iloc[0]
                    row[PROTOCOL_LABELS[protocol]] = fmt_percent(r["test_auprc_mean"], r["test_auprc_std"])
            paper_rows.append(row)
        paper = pd.DataFrame(paper_rows)
    paper.to_csv(PAPER_TABLE_OUT, index=False)
    PAPER_TABLE_MD_OUT.write_text(paper.to_markdown(index=False) + "\n", encoding="utf-8")

    print(f"Read {len(raw)} metrics files from {EXP_DIR}")
    print(f"Wrote {RAW_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {SUMMARY_MD_OUT}")
    print(f"Wrote {PAPER_TABLE_OUT}")
    print(f"Wrote {PAPER_TABLE_MD_OUT}")


if __name__ == "__main__":
    main()
