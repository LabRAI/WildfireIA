from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(".")
EXP_DIR = BASE_DIR / "experiments" / "ia_failure" / "ablation" / "leave_one_source_out"
FULL_SUMMARY = BASE_DIR / "results" / "ia_failure_full_all_seeds_mean_std.csv"
RESULTS_DIR = BASE_DIR / "results"
RAW_OUT = RESULTS_DIR / "ia_failure_leave_one_source_out_5seeds_raw.csv"
SUMMARY_OUT = RESULTS_DIR / "ia_failure_leave_one_source_out_5seeds_summary.csv"
SUMMARY_MD_OUT = RESULTS_DIR / "ia_failure_leave_one_source_out_5seeds_summary.md"
DROP_TABLE_OUT = RESULTS_DIR / "ia_failure_leave_one_source_out_auprc_drop_paper_table.csv"
DROP_TABLE_MD_OUT = RESULTS_DIR / "ia_failure_leave_one_source_out_auprc_drop_paper_table.md"

PROTOCOL_LABELS = {
    "all_without_fire": "w/o FIRMS",
    "all_without_weather": "w/o Weather",
    "all_without_vegetation": "w/o Vegetation",
    "all_without_fuel": "w/o Fuel",
    "all_without_topography": "w/o Topography",
    "all_without_access": "w/o Access",
    "all_without_human": "w/o Population",
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
        "ablation_name": payload.get("ablation_name", "leave_one_source_out"),
        "protocol": protocol,
        "removed_source": PROTOCOL_LABELS.get(protocol, protocol),
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


def fmt_drop(mean: float, std: float) -> str:
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
        "removed_source",
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
        grouped = raw.groupby(["protocol", "removed_source", "representation", "model_name", "model_label"], dropna=False)
        count = grouped["seed"].nunique().rename("num_seeds_completed")
        means = grouped[METRIC_COLUMNS].mean(numeric_only=True).add_suffix("_mean")
        stds = grouped[METRIC_COLUMNS].std(numeric_only=True).add_suffix("_std")
        summary = pd.concat([count, means, stds], axis=1).reset_index()
        summary = summary.sort_values(["representation", "model_name", "protocol"])

    if FULL_SUMMARY.exists() and not summary.empty:
        full = pd.read_csv(FULL_SUMMARY)
        full = full[["representation", "model_name", "test_auprc_mean", "test_auprc_std"]].rename(
            columns={
                "test_auprc_mean": "full_test_auprc_mean",
                "test_auprc_std": "full_test_auprc_std",
            }
        )
        summary = summary.merge(full, on=["representation", "model_name"], how="left")
        summary["test_auprc_drop_mean"] = summary["full_test_auprc_mean"] - summary["test_auprc_mean"]
        summary["test_auprc_drop_std"] = (summary["full_test_auprc_std"].fillna(0) ** 2 + summary["test_auprc_std"].fillna(0) ** 2) ** 0.5

    summary.to_csv(SUMMARY_OUT, index=False)
    SUMMARY_MD_OUT.write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")

    if summary.empty or "test_auprc_drop_mean" not in summary.columns:
        paper = pd.DataFrame()
    else:
        keep_order = [
            ("tabular", "xgboost"),
            ("temporal", "transformer"),
            ("spatial", "swin_unet"),
            ("spatiotemporal", "swinlstm"),
        ]
        protocol_order = [
            "all_without_fire",
            "all_without_weather",
            "all_without_vegetation",
            "all_without_fuel",
            "all_without_topography",
            "all_without_access",
            "all_without_human",
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
                    row[PROTOCOL_LABELS[protocol]] = fmt_drop(r["test_auprc_drop_mean"], r["test_auprc_drop_std"])
            paper_rows.append(row)
        paper = pd.DataFrame(paper_rows)
    paper.to_csv(DROP_TABLE_OUT, index=False)
    DROP_TABLE_MD_OUT.write_text(paper.to_markdown(index=False) + "\n", encoding="utf-8")

    print(f"Read {len(raw)} metrics files from {EXP_DIR}")
    print(f"Wrote {RAW_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {SUMMARY_MD_OUT}")
    print(f"Wrote {DROP_TABLE_OUT}")
    print(f"Wrote {DROP_TABLE_MD_OUT}")


if __name__ == "__main__":
    main()
