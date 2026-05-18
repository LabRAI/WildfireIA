from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(".")
PATCH_SIZE = 29
CELLS_PER_PATCH = PATCH_SIZE * PATCH_SIZE
WEATHER_DAY_MAP = {
    1: [0],
    2: [-1, 0],
    3: [-2, -1, 0],
    4: [-3, -2, -1, 0],
    5: [-4, -3, -2, -1, 0],
}

TASKS = {
    "ia_failure": {
        "target_column": "ia_failure_label",
        "task_type": "binary_classification",
        "raw_column": None,
    },
    "containment_time": {
        "target_column": "log_containment_hours",
        "task_type": "regression",
        "raw_column": "containment_hours",
    },
}

PROTOCOL_GROUPS = {
    "metadata": ["metadata"],
    "firms": ["fire_strict"],
    "fire": ["fire_strict"],
    "fire_wide": ["fire_strict", "fire_wide"],
    "weather": ["weather_aggregate", "fire_danger_aggregate"],
    "fuel": ["fuel"],
    "vegetation": ["vegetation"],
    "topography": ["topography"],
    "access": ["access"],
    "human": ["human"],
    "metadata_vegetation": ["metadata", "vegetation"],
    "metadata_fuel": ["metadata", "fuel"],
    "metadata_topography": ["metadata", "topography"],
    "metadata_access": ["metadata", "access"],
    "metadata_human": ["metadata", "human"],
    "all": [
        "metadata",
        "fire_strict",
        "weather_aggregate",
        "fire_danger_aggregate",
        "fuel",
        "vegetation",
        "topography",
        "access",
        "human",
    ],
    "all_without_fire": [
        "metadata",
        "weather_aggregate",
        "fire_danger_aggregate",
        "fuel",
        "vegetation",
        "topography",
        "access",
        "human",
    ],
    "all_without_weather": [
        "metadata",
        "fire_strict",
        "fuel",
        "vegetation",
        "topography",
        "access",
        "human",
    ],
    "all_without_vegetation": [
        "metadata",
        "fire_strict",
        "weather_aggregate",
        "fire_danger_aggregate",
        "fuel",
        "topography",
        "access",
        "human",
    ],
    "all_without_fuel": [
        "metadata",
        "fire_strict",
        "weather_aggregate",
        "fire_danger_aggregate",
        "vegetation",
        "topography",
        "access",
        "human",
    ],
    "all_without_topography": [
        "metadata",
        "fire_strict",
        "weather_aggregate",
        "fire_danger_aggregate",
        "fuel",
        "vegetation",
        "access",
        "human",
    ],
    "all_without_access": [
        "metadata",
        "fire_strict",
        "weather_aggregate",
        "fire_danger_aggregate",
        "fuel",
        "vegetation",
        "topography",
        "human",
    ],
    "all_without_human": [
        "metadata",
        "fire_strict",
        "weather_aggregate",
        "fire_danger_aggregate",
        "fuel",
        "vegetation",
        "topography",
        "access",
    ],
}

TEMPORAL_WEATHER_CHANNELS = [
    "tmmx",
    "tmmn",
    "pr",
    "rmax",
    "rmin",
    "sph",
    "vpd",
    "vs",
    "srad",
    "erc",
    "bi",
    "fm100",
    "fm1000",
    "wind_dir_sin",
    "wind_dir_cos",
]

FUEL_PATCH_CHANNELS = ["fbfm40", "cbd", "cbh", "cc", "ch", "fd", "fvt", "fvc", "fvh"]
VEGETATION_PATCH_CHANNELS = ["evt", "evc", "evh"]
STATIC_PATCH_CHANNELS = [*FUEL_PATCH_CHANNELS, *VEGETATION_PATCH_CHANNELS, "elev", "slope", "aspect_sin", "aspect_cos", "pop_density"]
VIIRS_PATCH_CHANNELS = [
    "viirs_cell_count_D",
    "viirs_cell_sum_frp_D",
    "viirs_cell_max_frp_D",
    "viirs_cell_mean_frp_D",
    "viirs_cell_max_bright_ti4_D",
    "viirs_cell_mean_bright_ti4_D",
    "viirs_cell_day_count_D",
    "viirs_cell_night_count_D",
    "viirs_cell_has_detection_D",
]
OSM_PATCH_CHANNELS = [
    "cell_distance_to_nearest_drivable_road_m",
    "cell_distance_to_nearest_fire_station_m",
    "cell_road_length_375m_m",
    "cell_has_drivable_road",
    "cell_distance_to_nearest_major_road_m",
    "cell_distance_to_nearest_track_or_service_road_m",
]
TOPO_PATCH_CHANNELS = ["elev", "slope", "aspect_sin", "aspect_cos"]
PATCH_METADATA_CHANNELS = ["lat", "lon", "discovery_month", "discovery_doy", "discovery_hour"]

KNOWN_CATEGORICAL_COLUMNS = {
    "state",
    "county",
    "cause_classification",
    "general_cause",
    "fbfm40_point",
    "fd_point",
    "fvt_point",
    "fvc_point",
    "fvh_point",
    "evt_point",
    "evc_point",
    "evh_point",
    "fbfm40_mode_1km",
    "fbfm40_mode_3km",
    "fbfm40_mode_5km",
    "fd_mode_1km",
    "fd_mode_3km",
    "fd_mode_5km",
    "fvt_mode_1km",
    "fvt_mode_3km",
    "fvt_mode_5km",
    "fvc_mode_1km",
    "fvc_mode_3km",
    "fvc_mode_5km",
    "fvh_mode_1km",
    "fvh_mode_3km",
    "fvh_mode_5km",
    "evt_mode_1km",
    "evt_mode_3km",
    "evt_mode_5km",
    "evc_mode_1km",
    "evc_mode_3km",
    "evc_mode_5km",
    "evh_mode_1km",
    "evh_mode_3km",
    "evh_mode_5km",
}

CORE_INDEX_COLUMNS = {"fire_id", "year", "split"}


def protocol_groups(input_protocol: str) -> list[str]:
    if input_protocol not in PROTOCOL_GROUPS:
        raise KeyError(f"Unknown input_protocol: {input_protocol}")
    return PROTOCOL_GROUPS[input_protocol]


def protocol_includes_weather(input_protocol: str) -> bool:
    groups = protocol_groups(input_protocol)
    return "weather_aggregate" in groups or "fire_danger_aggregate" in groups


def ensure_project_path(path: Path) -> Path:
    resolved = path.resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path must stay inside {PROJECT_ROOT}: {resolved}")
    return resolved


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def remove_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_paths(canonical_dir: Path) -> dict[str, Path]:
    return {
        "master": canonical_dir / "master_features_natural_2016_2020.parquet",
        "weather_daily_event": canonical_dir / "gridmet_daily_event_features_natural_2016_2020.parquet",
        "grid_index": canonical_dir / "event_grid_375m_index_natural_2016_2020.parquet",
        "static_patch": canonical_dir / "event_static_patch_375m_natural_2016_2020.parquet",
        "viirs_patch": canonical_dir / "event_viirs_patch_375m_D_natural_2016_2020.parquet",
        "weather_daily_patch": canonical_dir / "event_weather_daily_patch_375m_natural_2016_2020.parquet",
        "weather_aggregate_patch": canonical_dir / "event_weather_aggregate_patch_375m_natural_2016_2020.parquet",
        "osm_patch": canonical_dir / "event_osm_patch_375m_natural_2016_2020.parquet",
        "feature_manifest": canonical_dir / "feature_manifest_natural.json",
        "label_manifest": canonical_dir / "label_manifest_natural.json",
        "patch_manifest": canonical_dir / "event_patch_manifest_375m_natural.json",
        "temporal_manifest": canonical_dir / "temporal_protocol_manifest_natural.json",
    }


def required_path_keys() -> list[str]:
    return [
        "master",
        "weather_daily_event",
        "grid_index",
        "static_patch",
        "viirs_patch",
        "weather_daily_patch",
        "weather_aggregate_patch",
        "osm_patch",
        "feature_manifest",
        "label_manifest",
        "patch_manifest",
        "temporal_manifest",
    ]


def load_manifests(paths: dict[str, Path]) -> tuple[dict, dict, dict, dict]:
    return (
        load_json(paths["feature_manifest"]),
        load_json(paths["label_manifest"]),
        load_json(paths["patch_manifest"]),
        load_json(paths["temporal_manifest"]),
    )


def output_subdir(output_dir: Path, task: str, representation: str, weather_days: int, input_protocol: str) -> Path:
    return output_dir / task / representation / f"weather{weather_days}_{input_protocol}"


def target_info(task: str) -> dict:
    if task not in TASKS:
        raise KeyError(f"Unknown task: {task}")
    return TASKS[task]


def filter_task_samples(master: pd.DataFrame, task: str) -> pd.DataFrame:
    info = target_info(task)
    target_col = info["target_column"]
    if target_col not in master.columns:
        raise KeyError(f"Target column not found in master_features: {target_col}")
    filtered = master.loc[master[target_col].notna()].copy()
    filtered["target"] = filtered[target_col]
    if task == "ia_failure":
        filtered["target"] = filtered["target"].astype(int)
    return filtered


def apply_input_protocol_sample_filter(samples: pd.DataFrame, input_protocol: str) -> pd.DataFrame:
    """Apply protocol-level sample restrictions without changing task labels."""
    if input_protocol != "firms":
        return samples
    match_col = "viirs_num_assigned_detections_D"
    fallback_col = "has_viirs_detection_1km_D"
    filtered = samples.copy()
    if match_col in filtered.columns:
        mask = pd.to_numeric(filtered[match_col], errors="coerce").fillna(0) > 0
    elif fallback_col in filtered.columns:
        mask = pd.to_numeric(filtered[fallback_col], errors="coerce").fillna(0) > 0
    else:
        raise KeyError(
            "input_protocol=firms requires viirs_num_assigned_detections_D "
            "or has_viirs_detection_1km_D in master_features."
        )
    before = len(filtered)
    filtered = filtered.loc[mask].copy()
    print(f"FIRMS-only sample filter retained {len(filtered)} / {before} task samples.")
    return filtered


def normalize_master_metadata(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    if "year" not in master.columns:
        for candidate in ["year_x", "year_y", "FIRE_YEAR"]:
            if candidate in master.columns:
                master["year"] = master[candidate]
                break
    if "split" not in master.columns and "split_x" in master.columns:
        master["split"] = master["split_x"]
    if "year" in master.columns:
        master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("Int64")
    return master


def build_sample_index(samples: pd.DataFrame, master: pd.DataFrame, task: str) -> pd.DataFrame:
    samples = normalize_master_metadata(samples)
    master = normalize_master_metadata(master)
    info = target_info(task)
    target_col = info["target_column"]
    raw_col = info["raw_column"]
    sample_index = samples.copy()
    recover_cols = ["fire_id", "year", "split", target_col]
    if raw_col:
        recover_cols.append(raw_col)
    missing = [col for col in recover_cols if col not in sample_index.columns]
    if missing:
        recovered = master[[col for col in recover_cols if col in master.columns]].copy()
        recovered["fire_id"] = recovered["fire_id"].astype(str)
        sample_index["fire_id"] = sample_index["fire_id"].astype(str)
        sample_index = sample_index.merge(recovered, on="fire_id", how="left", suffixes=("", "_from_master"))
        for col in missing:
            from_master = f"{col}_from_master"
            if from_master in sample_index.columns:
                sample_index[col] = sample_index[from_master]

    if target_col not in sample_index.columns and "target" in sample_index.columns:
        sample_index[target_col] = sample_index["target"]
    required = ["fire_id", "year", "split", target_col]
    if raw_col:
        required.append(raw_col)
    still_missing = [col for col in required if col not in sample_index.columns]
    if still_missing:
        raise KeyError(f"Sample index is missing required metadata columns: {still_missing}")
    sample_index = sample_index[required].copy()
    sample_index["fire_id"] = sample_index["fire_id"].astype(str)
    return sample_index


def split_frames(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": samples.loc[samples["split"] == "train"].copy(),
        "val": samples.loc[samples["split"] == "val"].copy(),
        "test": samples.loc[samples["split"] == "test"].copy(),
    }


def split_years(split: str) -> list[int]:
    if split == "train":
        return [2016, 2017, 2018]
    if split == "val":
        return [2019]
    if split == "test":
        return [2020]
    raise ValueError(split)


def weather_feature_allowed(col: str, weather_days: int) -> bool:
    suffix_windows = {
        "lag1": 2,
        "mean2": 2,
        "sum2": 2,
        "mean3": 3,
        "sum3": 3,
        "mean4": 4,
        "sum4": 4,
        "mean5": 5,
        "sum5": 5,
    }
    if col.endswith("_day0"):
        return True
    for suffix, min_days in suffix_windows.items():
        if col.endswith(f"_{suffix}"):
            return weather_days >= min_days
    return True


def requested_feature_columns(
    feature_manifest: dict,
    input_protocol: str,
    weather_days: int,
    master_columns: Iterable[str],
    target_col: str,
) -> tuple[list[str], list[str]]:
    forbidden = set(feature_manifest.get("forbidden_as_features", []))
    forbidden.update(
        [
            target_col,
            "fire_size_acres",
            "fire_size_ha",
            "ia_failure_label",
            "contain_dt",
            "containment_hours",
            "log_containment_hours",
            "log_fire_size_ha",
        ]
    )
    master_columns = set(master_columns)
    requested = []
    for group in protocol_groups(input_protocol):
        requested.extend(feature_manifest.get(group, []))
    requested = list(dict.fromkeys(requested))
    removed = [col for col in requested if col in forbidden]
    selected = []
    missing = []
    for col in requested:
        if col in forbidden:
            continue
        if col not in master_columns:
            missing.append(col)
            continue
        if col in feature_manifest.get("weather_aggregate", []) or col in feature_manifest.get("fire_danger_aggregate", []):
            if not weather_feature_allowed(col, weather_days):
                removed.append(col)
                continue
        selected.append(col)
    return selected, sorted(set(removed + missing))


def static_protocol_columns(
    feature_manifest: dict,
    input_protocol: str,
    weather_days: int,
    master_columns: Iterable[str],
    target_col: str,
) -> tuple[list[str], list[str]]:
    if input_protocol == "weather":
        return [], []
    if input_protocol == "all" or input_protocol.startswith("all_without_"):
        groups = [
            group
            for group in protocol_groups(input_protocol)
            if group not in {"weather_aggregate", "fire_danger_aggregate"}
        ]
    elif input_protocol == "fire_wide":
        groups = ["fire_strict", "fire_wide"]
    else:
        groups = protocol_groups(input_protocol)
    pseudo_manifest = dict(feature_manifest)
    selected_requested = []
    for group in groups:
        selected_requested.extend(feature_manifest.get(group, []))
    pseudo_manifest["_static"] = selected_requested
    old = PROTOCOL_GROUPS.get("_static")
    PROTOCOL_GROUPS["_static"] = ["_static"]
    try:
        cols, removed = requested_feature_columns(pseudo_manifest, "_static", weather_days, master_columns, target_col)
    finally:
        if old is None:
            PROTOCOL_GROUPS.pop("_static", None)
        else:
            PROTOCOL_GROUPS["_static"] = old
    return cols, removed


def infer_categorical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    categorical = []
    for col in columns:
        if col in KNOWN_CATEGORICAL_COLUMNS:
            categorical.append(col)
        elif (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ):
            categorical.append(col)
    return categorical


def fit_transform_event_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    standardize: bool,
) -> tuple[dict[str, np.ndarray], list[str], dict]:
    if not feature_cols:
        arrays = {
            "train": np.zeros((len(train_df), 0), dtype=np.float32),
            "val": np.zeros((len(val_df), 0), dtype=np.float32),
            "test": np.zeros((len(test_df), 0), dtype=np.float32),
        }
        return arrays, [], {
            "categorical_columns": [],
            "numeric_columns": [],
            "imputer_policy": "none",
            "scaler_policy": "none",
        }

    categorical_cols = infer_categorical_columns(train_df, feature_cols)
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    numeric_train = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=train_df.index)
    numeric_val = val_df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=val_df.index)
    numeric_test = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=test_df.index)

    medians = numeric_train.median()
    medians = medians.fillna(0.0)
    numeric_train = numeric_train.fillna(medians)
    numeric_val = numeric_val.fillna(medians)
    numeric_test = numeric_test.fillna(medians)

    scaler_metadata = {"standardize": bool(standardize), "mean": {}, "scale": {}}
    if standardize and numeric_cols:
        scaler = StandardScaler()
        train_numeric_values = scaler.fit_transform(numeric_train.to_numpy(dtype=np.float64))
        val_numeric_values = scaler.transform(numeric_val.to_numpy(dtype=np.float64))
        test_numeric_values = scaler.transform(numeric_test.to_numpy(dtype=np.float64))
        scaler_metadata["mean"] = dict(zip(numeric_cols, scaler.mean_.tolist()))
        scaler_metadata["scale"] = dict(zip(numeric_cols, scaler.scale_.tolist()))
    else:
        train_numeric_values = numeric_train.to_numpy(dtype=np.float64)
        val_numeric_values = numeric_val.to_numpy(dtype=np.float64)
        test_numeric_values = numeric_test.to_numpy(dtype=np.float64)

    if categorical_cols:
        train_cat = pd.get_dummies(train_df[categorical_cols].fillna("Unknown").astype(str), columns=categorical_cols)
        val_cat = pd.get_dummies(val_df[categorical_cols].fillna("Unknown").astype(str), columns=categorical_cols)
        test_cat = pd.get_dummies(test_df[categorical_cols].fillna("Unknown").astype(str), columns=categorical_cols)
        cat_cols = list(train_cat.columns)
        val_cat = val_cat.reindex(columns=cat_cols, fill_value=0)
        test_cat = test_cat.reindex(columns=cat_cols, fill_value=0)
        train_values = np.concatenate([train_numeric_values, train_cat.to_numpy(dtype=np.float64)], axis=1)
        val_values = np.concatenate([val_numeric_values, val_cat.to_numpy(dtype=np.float64)], axis=1)
        test_values = np.concatenate([test_numeric_values, test_cat.to_numpy(dtype=np.float64)], axis=1)
    else:
        cat_cols = []
        train_values = train_numeric_values
        val_values = val_numeric_values
        test_values = test_numeric_values

    feature_names = numeric_cols + cat_cols
    arrays = {
        "train": train_values.astype(np.float32),
        "val": val_values.astype(np.float32),
        "test": test_values.astype(np.float32),
    }
    metadata = {
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "categorical_encoder": "pandas.get_dummies_fit_on_train_align_val_test",
        "categorical_dummy_columns": cat_cols,
        "imputer_policy": "numeric train median, categorical Unknown",
        "numeric_medians": medians.to_dict(),
        "scaler_policy": "StandardScaler fit on train numeric columns" if standardize and numeric_cols else "not_standardized",
        "scaler": scaler_metadata,
    }
    return arrays, feature_names, metadata


def write_sample_index_outputs(
    out_dir: Path,
    split: str,
    n_rows: int,
    y: np.ndarray,
    sample_index: pd.DataFrame,
    task: str,
) -> None:
    target_col = target_info(task)["target_column"]
    raw_col = target_info(task)["raw_column"]
    expected_cols = ["fire_id", "year", "split", target_col]
    if raw_col:
        expected_cols.append(raw_col)
    missing = [col for col in expected_cols if col not in sample_index.columns]
    if missing:
        raise KeyError(f"sample_index for {split} missing columns: {missing}")
    if n_rows != len(y):
        raise ValueError(f"{split}: n_rows={n_rows} does not match len(y)={len(y)}")
    if n_rows != len(sample_index):
        raise ValueError(f"{split}: n_rows={n_rows} does not match len(sample_index)={len(sample_index)}")
    target_values = sample_index[target_col].to_numpy()
    if not np.allclose(y.astype(float), target_values.astype(float), equal_nan=True):
        raise ValueError(f"{split}: y array order/values do not match sample_index {target_col}")
    fire_ids = sample_index["fire_id"].astype(str).to_numpy()
    if len(fire_ids) != n_rows:
        raise ValueError(f"{split}: fire_id order length does not match row count")

    np.save(out_dir / f"y_{split}.npy", y)
    np.save(out_dir / f"fire_id_{split}.npy", fire_ids)
    saved_fire_ids = np.load(out_dir / f"fire_id_{split}.npy", allow_pickle=True).astype(str)
    if not np.array_equal(saved_fire_ids, fire_ids):
        raise ValueError(f"{split}: saved fire_id array does not match sample_index order")
    if raw_col:
        np.save(out_dir / f"{raw_col}_{split}.npy", sample_index[raw_col].to_numpy(dtype=np.float32))
    sample_index[expected_cols].to_parquet(out_dir / f"sample_index_{split}.parquet", index=False)


def write_split_outputs(
    out_dir: Path,
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_index: pd.DataFrame,
    task: str,
) -> None:
    write_sample_index_outputs(out_dir, split, len(X), y, sample_index, task)
    np.save(out_dir / f"X_{split}.npy", X)


def write_seq_split_outputs(
    out_dir: Path,
    split: str,
    X_seq: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    sample_index: pd.DataFrame,
    task: str,
) -> None:
    if len(X_seq) != len(y):
        raise ValueError(f"{split}: len(X_seq)={len(X_seq)} does not match len(y)={len(y)}")
    if len(X_static) != len(y):
        raise ValueError(f"{split}: len(X_static)={len(X_static)} does not match len(y)={len(y)}")
    write_sample_index_outputs(out_dir, split, len(X_seq), y, sample_index, task)
    np.save(out_dir / f"X_seq_{split}.npy", X_seq)
    np.save(out_dir / f"X_static_{split}.npy", X_static)


def save_metadata(out_dir: Path, metadata: dict) -> None:
    write_json(out_dir / "metadata.json", metadata)


def build_tabular_cache(args, paths: dict[str, Path], feature_manifest: dict) -> Path:
    out_dir = output_subdir(args.output_dir, args.task, "tabular", args.weather_days, args.input_protocol)
    if (out_dir / "X_train.npy").exists() and not args.overwrite:
        print(f"Reusing existing tabular cache: {out_dir}")
        return out_dir
    remove_output_dir(out_dir, args.overwrite)

    master = normalize_master_metadata(read_parquet_robust(paths["master"]))
    samples = apply_input_protocol_sample_filter(filter_task_samples(master, args.task), args.input_protocol)
    splits = split_frames(samples)
    sample_index_splits = split_frames(build_sample_index(samples, master, args.task))
    target_col = target_info(args.task)["target_column"]
    feature_cols, removed = requested_feature_columns(
        feature_manifest, args.input_protocol, args.weather_days, master.columns, target_col
    )
    arrays, feature_names, preprocessing = fit_transform_event_features(
        splits["train"], splits["val"], splits["test"], feature_cols, args.standardize
    )
    for split, frame in splits.items():
        y = frame["target"].to_numpy(dtype=np.float32)
        write_split_outputs(out_dir, split, arrays[split], y, sample_index_splits[split], args.task)
    write_json(out_dir / "feature_names.json", {"feature_names": feature_names})
    save_metadata(
        out_dir,
        {
            "task": args.task,
            "representation": "tabular",
            "input_protocol": args.input_protocol,
            "weather_days": args.weather_days,
            "feature_names": feature_names,
            "source_feature_columns": feature_cols,
            "target_column": target_col,
            "split_counts": {split: int(len(frame)) for split, frame in splits.items()},
            "forbidden_columns_removed": removed,
            "missing_value_policy": preprocessing["imputer_policy"],
            "imputer_policy": preprocessing["imputer_policy"],
            "scaler_policy": preprocessing["scaler_policy"],
            "preprocessing": preprocessing,
            "created_at": created_at(),
        },
    )
    print(f"Wrote tabular cache: {out_dir}")
    return out_dir


def train_weather_medians(weather_daily: pd.DataFrame, train_fire_ids: np.ndarray, rel_days: list[int], channels: list[str]) -> pd.Series:
    train_weather = weather_daily.loc[
        weather_daily["fire_id"].astype(str).isin(set(train_fire_ids.astype(str)))
        & weather_daily["relative_day"].isin(rel_days)
    ]
    medians = train_weather[channels].apply(pd.to_numeric, errors="coerce").median()
    return medians.fillna(0.0)


def build_sequence_array(
    weather_daily: pd.DataFrame,
    sample_df: pd.DataFrame,
    rel_days: list[int],
    channels: list[str],
    medians: pd.Series,
    scaler: StandardScaler | None,
) -> np.ndarray:
    n = len(sample_df)
    t = len(rel_days)
    if not channels:
        return np.zeros((n, t, 0), dtype=np.float32)
    fire_order = sample_df[["fire_id"]].copy()
    fire_order["fire_order"] = np.arange(n)
    base = pd.MultiIndex.from_product(
        [sample_df["fire_id"].astype(str).to_list(), rel_days],
        names=["fire_id", "relative_day"],
    ).to_frame(index=False)
    base = base.merge(fire_order, on="fire_id", how="left")
    weather = weather_daily[["fire_id", "relative_day"] + channels].copy()
    weather["fire_id"] = weather["fire_id"].astype(str)
    merged = base.merge(weather, on=["fire_id", "relative_day"], how="left")
    values = merged[channels].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(dtype=np.float64)
    if scaler is not None and channels:
        values = scaler.transform(values)
    merged_values = values.reshape(n, t, len(channels))
    return merged_values.astype(np.float32)


def build_temporal_cache(args, paths: dict[str, Path], feature_manifest: dict) -> Path:
    out_dir = output_subdir(args.output_dir, args.task, "temporal", args.weather_days, args.input_protocol)
    if (out_dir / "X_seq_train.npy").exists() and not args.overwrite:
        print(f"Reusing existing temporal cache: {out_dir}")
        return out_dir
    remove_output_dir(out_dir, args.overwrite)

    master = normalize_master_metadata(read_parquet_robust(paths["master"]))
    weather_daily = read_parquet_robust(paths["weather_daily_event"])
    samples = apply_input_protocol_sample_filter(filter_task_samples(master, args.task), args.input_protocol)
    splits = split_frames(samples)
    sample_index_splits = split_frames(build_sample_index(samples, master, args.task))
    rel_days = WEATHER_DAY_MAP[args.weather_days]
    target_col = target_info(args.task)["target_column"]

    seq_channels = TEMPORAL_WEATHER_CHANNELS if protocol_includes_weather(args.input_protocol) else []
    seq_channels = [col for col in seq_channels if col in weather_daily.columns]
    medians = train_weather_medians(weather_daily, splits["train"]["fire_id"].astype(str).to_numpy(), rel_days, seq_channels) if seq_channels else pd.Series(dtype=float)
    seq_scaler = None
    if args.standardize and seq_channels:
        train_long = weather_daily.loc[
            weather_daily["fire_id"].astype(str).isin(set(splits["train"]["fire_id"].astype(str)))
            & weather_daily["relative_day"].isin(rel_days),
            seq_channels,
        ].apply(pd.to_numeric, errors="coerce").fillna(medians)
        seq_scaler = StandardScaler().fit(train_long.to_numpy(dtype=np.float64))

    static_cols, removed = static_protocol_columns(
        feature_manifest, args.input_protocol, args.weather_days, master.columns, target_col
    )
    static_arrays, static_feature_names, static_preprocessing = fit_transform_event_features(
        splits["train"], splits["val"], splits["test"], static_cols, args.standardize
    )

    for split, frame in splits.items():
        X_seq = build_sequence_array(weather_daily, frame, rel_days, seq_channels, medians, seq_scaler)
        X_static = static_arrays[split]
        y = frame["target"].to_numpy(dtype=np.float32)
        write_seq_split_outputs(out_dir, split, X_seq, X_static, y, sample_index_splits[split], args.task)
    write_json(out_dir / "temporal_feature_names.json", {"feature_names": seq_channels})
    write_json(out_dir / "static_feature_names.json", {"feature_names": static_feature_names})
    save_metadata(
        out_dir,
        {
            "task": args.task,
            "representation": "temporal",
            "input_protocol": args.input_protocol,
            "weather_days": args.weather_days,
            "relative_days": rel_days,
            "temporal_feature_names": seq_channels,
            "static_feature_names": static_feature_names,
            "target_column": target_col,
            "split_counts": {split: int(len(frame)) for split, frame in splits.items()},
            "forbidden_columns_removed": removed,
            "missing_value_policy": "weather numeric train median; static uses tabular policy",
            "scaler_policy": "StandardScaler fit on train only" if args.standardize else "not_standardized",
            "static_preprocessing": static_preprocessing,
            "created_at": created_at(),
        },
    )
    print(f"Wrote temporal cache: {out_dir}")
    return out_dir


def parquet_files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.parquet") if p.is_file())


def _file_parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def _read_parquet_file_existing_columns(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    available = _file_parquet_columns(path)
    keep = [col for col in columns if col in available]
    if not keep:
        return pd.DataFrame(index=range(0))
    return pd.read_parquet(path, columns=keep)


def read_parquet_robust(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet files without asking pyarrow to merge hive partitions."""
    path = Path(path)
    if path.is_file():
        return _read_parquet_file_existing_columns(path, columns=columns)
    files = parquet_files_under(path)
    if not files:
        return pd.DataFrame()
    frames = [_read_parquet_file_existing_columns(part, columns=columns) for part in files]
    return pd.concat(frames, ignore_index=True)


def read_parquet_parts_sample(path: Path, columns: list[str] | None = None, max_parts: int = 2) -> pd.DataFrame:
    """Read a few physical parquet part files for cheap validation."""
    path = Path(path)
    files = parquet_files_under(path)[:max_parts]
    if not files:
        return pd.DataFrame()
    frames = [_read_parquet_file_existing_columns(part, columns=columns) for part in files]
    return pd.concat(frames, ignore_index=True)


def read_parquet_files(files: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    return pd.concat([_read_parquet_file_existing_columns(file, columns=columns) for file in files], ignore_index=True)


def read_patch_table(path: Path, years: list[int], columns: list[str] | None = None, relative_days: list[int] | None = None) -> pd.DataFrame:
    files = []
    if path.is_file():
        frame = read_parquet_robust(path, columns=columns)
        if "year" in frame.columns:
            frame = frame.loc[frame["year"].isin(years)]
        if relative_days is not None and "relative_day" in frame.columns:
            frame = frame.loc[frame["relative_day"].isin(relative_days)]
        return frame
    for year in years:
        year_dir = path / f"year={year}"
        if not year_dir.exists():
            continue
        if relative_days is None:
            files.extend(parquet_files_under(year_dir))
        else:
            for rel in relative_days:
                rel_dir = year_dir / f"relative_day={rel}"
                files.extend(parquet_files_under(rel_dir))
    return read_parquet_files(files, columns=columns)


def protocol_patch_sources(input_protocol: str, weather_days: int) -> dict[str, list[str]]:
    sources = {"metadata": [], "static": [], "viirs": [], "weather": [], "osm": []}
    if input_protocol.startswith("metadata_"):
        sources["metadata"] = PATCH_METADATA_CHANNELS
        input_protocol = input_protocol.replace("metadata_", "", 1)
    if input_protocol in {"firms", "fire"}:
        sources["viirs"] = VIIRS_PATCH_CHANNELS
    elif input_protocol == "fire_wide":
        sources["viirs"] = VIIRS_PATCH_CHANNELS
    elif input_protocol == "weather":
        sources["weather"] = None
    elif input_protocol == "fuel":
        sources["static"] = FUEL_PATCH_CHANNELS
    elif input_protocol == "vegetation":
        sources["static"] = VEGETATION_PATCH_CHANNELS
    elif input_protocol == "topography":
        sources["static"] = TOPO_PATCH_CHANNELS
    elif input_protocol == "access":
        sources["osm"] = OSM_PATCH_CHANNELS
    elif input_protocol == "human":
        sources["static"] = ["pop_density"]
    elif input_protocol == "all":
        sources["static"] = STATIC_PATCH_CHANNELS
        sources["viirs"] = VIIRS_PATCH_CHANNELS
        sources["weather"] = None
        sources["osm"] = OSM_PATCH_CHANNELS
    elif input_protocol.startswith("all_without_"):
        removed = input_protocol.replace("all_without_", "", 1)
        static_channels = list(STATIC_PATCH_CHANNELS)
        if removed == "fire":
            sources["static"] = static_channels
            sources["weather"] = None
            sources["osm"] = OSM_PATCH_CHANNELS
        elif removed == "weather":
            sources["static"] = static_channels
            sources["viirs"] = VIIRS_PATCH_CHANNELS
            sources["osm"] = OSM_PATCH_CHANNELS
        elif removed == "vegetation":
            sources["static"] = [ch for ch in static_channels if ch not in VEGETATION_PATCH_CHANNELS]
            sources["viirs"] = VIIRS_PATCH_CHANNELS
            sources["weather"] = None
            sources["osm"] = OSM_PATCH_CHANNELS
        elif removed == "fuel":
            sources["static"] = [ch for ch in static_channels if ch not in FUEL_PATCH_CHANNELS]
            sources["viirs"] = VIIRS_PATCH_CHANNELS
            sources["weather"] = None
            sources["osm"] = OSM_PATCH_CHANNELS
        elif removed == "topography":
            sources["static"] = [ch for ch in static_channels if ch not in TOPO_PATCH_CHANNELS]
            sources["viirs"] = VIIRS_PATCH_CHANNELS
            sources["weather"] = None
            sources["osm"] = OSM_PATCH_CHANNELS
        elif removed == "access":
            sources["static"] = static_channels
            sources["viirs"] = VIIRS_PATCH_CHANNELS
            sources["weather"] = None
        elif removed == "human":
            sources["static"] = [ch for ch in static_channels if ch != "pop_density"]
            sources["viirs"] = VIIRS_PATCH_CHANNELS
            sources["weather"] = None
            sources["osm"] = OSM_PATCH_CHANNELS
        else:
            raise KeyError(input_protocol)
    elif input_protocol == "metadata":
        sources["metadata"] = PATCH_METADATA_CHANNELS
    else:
        raise KeyError(input_protocol)
    return sources


def weather_aggregate_patch_channels(path: Path, years: list[int], weather_days: int) -> list[str]:
    sample = read_patch_table(path, years[:1], columns=None)
    if sample.empty:
        return []
    key_cols = {"fire_id", "year", "split", "row", "col", "cell_id"}
    channels = [col for col in sample.columns if col not in key_cols and weather_feature_allowed(col, weather_days)]
    return channels


def read_joined_spatial_patch(
    paths: dict[str, Path],
    sample_df: pd.DataFrame,
    input_protocol: str,
    weather_days: int,
    include_weather_daily: bool = False,
    relative_days: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    years = sorted(sample_df["year"].unique().tolist())
    fire_ids = set(sample_df["fire_id"].astype(str))
    patch_key_cols = ["fire_id", "row", "col", "cell_id"]
    base_cols = patch_key_cols
    base = read_patch_table(paths["grid_index"], years, columns=base_cols)
    if base.empty:
        raise ValueError("Grid index patch table is empty or missing for requested years.")
    base["fire_id"] = base["fire_id"].astype(str)
    base = base.loc[base["fire_id"].isin(fire_ids)].copy()

    channel_names = []
    sources = protocol_patch_sources(input_protocol, weather_days)

    if sources["metadata"]:
        present = [c for c in sources["metadata"] if c in sample_df.columns]
        if present:
            metadata = sample_df[["fire_id"] + present].copy()
            metadata["fire_id"] = metadata["fire_id"].astype(str)
            base = base.merge(metadata, on="fire_id", how="left")
            channel_names.extend(present)

    if sources["static"]:
        cols = patch_key_cols + sources["static"]
        static = read_patch_table(paths["static_patch"], years, columns=cols)
        static["fire_id"] = static["fire_id"].astype(str)
        keep_cols = [c for c in cols if c in static.columns]
        static = static.loc[static["fire_id"].isin(fire_ids), keep_cols]
        present = [c for c in sources["static"] if c in static.columns]
        base = base.merge(static[["fire_id", "row", "col", "cell_id"] + present], on=["fire_id", "row", "col", "cell_id"], how="left")
        channel_names.extend(present)

    if sources["viirs"]:
        viirs = read_patch_table(paths["viirs_patch"], years, columns=patch_key_cols + sources["viirs"])
        viirs["fire_id"] = viirs["fire_id"].astype(str)
        present = [c for c in sources["viirs"] if c in viirs.columns]
        viirs = viirs.loc[viirs["fire_id"].isin(fire_ids), ["fire_id", "row", "col", "cell_id"] + present]
        base = base.merge(viirs, on=["fire_id", "row", "col", "cell_id"], how="left")
        channel_names.extend(present)

    if sources["weather"] is None and protocol_includes_weather(input_protocol) and not include_weather_daily:
        weather_channels = weather_aggregate_patch_channels(paths["weather_aggregate_patch"], years, weather_days)
        weather = read_patch_table(paths["weather_aggregate_patch"], years, columns=patch_key_cols + weather_channels)
        weather["fire_id"] = weather["fire_id"].astype(str)
        weather = weather.loc[weather["fire_id"].isin(fire_ids), ["fire_id", "row", "col", "cell_id"] + weather_channels]
        base = base.merge(weather, on=["fire_id", "row", "col", "cell_id"], how="left")
        channel_names.extend(weather_channels)

    if sources["osm"]:
        osm = read_patch_table(paths["osm_patch"], years, columns=patch_key_cols + sources["osm"])
        osm["fire_id"] = osm["fire_id"].astype(str)
        present = [c for c in sources["osm"] if c in osm.columns]
        osm = osm.loc[osm["fire_id"].isin(fire_ids), ["fire_id", "row", "col", "cell_id"] + present]
        base = base.merge(osm, on=["fire_id", "row", "col", "cell_id"], how="left")
        channel_names.extend(present)

    return base, list(dict.fromkeys(channel_names))


def patch_to_array(patch: pd.DataFrame, sample_df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    n = len(sample_df)
    c = len(channels)
    if c == 0:
        return np.zeros((n, 0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    order = pd.DataFrame({"fire_id": sample_df["fire_id"].astype(str), "fire_order": np.arange(n)})
    patch = patch.copy()
    patch["fire_id"] = patch["fire_id"].astype(str)
    patch = patch.merge(order, on="fire_id", how="inner")
    patch = patch.sort_values(["fire_order", "row", "col"])
    expected = n * CELLS_PER_PATCH
    if len(patch) != expected:
        raise ValueError(f"Patch cannot reshape: got {len(patch)} rows, expected {expected}.")
    values = patch[channels].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return values.reshape(n, PATCH_SIZE, PATCH_SIZE, c).transpose(0, 3, 1, 2)


def standardizable_patch_indices(channels: list[str]) -> list[int]:
    skip_exact = {
        "fbfm40",
        "fd",
        "fvt",
        "fvc",
        "fvh",
        "evt",
        "evc",
        "evh",
        "viirs_cell_has_detection_D",
        "cell_has_drivable_road",
    }
    return [idx for idx, ch in enumerate(channels) if ch not in skip_exact and not ch.endswith("_missing_mask")]


def fit_patch_stats_train(X_train: np.ndarray, channels: list[str], spatiotemporal: bool = False) -> dict:
    indices = standardizable_patch_indices(channels)
    if not indices:
        return {"indices": [], "mean": [], "std": []}
    if spatiotemporal:
        mean = X_train[:, :, indices, :, :].mean(axis=(0, 1, 3, 4))
        std = X_train[:, :, indices, :, :].std(axis=(0, 1, 3, 4))
    else:
        mean = X_train[:, indices, :, :].mean(axis=(0, 2, 3))
        std = X_train[:, indices, :, :].std(axis=(0, 2, 3))
    std = np.where(std == 0, 1.0, std)
    return {"indices": indices, "mean": mean.tolist(), "std": std.tolist()}


def apply_patch_standardization(X: np.ndarray, stats: dict, spatiotemporal: bool = False) -> np.ndarray:
    indices = stats["indices"]
    if not indices:
        return X
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    X = X.copy()
    if spatiotemporal:
        X[:, :, indices, :, :] = (X[:, :, indices, :, :] - mean.reshape(1, 1, -1, 1, 1)) / std.reshape(1, 1, -1, 1, 1)
    else:
        X[:, indices, :, :] = (X[:, indices, :, :] - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
    return X


def build_spatial_cache(args, paths: dict[str, Path]) -> Path:
    out_dir = output_subdir(args.output_dir, args.task, "spatial", args.weather_days, args.input_protocol)
    if (out_dir / "X_train.npy").exists() and not args.overwrite:
        print(f"Reusing existing spatial cache: {out_dir}")
        return out_dir
    remove_output_dir(out_dir, args.overwrite)

    master = normalize_master_metadata(read_parquet_robust(paths["master"]))
    samples = apply_input_protocol_sample_filter(filter_task_samples(master, args.task), args.input_protocol)
    splits = split_frames(samples)
    sample_index_splits = split_frames(build_sample_index(samples, master, args.task))

    split_arrays = {}
    channel_names = None
    for split, frame in splits.items():
        patch, channels = read_joined_spatial_patch(paths, frame, args.input_protocol, args.weather_days)
        if channel_names is None:
            channel_names = channels
        X = patch_to_array(patch, frame, channel_names)
        split_arrays[split] = X

    patch_stats = {"indices": [], "mean": [], "std": []}
    if args.standardize and channel_names:
        patch_stats = fit_patch_stats_train(split_arrays["train"], channel_names, spatiotemporal=False)
        for split in split_arrays:
            split_arrays[split] = apply_patch_standardization(split_arrays[split], patch_stats, spatiotemporal=False)

    for split, frame in splits.items():
        y = frame["target"].to_numpy(dtype=np.float32)
        write_split_outputs(out_dir, split, split_arrays[split], y, sample_index_splits[split], args.task)
    write_json(out_dir / "channel_names.json", {"channel_names": channel_names or []})
    save_metadata(
        out_dir,
        {
            "task": args.task,
            "representation": "spatial",
            "input_protocol": args.input_protocol,
            "weather_days": args.weather_days,
            "target_column": target_info(args.task)["target_column"],
            "channel_names": channel_names or [],
            "split_counts": {split: int(len(frame)) for split, frame in splits.items()},
            "missing_value_policy": "patch NaN filled with 0; no mask channels in v1",
            "scaler_policy": "channel-wise train mean/std for non-categorical non-binary patch channels" if args.standardize else "not_standardized",
            "channel_stats": patch_stats,
            "created_at": created_at(),
        },
    )
    print(f"Wrote spatial cache: {out_dir}")
    return out_dir


def daily_weather_patch_channels(paths: dict[str, Path], years: list[int]) -> list[str]:
    sample = read_patch_table(paths["weather_daily_patch"], years[:1], columns=None, relative_days=[0])
    if sample.empty:
        return []
    key_cols = {"fire_id", "year", "split", "row", "col", "cell_id", "date", "relative_day"}
    return [col for col in sample.columns if col not in key_cols and col in TEMPORAL_WEATHER_CHANNELS]


def weather_daily_patch_to_array(
    paths: dict[str, Path],
    sample_df: pd.DataFrame,
    rel_days: list[int],
    channels: list[str],
) -> np.ndarray:
    n = len(sample_df)
    t = len(rel_days)
    c = len(channels)
    if c == 0:
        return np.zeros((n, t, 0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    years = sorted(sample_df["year"].unique().tolist())
    fire_ids = set(sample_df["fire_id"].astype(str))
    order = pd.DataFrame({"fire_id": sample_df["fire_id"].astype(str), "fire_order": np.arange(n)})
    rel_order = pd.DataFrame({"relative_day": rel_days, "time_order": np.arange(t)})
    weather = read_patch_table(
        paths["weather_daily_patch"],
        years,
        columns=["fire_id", "relative_day", "row", "col", "cell_id"] + channels,
        relative_days=rel_days,
    )
    weather["fire_id"] = weather["fire_id"].astype(str)
    weather = weather.loc[weather["fire_id"].isin(fire_ids), ["fire_id", "relative_day", "row", "col", "cell_id"] + channels]
    weather = weather.merge(order, on="fire_id", how="inner").merge(rel_order, on="relative_day", how="inner")
    weather = weather.sort_values(["fire_order", "time_order", "row", "col"])
    expected = n * t * CELLS_PER_PATCH
    if len(weather) != expected:
        raise ValueError(f"Weather daily patch cannot reshape: got {len(weather)} rows, expected {expected}.")
    values = weather[channels].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return values.reshape(n, t, PATCH_SIZE, PATCH_SIZE, c).transpose(0, 1, 4, 2, 3)


def build_spatiotemporal_cache(args, paths: dict[str, Path]) -> Path:
    out_dir = output_subdir(args.output_dir, args.task, "spatiotemporal", args.weather_days, args.input_protocol)
    if (out_dir / "X_train.npy").exists() and not args.overwrite:
        print(f"Reusing existing spatiotemporal cache: {out_dir}")
        return out_dir
    remove_output_dir(out_dir, args.overwrite)

    master = normalize_master_metadata(read_parquet_robust(paths["master"]))
    samples = apply_input_protocol_sample_filter(filter_task_samples(master, args.task), args.input_protocol)
    splits = split_frames(samples)
    sample_index_splits = split_frames(build_sample_index(samples, master, args.task))
    rel_days = WEATHER_DAY_MAP[args.weather_days]

    split_arrays = {}
    channel_names = None
    for split, frame in splits.items():
        years = sorted(frame["year"].unique().tolist())
        dynamic_channels = daily_weather_patch_channels(paths, years) if protocol_includes_weather(args.input_protocol) else []
        X_weather = weather_daily_patch_to_array(paths, frame, rel_days, dynamic_channels)

        static_patch, static_channels = read_joined_spatial_patch(
            paths, frame, args.input_protocol, args.weather_days, include_weather_daily=True
        )
        X_static = patch_to_array(static_patch, frame, static_channels)
        X_static_repeated = np.repeat(X_static[:, None, :, :, :], len(rel_days), axis=1)
        X = np.concatenate([X_weather, X_static_repeated], axis=2)
        split_arrays[split] = X
        if channel_names is None:
            channel_names = dynamic_channels + static_channels

    patch_stats = {"indices": [], "mean": [], "std": []}
    if args.standardize and channel_names:
        patch_stats = fit_patch_stats_train(split_arrays["train"], channel_names, spatiotemporal=True)
        for split in split_arrays:
            split_arrays[split] = apply_patch_standardization(split_arrays[split], patch_stats, spatiotemporal=True)

    for split, frame in splits.items():
        y = frame["target"].to_numpy(dtype=np.float32)
        write_split_outputs(out_dir, split, split_arrays[split], y, sample_index_splits[split], args.task)
    write_json(out_dir / "channel_names.json", {"channel_names": channel_names or []})
    write_json(out_dir / "relative_days.json", {"relative_days": rel_days})
    save_metadata(
        out_dir,
        {
            "task": args.task,
            "representation": "spatiotemporal",
            "input_protocol": args.input_protocol,
            "weather_days": args.weather_days,
            "relative_days": rel_days,
            "target_column": target_info(args.task)["target_column"],
            "channel_names": channel_names or [],
            "split_counts": {split: int(len(frame)) for split, frame in splits.items()},
            "fire_signal_policy": "VIIRS discovery-day D patch repeated across all T time steps when selected",
            "missing_value_policy": "patch NaN filled with 0; no mask channels in v1",
            "scaler_policy": "channel-wise train mean/std for non-categorical non-binary patch channels" if args.standardize else "not_standardized",
            "channel_stats": patch_stats,
            "created_at": created_at(),
        },
    )
    print(f"Wrote spatiotemporal cache: {out_dir}")
    return out_dir


def build_representation(args, paths: dict[str, Path], feature_manifest: dict, representation: str) -> Path:
    if representation == "tabular":
        return build_tabular_cache(args, paths, feature_manifest)
    if representation == "temporal":
        return build_temporal_cache(args, paths, feature_manifest)
    if representation == "spatial":
        return build_spatial_cache(args, paths)
    if representation == "spatiotemporal":
        return build_spatiotemporal_cache(args, paths)
    raise ValueError(representation)


def pass_fail(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{': ' + detail if detail else ''}")
    return ok


def validate_patch_table_sample(name: str, path: Path, max_parts: int = 1, sample_fire_count: int = 3) -> bool:
    required = ["fire_id", "row", "col", "cell_id"]
    df = read_parquet_parts_sample(path, columns=required + ["relative_day"], max_parts=max_parts)
    if df.empty:
        return pass_fail(f"patch table sample reshape: {name}", False, "no parquet part files read")
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pass_fail(f"patch table sample reshape: {name}", False, f"missing keys={missing}")

    if "relative_day" in df.columns:
        first_rel = sorted(int(day) for day in df["relative_day"].dropna().unique())[0]
        df = df.loc[df["relative_day"].astype(int) == first_rel].copy()

    fire_ids = df["fire_id"].astype(str).drop_duplicates().head(sample_fire_count).tolist()
    if not fire_ids:
        return pass_fail(f"patch table sample reshape: {name}", False, "no fire_id values in sample")

    details = []
    ok = True
    df = df.copy()
    df["fire_id"] = df["fire_id"].astype(str)
    duplicate_count = int(df.duplicated(subset=required).sum())
    if duplicate_count:
        ok = False
        details.append(f"duplicate key rows={duplicate_count}")

    for fire_id in fire_ids:
        group = df.loc[df["fire_id"] == fire_id]
        checks = {
            "rows": len(group) == CELLS_PER_PATCH,
            "row_range": int(group["row"].min()) == 0 and int(group["row"].max()) == PATCH_SIZE - 1,
            "col_range": int(group["col"].min()) == 0 and int(group["col"].max()) == PATCH_SIZE - 1,
            "cell_id_nunique": int(group["cell_id"].nunique()) == CELLS_PER_PATCH,
        }
        bad = [key for key, passed in checks.items() if not passed]
        if bad:
            ok = False
            details.append(f"fire_id={fire_id} failed {bad}")

    detail = f"sampled_fire_ids={fire_ids}" if ok else "; ".join(details)
    return pass_fail(f"patch table sample reshape: {name}", ok, detail)


def validate(args, paths: dict[str, Path]) -> bool:
    print("Validation checks")
    ok_all = True
    for key in required_path_keys():
        ok_all &= pass_fail(f"canonical input exists: {key}", paths[key].exists(), str(paths[key]))
    if not paths["master"].exists():
        return False

    feature_manifest = load_json(paths["feature_manifest"]) if paths["feature_manifest"].exists() else {"forbidden_as_features": []}
    master = normalize_master_metadata(read_parquet_robust(paths["master"]))
    ok_all &= pass_fail("master_features has one row per fire_id", not master["fire_id"].duplicated().any())

    samples = apply_input_protocol_sample_filter(filter_task_samples(master, args.task), args.input_protocol)
    splits = split_frames(samples)
    ok_all &= pass_fail("task filtering works", len(samples) > 0, f"{len(samples)} samples")
    ok_all &= pass_fail(
        "train/val/test split counts are nonzero",
        all(len(frame) > 0 for frame in splits.values()),
        str({split: len(frame) for split, frame in splits.items()}),
    )

    target_col = target_info(args.task)["target_column"]
    feature_cols, removed = requested_feature_columns(
        feature_manifest, args.input_protocol, args.weather_days, master.columns, target_col
    )
    forbidden = set(feature_manifest.get("forbidden_as_features", []))
    ok_all &= pass_fail("no forbidden columns used as input features", not bool(set(feature_cols) & forbidden))
    ok_all &= pass_fail("target column is not an input feature", target_col not in feature_cols)
    future_cols = [col for col in feature_cols if any(token in col.lower() for token in ["d+1", "dplus", "early_48", "post"])]
    ok_all &= pass_fail("no D+1/D+2/future fire columns are used", len(future_cols) == 0, str(future_cols))

    if paths["weather_daily_event"].exists():
        weather_days = set(
            int(day)
            for day in read_parquet_robust(paths["weather_daily_event"], columns=["relative_day"])["relative_day"].unique()
        )
        need = set(WEATHER_DAY_MAP[args.weather_days])
        ok_all &= pass_fail(
            "weather relative days exist",
            need <= weather_days,
            f"need={sorted(int(day) for day in need)} have={sorted(int(day) for day in weather_days)}",
        )

    patch_tables = {
        "grid_index": paths["grid_index"],
        "static_patch": paths["static_patch"],
        "viirs_patch": paths["viirs_patch"],
        "weather_daily_patch": paths["weather_daily_patch"],
        "weather_aggregate_patch": paths["weather_aggregate_patch"],
        "osm_patch": paths["osm_patch"],
    }
    for name, path in patch_tables.items():
        if path.exists():
            ok_all &= validate_patch_table_sample(name, path)

    if args.representation != "all":
        cache_dir = output_subdir(args.output_dir, args.task, args.representation, args.weather_days, args.input_protocol)
        if (cache_dir / "X_train.npy").exists():
            X = np.load(cache_dir / "X_train.npy", mmap_mode="r")
            ok_all &= pass_fail("output arrays have expected shapes", X.shape[0] == len(splits["train"]), str(X.shape))
            ok_all &= pass_fail("no NaN remains in saved X arrays", not np.isnan(np.asarray(X[: min(len(X), 10)])).any())
        else:
            ok_all &= pass_fail("output arrays have expected shapes", False, f"cache not found: {cache_dir}")
    return ok_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 model-ready wildfire benchmark caches.")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--canonical_dir", default="./data/cache/raw_feature_tables")
    parser.add_argument("--output_dir", default="./data/cache/model_ready")
    parser.add_argument("--task", choices=["ia_failure", "containment_time"], default="ia_failure")
    parser.add_argument(
        "--representation",
        choices=["tabular", "temporal", "spatial", "spatiotemporal", "all"],
        default="all",
    )
    parser.add_argument("--weather_days", choices=[1, 2, 3, 4, 5], type=int, default=5)
    parser.add_argument(
        "--input_protocol",
        choices=[
            "metadata",
            "firms",
            "fire",
            "fire_wide",
            "weather",
            "fuel",
            "vegetation",
            "topography",
            "access",
            "human",
            "metadata_vegetation",
            "metadata_fuel",
            "metadata_topography",
            "metadata_access",
            "metadata_human",
            "all_without_fire",
            "all_without_weather",
            "all_without_vegetation",
            "all_without_fuel",
            "all_without_topography",
            "all_without_access",
            "all_without_human",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--standardize", dest="standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_dir = ensure_project_path(Path(args.base_dir))
    args.canonical_dir = ensure_project_path(Path(args.canonical_dir))
    args.output_dir = ensure_project_path(Path(args.output_dir))
    paths = canonical_paths(args.canonical_dir)

    if args.validate_only:
        ok = validate(args, paths)
        raise SystemExit(0 if ok else 1)

    missing = [key for key in required_path_keys() if not paths[key].exists()]
    if missing:
        details = "\n".join(f"  {key}: {paths[key]}" for key in missing)
        raise FileNotFoundError(f"Missing required Phase 1 canonical inputs:\n{details}")

    feature_manifest, _, _, _ = load_manifests(paths)
    reps = ["tabular", "temporal", "spatial", "spatiotemporal"] if args.representation == "all" else [args.representation]
    for representation in reps:
        build_representation(args, paths, feature_manifest, representation)


if __name__ == "__main__":
    main()
