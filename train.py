from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_READY_ROOT = Path("data/cache/model_ready")
EXPERIMENT_ROOT = Path("experiments")

MODEL_REGISTRY = {
    "logistic_regression": {"families": ["tabular"], "implemented": True},
    "xgboost": {"families": ["tabular"], "implemented": True},
    "mlp": {"families": ["tabular"], "implemented": True},
    "gru": {"families": ["temporal", "sequence"], "implemented": True},
    "tcn": {"families": ["temporal", "sequence"], "implemented": True},
    "transformer": {"families": ["temporal", "sequence"], "implemented": True},
    "resnet18_unet": {"families": ["spatial"], "implemented": True},
    "resnet50_unet": {"families": ["spatial"], "implemented": True},
    "swin_unet": {"families": ["spatial"], "implemented": True},
    "segformer": {"families": ["spatial"], "implemented": True},
    "convlstm": {"families": ["spatiotemporal"], "implemented": True},
    "convgru": {"families": ["spatiotemporal"], "implemented": True},
    "predrnn_v2": {"families": ["spatiotemporal"], "implemented": True},
    "utae": {"families": ["spatiotemporal"], "implemented": True},
    "swinlstm": {"families": ["spatiotemporal"], "implemented": True},
    "resnet3d": {"families": ["spatiotemporal"], "implemented": True},
}


def created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_project_path(path: Path) -> Path:
    resolved = path.resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path must stay inside {PROJECT_ROOT}: {resolved}")
    return resolved


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def output_model_name(task: str, model_name: str) -> str:
    if task == "containment_time" and model_name == "logistic_regression":
        return "ridge_regression"
    return model_name


def cache_representation_name(representation: str) -> str:
    return "temporal" if representation == "sequence" else representation


def canonical_representation_name(representation: str) -> str:
    return "temporal" if representation == "sequence" else representation


def cache_dir(base_dir: Path, task: str, representation: str, weather_days: int, input_protocol: str) -> Path:
    rep = cache_representation_name(representation)
    primary = base_dir / MODEL_READY_ROOT / task / rep / f"weather{weather_days}_{input_protocol}"
    if primary.exists():
        return primary
    if representation == "sequence":
        fallback = base_dir / MODEL_READY_ROOT / task / "sequence" / f"weather{weather_days}_{input_protocol}"
        if fallback.exists():
            return fallback
    return primary


def run_dir(
    base_dir: Path,
    task: str,
    experiment_type: str,
    ablation_name: str | None,
    run_tag: str | None,
    representation: str,
    weather_days: int,
    input_protocol: str,
    model_name: str,
    seed: int,
) -> Path:
    if experiment_type == "ablation" and ablation_name == "protocol_sweep" and run_tag:
        return (
            base_dir
            / EXPERIMENT_ROOT
            / task
            / "ablation"
            / "protocol_sweep"
            / run_tag
            / representation
            / f"weather{weather_days}_{input_protocol}"
            / f"{model_name}_seed{seed}"
        )
    if experiment_type == "ablation" and ablation_name:
        return (
            base_dir
            / EXPERIMENT_ROOT
            / task
            / "ablation"
            / ablation_name
            / representation
            / f"weather{weather_days}_{input_protocol}"
            / f"{model_name}_seed{seed}"
        )
    return (
        base_dir
        / EXPERIMENT_ROOT
        / task
        / experiment_type
        / representation
        / f"weather{weather_days}_{input_protocol}"
        / f"{model_name}_seed{seed}"
    )


def prepare_run_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists. Pass --overwrite to replace it: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_temporal_representation(representation: str) -> bool:
    return representation in {"temporal", "sequence"}


def load_split_array(cache: Path, split: str, representation: str) -> np.ndarray:
    if is_temporal_representation(representation):
        candidates = [cache / f"X_seq_{split}.npy", cache / f"X_{split}.npy"]
    else:
        candidates = [cache / f"X_{split}.npy"]
    for path in candidates:
        if path.exists():
            mmap_mode = "r" if canonical_representation_name(representation) in {"spatial", "spatiotemporal"} else None
            return np.load(path, mmap_mode=mmap_mode)
    raise FileNotFoundError(f"Missing X array for {split}. Tried: {candidates}")


def finite_check_array(x: np.ndarray, max_exact_values: int = 50_000_000) -> bool:
    """Avoid a full multi-GB scan for patch caches during smoke runs."""
    if x.size <= max_exact_values:
        return bool(np.isfinite(x).all())
    rng = np.random.default_rng(0)
    sample_size = min(512, x.shape[0])
    rows = rng.choice(x.shape[0], size=sample_size, replace=False)
    return bool(np.isfinite(x[rows]).all())


def load_cache(cache: Path, task: str, representation: str) -> dict[str, Any]:
    required_common = ["y", "fire_id", "sample_index"]
    data: dict[str, Any] = {"cache_dir": str(cache)}
    for split in ["train", "val", "test"]:
        for kind in required_common:
            suffix = "npy" if kind in {"y", "fire_id"} else "parquet"
            file = cache / f"{kind}_{split}.{suffix}"
            if not file.exists():
                raise FileNotFoundError(f"Missing required cache file: {file}")
        if is_temporal_representation(representation):
            x_seq_path = cache / f"X_seq_{split}.npy"
            x_static_path = cache / f"X_static_{split}.npy"
            if not x_seq_path.exists() or not x_static_path.exists():
                raise FileNotFoundError(f"Missing temporal cache arrays: {x_seq_path}, {x_static_path}")
            X = {
                "seq": np.load(x_seq_path),
                "static": np.load(x_static_path),
            }
        else:
            X = load_split_array(cache, split, representation)
        y = np.load(cache / f"y_{split}.npy")
        fire_id = np.load(cache / f"fire_id_{split}.npy", allow_pickle=True).astype(str)
        index = pd.read_parquet(cache / f"sample_index_{split}.parquet")
        data[split] = {"X": X, "y": y, "fire_id": fire_id, "index": index}
    data["metadata"] = load_json_if_exists(cache / "metadata.json")
    data["feature_names"] = load_json_if_exists(cache / "feature_names.json").get("feature_names", [])
    data["channel_names"] = load_json_if_exists(cache / "channel_names.json").get("channel_names", [])
    data["relative_days"] = load_json_if_exists(cache / "relative_days.json").get("relative_days", [])
    data["temporal_feature_names"] = load_json_if_exists(cache / "temporal_feature_names.json").get("feature_names", [])
    data["static_feature_names"] = load_json_if_exists(cache / "static_feature_names.json").get("feature_names", [])
    validate_cache(data, task)
    return data


def validate_cache(data: dict[str, Any], task: str) -> None:
    target_col = "ia_failure_label" if task == "ia_failure" else "log_containment_hours"
    print("Pre-training cache validation")
    for split in ["train", "val", "test"]:
        X = data[split]["X"]
        y = data[split]["y"]
        fire_id = data[split]["fire_id"]
        index = data[split]["index"].copy()
        if isinstance(X, dict):
            x_len = len(X["seq"])
            if len(X["static"]) != x_len:
                raise ValueError(f"{split}: X_seq and X_static lengths do not match.")
            x_shape = f"X_seq={X['seq'].shape}, X_static={X['static'].shape}"
            finite_ok = np.isfinite(X["seq"]).all() and np.isfinite(X["static"]).all()
        else:
            x_len = len(X)
            x_shape = f"X={X.shape}"
            finite_ok = finite_check_array(X)
        if x_len != len(y) or x_len != len(fire_id) or x_len != len(index):
            raise ValueError(f"{split}: X/y/fire_id/sample_index lengths do not match.")
        for col in ["fire_id", "year", "split"]:
            if col not in index.columns:
                raise ValueError(f"{split}: sample_index missing required column {col}")
        if target_col not in index.columns:
            raise ValueError(f"{split}: sample_index missing target column {target_col}")
        if not np.array_equal(index["fire_id"].astype(str).to_numpy(), fire_id):
            raise ValueError(f"{split}: fire_id array order does not match sample_index.")
        if task == "ia_failure":
            values = set(np.unique(y).astype(int).tolist())
            if not values <= {0, 1}:
                raise ValueError(f"{split}: ia_failure y has invalid values: {values}")
        else:
            if not np.isfinite(y).all():
                raise ValueError(f"{split}: containment_time y has non-finite values.")
        if not finite_ok:
            raise ValueError(f"{split}: X contains NaN or infinite values.")
        if task == "ia_failure":
            print(f"  {split}: {x_shape}, N={len(y)}, positive_rate={float(np.mean(y)):.4f}")
        else:
            print(
                f"  {split}: {x_shape}, N={len(y)}, y_log_mean={float(np.mean(y)):.4f}, "
                f"y_log_std={float(np.std(y)):.4f}"
            )


def device_info(device_arg: str, gpu_id: int | None) -> dict[str, Any]:
    info = {"device": "cpu", "gpu_name": None, "gpu_memory_total": None, "torch_cuda_available": False}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info["torch_cuda_available"] = bool(cuda_available)
        if device_arg == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        if device_arg == "auto":
            device = "cuda" if cuda_available else "cpu"
        else:
            device = device_arg
        if device == "cuda":
            index = gpu_id if gpu_id is not None else 0
            info["device"] = f"cuda:{index}"
            info["gpu_name"] = torch.cuda.get_device_name(index)
            info["gpu_memory_total"] = int(torch.cuda.get_device_properties(index).total_memory)
        else:
            info["device"] = "cpu"
    except ImportError:
        if device_arg == "cuda":
            raise
    return info


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score <= hi if i == n_bins - 1 else y_score < hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_score[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return ece


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, percent: int) -> tuple[float, float]:
    n = len(y_true)
    k = max(1, int(math.ceil((percent / 100.0) * n)))
    order = np.argsort(-y_score)[:k]
    positives = float(np.sum(y_true == 1))
    top_pos = float(np.sum(y_true[order] == 1))
    precision = top_pos / k
    recall = top_pos / positives if positives > 0 else float("nan")
    return precision, recall


def safe_metric(fn, default: float = float("nan")) -> float:
    try:
        return float(fn())
    except Exception:
        return default


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.clip(np.asarray(y_score).astype(float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "auprc": safe_metric(lambda: average_precision_score(y_true, y_score)),
        "auroc": safe_metric(lambda: roc_auc_score(y_true, y_score)),
        "f1": safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "precision": safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": safe_metric(lambda: balanced_accuracy_score(y_true, y_pred)),
        "brier": safe_metric(lambda: brier_score_loss(y_true, y_score)),
        "bce": safe_metric(lambda: log_loss(y_true, y_score, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_score),
    }
    for percent in [1, 5, 10]:
        precision, recall = precision_recall_at_k(y_true, y_score, percent)
        metrics[f"precision_at_{percent}"] = precision
        metrics[f"recall_at_{percent}"] = recall
    return metrics


def score_logit_diagnostics(y_score: np.ndarray, prefix: str) -> dict[str, float]:
    score = np.clip(np.asarray(y_score).astype(float), 1e-7, 1.0 - 1e-7)
    logit = safe_logit(score)
    return {
        f"{prefix}_y_score_mean": float(np.mean(score)),
        f"{prefix}_y_score_std": float(np.std(score)),
        f"{prefix}_y_score_min": float(np.min(score)),
        f"{prefix}_y_score_max": float(np.max(score)),
        f"{prefix}_y_logit_mean": float(np.mean(logit)),
        f"{prefix}_y_logit_std": float(np.std(logit)),
        f"{prefix}_y_logit_min": float(np.min(logit)),
        f"{prefix}_y_logit_max": float(np.max(logit)),
    }


def train_epoch_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    threshold, _ = best_f1_threshold(y_true, y_score)
    metrics = classification_metrics(y_true, y_score, threshold)
    return {
        "train_auprc": metrics["auprc"],
        "train_auroc": metrics["auroc"],
        "train_f1": metrics["f1"],
    }


def subset_indices(y: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    n = len(y)
    if limit is None or limit >= n:
        return np.arange(n)
    if limit <= 0:
        raise ValueError("Sample limits must be positive when provided.")
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if set(np.unique(y).astype(int).tolist()) <= {0, 1} and len(np.unique(y)) == 2:
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = max(1, int(round(limit * len(pos) / n)))
        n_pos = min(n_pos, len(pos), limit)
        n_neg = min(limit - n_pos, len(neg))
        chosen = np.concatenate([rng.choice(pos, size=n_pos, replace=False), rng.choice(neg, size=n_neg, replace=False)])
        if len(chosen) < limit:
            remaining = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
            chosen = np.concatenate([chosen, rng.choice(remaining, size=limit - len(chosen), replace=False)])
        return np.sort(chosen)
    return np.arange(limit)


def subset_split(split_data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    X = split_data["X"]
    if isinstance(X, dict):
        X = {name: value[indices] for name, value in X.items()}
    else:
        X = X[indices]
    return {
        "X": X,
        "y": split_data["y"][indices],
        "fire_id": split_data["fire_id"][indices],
        "index": split_data["index"].iloc[indices].reset_index(drop=True),
    }


def apply_debug_sample_limits(data: dict[str, Any], args) -> dict[str, Any]:
    limits = {"train": args.limit_train_samples, "val": args.limit_val_samples}
    if limits["train"] is None and limits["val"] is None:
        return data
    new_data = data.copy()
    for split, limit in limits.items():
        if limit is not None:
            idx = subset_indices(data[split]["y"], int(limit), args.seed)
            new_data[split] = subset_split(data[split], idx)
            print(f"Debug subset: {split} limited to {len(idx)} samples.")
    return new_data


def compute_channel_standardization_stats(X: np.ndarray, out_dir: Path, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    shape = X.shape
    if len(shape) == 4:
        channels = shape[1]
        reduce_axes = (0, 2, 3)
        count_per_sample = shape[2] * shape[3]
    elif len(shape) == 5:
        channels = shape[2]
        reduce_axes = (0, 1, 3, 4)
        count_per_sample = shape[1] * shape[3] * shape[4]
    else:
        raise ValueError(f"Channel standardization expects spatial or spatiotemporal X, got shape {shape}")
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    count = 0
    for start_idx in range(0, len(X), chunk_size):
        arr = np.asarray(X[start_idx : start_idx + chunk_size], dtype=np.float32)
        total += arr.sum(axis=reduce_axes, dtype=np.float64)
        total_sq += np.square(arr, dtype=np.float32).sum(axis=reduce_axes, dtype=np.float64)
        count += arr.shape[0] * count_per_sample
    mean = total / max(1, count)
    var = np.maximum(total_sq / max(1, count) - mean**2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    np.savez(out_dir / "channel_standardization_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_patch_batch(batch: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(batch, dtype=np.float32)
    if mean is None or std is None:
        return arr
    if arr.ndim == 4:
        return (arr - mean[None, :, None, None]) / std[None, :, None, None]
    if arr.ndim == 5:
        return (arr - mean[None, None, :, None, None]) / std[None, None, :, None, None]
    raise ValueError(f"Unexpected patch batch shape: {arr.shape}")


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray, containment_hours_true: np.ndarray | None = None) -> dict[str, float]:
    y_true_log = np.asarray(y_true_log).astype(float)
    y_pred_log = np.asarray(y_pred_log).astype(float)
    if containment_hours_true is None:
        containment_hours_true = np.expm1(y_true_log)
    containment_hours_pred = np.maximum(0.0, np.expm1(y_pred_log))
    return {
        "mae_hours": safe_metric(lambda: mean_absolute_error(containment_hours_true, containment_hours_pred)),
        "rmse_hours": safe_metric(lambda: mean_squared_error(containment_hours_true, containment_hours_pred, squared=False)),
        "median_ae_hours": safe_metric(lambda: median_absolute_error(containment_hours_true, containment_hours_pred)),
        "log_mae": safe_metric(lambda: mean_absolute_error(y_true_log, y_pred_log)),
        "log_rmse": safe_metric(lambda: mean_squared_error(y_true_log, y_pred_log, squared=False)),
        "r2": safe_metric(lambda: r2_score(y_true_log, y_pred_log)),
        "spearman": safe_metric(lambda: spearmanr(y_true_log, y_pred_log).correlation),
        "pearson": safe_metric(lambda: pearsonr(y_true_log, y_pred_log)[0]),
    }


def classification_predictions(index: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray, threshold: float, model_name: str, seed: int) -> pd.DataFrame:
    score = np.clip(np.asarray(y_score).astype(float), 1e-7, 1.0 - 1e-7)
    pred = (score >= threshold).astype(int)
    return pd.DataFrame(
        {
            "fire_id": index["fire_id"].astype(str).to_numpy(),
            "year": index["year"].to_numpy(),
            "split": index["split"].to_numpy(),
            "y_true": y_true.astype(int),
            "y_score": score,
            "y_logit": safe_logit(score),
            "y_pred": pred,
            "threshold": threshold,
            "model_name": model_name,
            "seed": seed,
        }
    )


def regression_predictions(index: pd.DataFrame, y_true_log: np.ndarray, y_pred_log: np.ndarray, model_name: str, seed: int) -> pd.DataFrame:
    true_hours = index["containment_hours"].to_numpy(dtype=float) if "containment_hours" in index.columns else np.expm1(y_true_log)
    pred_hours = np.maximum(0.0, np.expm1(y_pred_log))
    return pd.DataFrame(
        {
            "fire_id": index["fire_id"].astype(str).to_numpy(),
            "year": index["year"].to_numpy(),
            "split": index["split"].to_numpy(),
            "y_true_log": y_true_log,
            "y_pred_log": y_pred_log,
            "containment_hours_true": true_hours,
            "containment_hours_pred": pred_hours,
            "model_name": model_name,
            "seed": seed,
        }
    )


def base_metrics_payload(args, model_name: str, cache: Path, out_dir: Path, data: dict[str, Any], runtime_seconds: float, device: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task": args.task,
        "representation": args.representation,
        "model_name": model_name,
        "weather_days": args.weather_days,
        "input_protocol": args.input_protocol,
        "seed": args.seed,
        "train_size": int(len(data["train"]["y"])),
        "val_size": int(len(data["val"]["y"])),
        "test_size": int(len(data["test"]["y"])),
        "created_at": created_at(),
        "runtime_seconds": float(runtime_seconds),
        "device": device.get("device"),
        "gpu_name": device.get("gpu_name"),
        "gpu_memory_total": device.get("gpu_memory_total"),
        "input_cache_dir": str(cache),
        "output_dir": str(out_dir),
        "sampling_strategy": args.sampling_strategy,
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int((args.batch_size or default_batch_size(args.representation)) * args.grad_accum_steps),
        "standardize_channels": bool(args.standardize_channels),
        "limit_train_samples": args.limit_train_samples,
        "limit_val_samples": args.limit_val_samples,
        "disable_pos_weight": bool(args.disable_pos_weight),
        "pos_weight_scale": float(args.pos_weight_scale),
        "loss_type": args.loss_type,
        "label_smoothing": float(args.label_smoothing),
        "focal_gamma": float(args.focal_gamma),
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr_ratio": float(args.min_lr_ratio),
    }
    if args.task == "ia_failure":
        for split in ["train", "val", "test"]:
            y = data[split]["y"].astype(int)
            payload[f"{split}_positive"] = int(np.sum(y == 1))
            payload[f"{split}_positive_rate"] = float(np.mean(y))
    return payload


def ia_pos_weight_settings(args, y_train: np.ndarray) -> tuple[float, float | None, str]:
    """Return raw/effective IA class weight settings without changing defaults."""
    y = np.asarray(y_train).astype(int)
    num_pos = int(np.sum(y == 1))
    num_neg = int(np.sum(y == 0))
    raw_pos_weight = float(num_neg / max(1, num_pos))
    if args.disable_pos_weight:
        return raw_pos_weight, None, "BCEWithLogitsLoss unweighted"
    effective_pos_weight = float(raw_pos_weight * args.pos_weight_scale)
    return raw_pos_weight, effective_pos_weight, "BCEWithLogitsLoss pos_weight"


def bce_with_optional_pos_weight(nn_module, args, y_train: np.ndarray, torch_device):
    import torch

    class SmoothedBCEWithLogitsLoss(nn_module.Module):
        def __init__(self, pos_weight=None, smoothing: float = 0.0):
            super().__init__()
            self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
            self.smoothing = float(smoothing)

        def forward(self, logits, targets):
            targets = targets.float()
            if self.smoothing > 0:
                targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
            return nn_module.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
            )

    class FocalWithLogitsLoss(nn_module.Module):
        def __init__(self, pos_weight=None, smoothing: float = 0.0, gamma: float = 2.0):
            super().__init__()
            self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
            self.smoothing = float(smoothing)
            self.gamma = float(gamma)

        def forward(self, logits, targets):
            targets = targets.float()
            if self.smoothing > 0:
                targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
            bce = nn_module.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
                reduction="none",
            )
            prob = torch.sigmoid(logits)
            pt = prob * targets + (1.0 - prob) * (1.0 - targets)
            focal = (1.0 - pt).clamp(min=1e-6).pow(self.gamma) * bce
            return focal.mean()

    raw_pos_weight, effective_pos_weight, strategy = ia_pos_weight_settings(args, y_train)
    pos_weight_tensor = None
    if effective_pos_weight is None:
        strategy = f"{args.loss_type} unweighted"
    else:
        pos_weight_tensor = torch.tensor(effective_pos_weight, dtype=torch.float32, device=torch_device)
        strategy = f"{args.loss_type} pos_weight"
    if args.loss_type == "bce":
        criterion = SmoothedBCEWithLogitsLoss(pos_weight=pos_weight_tensor, smoothing=args.label_smoothing)
    elif args.loss_type == "focal":
        criterion = FocalWithLogitsLoss(pos_weight=pos_weight_tensor, smoothing=args.label_smoothing, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unsupported loss_type={args.loss_type}")
    criterion_unweighted = nn_module.BCEWithLogitsLoss()
    return criterion, criterion_unweighted, raw_pos_weight, effective_pos_weight, strategy


def build_neural_scheduler(optimizer, args, task: str):
    scheduler_type = args.lr_scheduler_type
    if args.use_lr_scheduler and scheduler_type == "none":
        scheduler_type = "plateau"
    if scheduler_type == "none":
        return None, "none"
    if scheduler_type == "plateau":
        mode = "max" if task == "ia_failure" else "min"
        return torch_scheduler_reduce_on_plateau(optimizer, mode=mode), "plateau"
    if scheduler_type == "cosine":
        import math
        import torch

        total_epochs = max(1, int(args.max_epochs))
        warmup_epochs = max(0, int(args.warmup_epochs))
        min_lr_ratio = max(0.0, min(1.0, float(args.min_lr_ratio)))

        def lr_lambda(epoch_idx: int):
            epoch_num = epoch_idx + 1
            if warmup_epochs > 0 and epoch_num <= warmup_epochs:
                return max(min_lr_ratio, epoch_num / warmup_epochs)
            decay_steps = max(1, total_epochs - warmup_epochs)
            progress = min(1.0, max(0.0, (epoch_num - warmup_epochs) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), "cosine"
    raise ValueError(f"Unsupported lr_scheduler_type={args.lr_scheduler_type}")


def torch_scheduler_reduce_on_plateau(optimizer, mode: str):
    import torch

    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=5)


def step_neural_scheduler(scheduler, scheduler_type: str, monitor: float | None = None) -> None:
    if scheduler is None:
        return
    if scheduler_type == "plateau":
        scheduler.step(monitor)
    else:
        scheduler.step()


def save_common_artifacts(out_dir: Path, cache: Path) -> None:
    for name in ["feature_names.json", "channel_names.json", "relative_days.json", "temporal_feature_names.json", "static_feature_names.json"]:
        src = cache / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def _plot_lines(
    history_df: pd.DataFrame,
    output_path: Path,
    title: str,
    y_label: str,
    columns: list[tuple[str, str]],
    best_epoch: int | None = None,
) -> None:
    present = [(col, label) for col, label in columns if col in history_df.columns]
    if not present or history_df.empty:
        return
    epoch = history_df["epoch"] if "epoch" in history_df.columns else np.arange(1, len(history_df) + 1)
    plt.figure(figsize=(8, 5))
    for col, label in present:
        plt.plot(epoch, history_df[col], marker="o", linewidth=1.8, markersize=3, label=label)
    if best_epoch is not None and best_epoch > 0:
        plt.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label=f"best epoch {best_epoch}")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_training_curves(history_df: pd.DataFrame, output_dir: Path, task: str, best_epoch: int | None = None) -> None:
    """Save training-curve PNGs when metric columns are available."""
    if history_df is None or history_df.empty:
        return
    output_dir = Path(output_dir)
    weighted_loss = False
    if task == "ia_failure":
        if "pos_weight_effective" in history_df.columns:
            weights = pd.to_numeric(history_df["pos_weight_effective"], errors="coerce").dropna()
            weighted_loss = bool(len(weights) and np.any(np.abs(weights.to_numpy(dtype=float) - 1.0) > 1e-9))
        else:
            weighted_loss = "val_bce_unweighted" in history_df.columns or "train_bce_unweighted" in history_df.columns
    loss_columns = [
        ("train_loss", "train_loss (weighted objective)" if weighted_loss else "train_loss"),
        ("val_loss", "val_loss (weighted objective)" if weighted_loss else "val_loss"),
    ]
    _plot_lines(
        history_df,
        output_dir / "loss_curve.png",
        "Training and Validation Loss",
        "loss",
        loss_columns,
        best_epoch=best_epoch,
    )
    _plot_lines(
        history_df,
        output_dir / "bce_unweighted_curve.png",
        "Unweighted BCE / NLL",
        "BCE",
        [("train_bce_unweighted", "train_BCE_unweighted"), ("val_bce_unweighted", "val_BCE_unweighted")],
        best_epoch=best_epoch,
    )
    if task == "ia_failure":
        _plot_lines(
            history_df,
            output_dir / "val_auprc_curve.png",
            "Validation AUPRC",
            "AUPRC",
            [("val_auprc", "val_AUPRC")],
            best_epoch=best_epoch,
        )
        _plot_lines(
            history_df,
            output_dir / "val_auroc_curve.png",
            "Validation AUROC",
            "AUROC",
            [("val_auroc", "val_AUROC")],
            best_epoch=best_epoch,
        )
        _plot_lines(
            history_df,
            output_dir / "metric_curve.png",
            "Validation Metrics",
            "metric value",
            [("val_auprc", "val_AUPRC"), ("val_auroc", "val_AUROC"), ("val_f1", "val_F1")],
            best_epoch=best_epoch,
        )
    elif task == "containment_time":
        _plot_lines(
            history_df,
            output_dir / "val_mae_curve.png",
            "Validation MAE",
            "MAE hours",
            [("val_mae_hours", "val_MAE_hours")],
            best_epoch=best_epoch,
        )
        _plot_lines(
            history_df,
            output_dir / "val_rmse_curve.png",
            "Validation RMSE",
            "RMSE hours",
            [("val_rmse_hours", "val_RMSE_hours")],
            best_epoch=best_epoch,
        )
        _plot_lines(
            history_df,
            output_dir / "metric_curve.png",
            "Validation Metrics",
            "metric value",
            [
                ("val_mae_hours", "val_MAE_hours"),
                ("val_rmse_hours", "val_RMSE_hours"),
                ("val_log_mae", "val_log_MAE"),
                ("val_log_rmse", "val_log_RMSE"),
            ],
            best_epoch=best_epoch,
        )


def train_logistic_or_ridge(args, data: dict[str, Any], out_dir: Path, cache: Path, device: dict[str, Any]) -> None:
    start = time.time()
    effective_name = output_model_name(args.task, "logistic_regression")
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    if args.task == "ia_failure":
        try:
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                C=1.0,
                random_state=args.seed,
                n_jobs=-1,
                verbose=0,
            )
            model.fit(X_train, y_train)
        except Exception as exc:
            print(f"Warning: LogisticRegression solver='lbfgs' failed ({exc}); retrying solver='saga'.")
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                solver="saga",
                C=1.0,
                random_state=args.seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]
        threshold, _ = best_f1_threshold(y_val, val_score)
        val_metrics = classification_metrics(y_val, val_score, threshold)
        test_metrics = classification_metrics(y_test, test_score, threshold)
        history = pd.DataFrame([{f"val_{k}": v for k, v in val_metrics.items()} | {"threshold": threshold}])
        pred_val = classification_predictions(data["val"]["index"], y_val, val_score, threshold, effective_name, args.seed)
        pred_test = classification_predictions(data["test"]["index"], y_test, test_score, threshold, effective_name, args.seed)
        imbalance = {
            "class_imbalance_strategy": 'LogisticRegression class_weight="balanced"',
            "pos_weight_or_scale_pos_weight": float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1))),
            "best_threshold_from_val": threshold,
        }
        imbalance.update(score_logit_diagnostics(val_score, "val"))
        imbalance.update(score_logit_diagnostics(test_score, "test"))
    else:
        model = Ridge(alpha=1.0, random_state=args.seed)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_metrics = regression_metrics(y_val, val_pred, data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val))).to_numpy(dtype=float))
        test_metrics = regression_metrics(y_test, test_pred, data["test"]["index"].get("containment_hours", pd.Series(np.expm1(y_test))).to_numpy(dtype=float))
        history = pd.DataFrame([{f"val_{k}": v for k, v in val_metrics.items()}])
        pred_val = regression_predictions(data["val"]["index"], y_val, val_pred, effective_name, args.seed)
        pred_test = regression_predictions(data["test"]["index"], y_test, test_pred, effective_name, args.seed)
        imbalance = {"class_imbalance_strategy": None, "pos_weight_or_scale_pos_weight": None}

    with (out_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    history.to_csv(out_dir / "history.csv", index=False)
    pred_val.to_parquet(out_dir / "predictions_val.parquet", index=False)
    pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)
    runtime = time.time() - start
    metrics = base_metrics_payload(args, effective_name, cache, out_dir, data, runtime, device)
    metrics.update(imbalance)
    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    write_json(out_dir / "metrics.json", metrics)


def xgb_predict_scores(model: Any, X: np.ndarray, task: str) -> np.ndarray:
    if task == "ia_failure":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def build_xgboost_model(args, device: dict[str, Any], use_cuda: bool, scale_pos_weight: float | None):
    from xgboost import XGBClassifier, XGBRegressor

    common = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda" if use_cuda else "cpu",
        "random_state": args.seed,
        "n_jobs": -1,
    }
    if args.task == "ia_failure":
        kwargs = {**common, "eval_metric": "aucpr"}
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**kwargs)
    return XGBRegressor(
        **common,
        objective="reg:squarederror",
        eval_metric="rmse",
    )


def fit_xgboost_with_fallback(model, X_train, y_train, X_val, y_val):
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
        return model, "early_stopping_rounds=50"
    except TypeError as exc:
        print(f"Warning: XGBoost early stopping API not supported ({exc}); retrying without early stopping.")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model, "no_early_stopping_api"


def train_xgboost(args, data: dict[str, Any], out_dir: Path, cache: Path, device: dict[str, Any]) -> None:
    start = time.time()
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise ImportError("xgboost is required for --model xgboost.") from exc

    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["val"]["X"], data["val"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    scale_pos_weight = None
    raw_scale_pos_weight = None
    imbalance_strategy = None
    if args.task == "ia_failure":
        raw_scale_pos_weight, scale_pos_weight, imbalance_strategy = ia_pos_weight_settings(args, y_train)
    use_cuda = str(device.get("device", "cpu")).startswith("cuda")
    model = build_xgboost_model(args, device, use_cuda=use_cuda, scale_pos_weight=scale_pos_weight)
    fit_note = ""
    try:
        model, fit_note = fit_xgboost_with_fallback(model, X_train, y_train, X_val, y_val)
    except Exception as exc:
        if use_cuda:
            print(f"Warning: XGBoost CUDA fit failed ({exc}); retrying on CPU.")
            model = build_xgboost_model(args, device, use_cuda=False, scale_pos_weight=scale_pos_weight)
            model, fit_note = fit_xgboost_with_fallback(model, X_train, y_train, X_val, y_val)
        else:
            raise

    if args.task == "ia_failure":
        val_score = xgb_predict_scores(model, X_val, args.task)
        test_score = xgb_predict_scores(model, X_test, args.task)
        threshold, _ = best_f1_threshold(y_val, val_score)
        val_metrics = classification_metrics(y_val, val_score, threshold)
        test_metrics = classification_metrics(y_test, test_score, threshold)
        pred_val = classification_predictions(data["val"]["index"], y_val, val_score, threshold, "xgboost", args.seed)
        pred_test = classification_predictions(data["test"]["index"], y_test, test_score, threshold, "xgboost", args.seed)
        extra = {
            "class_imbalance_strategy": "XGBClassifier scale_pos_weight" if scale_pos_weight is not None else "XGBClassifier unweighted",
            "raw_pos_weight_or_scale_pos_weight": raw_scale_pos_weight,
            "pos_weight_or_scale_pos_weight": scale_pos_weight,
            "disable_pos_weight": bool(args.disable_pos_weight),
            "pos_weight_scale": float(args.pos_weight_scale),
            "best_threshold_from_val": threshold,
        }
        extra.update(score_logit_diagnostics(val_score, "val"))
        extra.update(score_logit_diagnostics(test_score, "test"))
    else:
        val_pred = xgb_predict_scores(model, X_val, args.task)
        test_pred = xgb_predict_scores(model, X_test, args.task)
        val_metrics = regression_metrics(y_val, val_pred, data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val))).to_numpy(dtype=float))
        test_metrics = regression_metrics(y_test, test_pred, data["test"]["index"].get("containment_hours", pd.Series(np.expm1(y_test))).to_numpy(dtype=float))
        pred_val = regression_predictions(data["val"]["index"], y_val, val_pred, "xgboost", args.seed)
        pred_test = regression_predictions(data["test"]["index"], y_test, test_pred, "xgboost", args.seed)
        extra = {"class_imbalance_strategy": None, "pos_weight_or_scale_pos_weight": None}

    history = pd.DataFrame([{f"val_{k}": v for k, v in val_metrics.items()} | {"fit_note": fit_note}])
    with (out_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    history.to_csv(out_dir / "history.csv", index=False)
    pred_val.to_parquet(out_dir / "predictions_val.parquet", index=False)
    pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)
    runtime = time.time() - start
    metrics = base_metrics_payload(args, "xgboost", cache, out_dir, data, runtime, device)
    metrics.update(extra)
    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    metrics["xgboost_fit_note"] = fit_note
    write_json(out_dir / "metrics.json", metrics)


def train_mlp(args, data: dict[str, Any], out_dir: Path, cache: Path, device: dict[str, Any]) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    except ImportError as exc:
        raise ImportError("PyTorch is required for --model mlp.") from exc

    start = time.time()
    torch_device = torch.device(device["device"] if str(device.get("device", "cpu")).startswith("cuda") else "cpu")
    X_train = torch.tensor(data["train"]["X"], dtype=torch.float32)
    y_train = torch.tensor(data["train"]["y"], dtype=torch.float32)
    X_val = torch.tensor(data["val"]["X"], dtype=torch.float32)
    y_val_np = data["val"]["y"].astype(np.float32)
    X_test = torch.tensor(data["test"]["X"], dtype=torch.float32)
    y_test_np = data["test"]["y"].astype(np.float32)

    class TabularMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = TabularMLP(X_train.shape[1]).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.task == "ia_failure":
        criterion, criterion_unweighted, raw_pos_weight, pos_weight, imbalance_strategy = bce_with_optional_pos_weight(
            nn, args, data["train"]["y"], torch_device
        )
        maximize = True
        best_metric_name = "val_auprc"
    else:
        raw_pos_weight = None
        pos_weight = None
        imbalance_strategy = None
        criterion = nn.HuberLoss()
        criterion_unweighted = None
        maximize = False
        best_metric_name = "val_mae_hours"
    scheduler, scheduler_type = build_neural_scheduler(optimizer, args, args.task)

    batch_size = args.batch_size or 512
    sampler = None
    shuffle = True
    if args.task == "ia_failure" and args.sampling_strategy == "weighted":
        y_np = data["train"]["y"].astype(int)
        class_counts = np.bincount(y_np, minlength=2).astype(float)
        weights = 1.0 / np.maximum(class_counts[y_np], 1.0)
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)
    history = []
    best_value = -float("inf") if maximize else float("inf")
    best_epoch = -1
    patience = 0

    def predict_numpy(X_tensor: Any) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start_idx in range(0, len(X_tensor), batch_size * 4):
                batch = X_tensor[start_idx : start_idx + batch_size * 4].to(torch_device)
                outputs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outputs)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_unweighted_bce = 0.0
        total_n = 0
        train_logits_epoch = []
        train_y_epoch = []
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(torch_device)
            yb = yb.to(torch_device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if criterion_unweighted is not None:
                unweighted_loss = criterion_unweighted(pred, yb)
                total_unweighted_bce += float(unweighted_loss.detach().cpu()) * len(xb)
                train_logits_epoch.append(pred.detach().cpu().numpy())
                train_y_epoch.append(yb.detach().cpu().numpy())
            (loss / max(1, args.grad_accum_steps)).backward()
            if batch_idx % args.grad_accum_steps == 0 or batch_idx == num_batches:
                if args.gradient_clip and args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_n += len(xb)
        train_loss = total_loss / max(1, total_n)
        train_bce_unweighted = total_unweighted_bce / max(1, total_n) if criterion_unweighted is not None else None
        val_raw = predict_numpy(X_val)
        with torch.no_grad():
            val_loss = float(criterion(torch.tensor(val_raw, dtype=torch.float32, device=torch_device), torch.tensor(y_val_np, dtype=torch.float32, device=torch_device)).detach().cpu())
            val_bce_unweighted = None
            if criterion_unweighted is not None:
                val_bce_unweighted = float(
                    criterion_unweighted(
                        torch.tensor(val_raw, dtype=torch.float32, device=torch_device),
                        torch.tensor(y_val_np, dtype=torch.float32, device=torch_device),
                    ).detach().cpu()
                )

        if args.task == "ia_failure":
            val_score = sigmoid(val_raw)
            threshold, _ = best_f1_threshold(y_val_np, val_score)
            metrics = classification_metrics(y_val_np, val_score, threshold)
            train_score = sigmoid(np.concatenate(train_logits_epoch)) if train_logits_epoch else np.array([])
            train_y_diag = np.concatenate(train_y_epoch) if train_y_epoch else np.array([])
            train_metrics = train_epoch_metrics(train_y_diag, train_score) if len(train_score) else {}
            monitor = metrics["auprc"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auprc": metrics["auprc"],
                "val_auroc": metrics["auroc"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_brier": metrics["brier"],
                "val_bce": metrics["bce"],
                "val_ece": metrics["ece"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_bce_unweighted": train_bce_unweighted,
                "val_bce_unweighted": val_bce_unweighted,
                "pos_weight_raw": raw_pos_weight,
                "pos_weight_effective": pos_weight,
            }
            row.update(train_metrics)
        else:
            hours_true = data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val_np))).to_numpy(dtype=float)
            metrics = regression_metrics(y_val_np, val_raw, hours_true)
            monitor = metrics["mae_hours"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae_hours": metrics["mae_hours"],
                "val_rmse_hours": metrics["rmse_hours"],
                "val_median_ae_hours": metrics["median_ae_hours"],
                "val_log_mae": metrics["log_mae"],
                "val_log_rmse": metrics["log_rmse"],
                "val_r2": metrics["r2"],
                "val_spearman": metrics["spearman"],
                "val_pearson": metrics["pearson"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        if scheduler is not None:
            step_neural_scheduler(scheduler, scheduler_type, monitor)
            row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        improved = monitor > best_value if maximize else monitor < best_value
        if improved:
            best_value = monitor
            best_epoch = epoch
            patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metric": best_value}, out_dir / "best_checkpoint.pt")
        else:
            patience += 1
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metric": monitor}, out_dir / "last_checkpoint.pt")
        if patience >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, {best_metric_name}={best_value:.6f}")
            break

    checkpoint = torch.load(out_dir / "best_checkpoint.pt", map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_raw = predict_numpy(X_val)
    test_raw = predict_numpy(X_test)

    if args.task == "ia_failure":
        val_score = sigmoid(val_raw)
        test_score = sigmoid(test_raw)
        threshold, _ = best_f1_threshold(y_val_np, val_score)
        val_metrics = classification_metrics(y_val_np, val_score, threshold)
        test_metrics = classification_metrics(y_test_np, test_score, threshold)
        pred_val = classification_predictions(data["val"]["index"], y_val_np, val_score, threshold, "mlp", args.seed)
        pred_test = classification_predictions(data["test"]["index"], y_test_np, test_score, threshold, "mlp", args.seed)
        if args.save_train_predictions:
            train_raw = predict_numpy(X_train)
            train_score = sigmoid(train_raw)
            classification_predictions(data["train"]["index"], data["train"]["y"].astype(np.float32), train_score, threshold, "mlp", args.seed).to_parquet(
                out_dir / "predictions_train.parquet", index=False
            )
        extra = {
            "class_imbalance_strategy": imbalance_strategy,
            "raw_pos_weight_or_scale_pos_weight": raw_pos_weight,
            "pos_weight_or_scale_pos_weight": pos_weight,
            "disable_pos_weight": bool(args.disable_pos_weight),
            "pos_weight_scale": float(args.pos_weight_scale),
            "loss_type": args.loss_type,
            "label_smoothing": float(args.label_smoothing),
            "focal_gamma": float(args.focal_gamma),
            "lr_scheduler_type": scheduler_type,
            "best_threshold_from_val": threshold,
        }
        extra.update(score_logit_diagnostics(val_score, "val"))
        extra.update(score_logit_diagnostics(test_score, "test"))
    else:
        val_hours = data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val_np))).to_numpy(dtype=float)
        test_hours = data["test"]["index"].get("containment_hours", pd.Series(np.expm1(y_test_np))).to_numpy(dtype=float)
        val_metrics = regression_metrics(y_val_np, val_raw, val_hours)
        test_metrics = regression_metrics(y_test_np, test_raw, test_hours)
        pred_val = regression_predictions(data["val"]["index"], y_val_np, val_raw, "mlp", args.seed)
        pred_test = regression_predictions(data["test"]["index"], y_test_np, test_raw, "mlp", args.seed)
        extra = {"class_imbalance_strategy": None, "pos_weight_or_scale_pos_weight": None}

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    plot_training_curves(history_df, out_dir, args.task, best_epoch=best_epoch)
    pred_val.to_parquet(out_dir / "predictions_val.parquet", index=False)
    pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)
    runtime = time.time() - start
    metrics_payload = base_metrics_payload(args, "mlp", cache, out_dir, data, runtime, device)
    metrics_payload.update(extra)
    metrics_payload.update({f"val_{k}": v for k, v in val_metrics.items()})
    metrics_payload.update({f"test_{k}": v for k, v in test_metrics.items()})
    metrics_payload["best_epoch"] = int(best_epoch)
    write_json(out_dir / "metrics.json", metrics_payload)


def train_temporal_neural(args, data: dict[str, Any], out_dir: Path, cache: Path, device: dict[str, Any], model_name: str) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    except ImportError as exc:
        raise ImportError("PyTorch is required for temporal neural models.") from exc

    start = time.time()
    torch_device = torch.device(device["device"] if str(device.get("device", "cpu")).startswith("cuda") else "cpu")
    X_seq_train = torch.tensor(data["train"]["X"]["seq"], dtype=torch.float32)
    X_static_train = torch.tensor(data["train"]["X"]["static"], dtype=torch.float32)
    y_train = torch.tensor(data["train"]["y"], dtype=torch.float32)
    X_seq_val = torch.tensor(data["val"]["X"]["seq"], dtype=torch.float32)
    X_static_val = torch.tensor(data["val"]["X"]["static"], dtype=torch.float32)
    y_val_np = data["val"]["y"].astype(np.float32)
    X_seq_test = torch.tensor(data["test"]["X"]["seq"], dtype=torch.float32)
    X_static_test = torch.tensor(data["test"]["X"]["static"], dtype=torch.float32)
    y_test_np = data["test"]["y"].astype(np.float32)

    seq_dim = X_seq_train.shape[-1]
    static_dim = X_static_train.shape[-1]

    class StaticEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.output_dim = output_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            if x.shape[1] == 0:
                return torch.zeros((x.shape[0], self.output_dim), dtype=x.dtype, device=x.device)
            return self.net(x)

    class GRUClassifier(nn.Module):
        def __init__(self, seq_dim: int, static_dim: int):
            super().__init__()
            hidden = 128
            self.gru = nn.GRU(seq_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.static_encoder = StaticEncoder(static_dim, args.static_hidden, args.static_out)
            self.head = nn.Sequential(
                nn.Linear(hidden + args.static_out, args.fusion_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x_seq, x_static):
            _, h = self.gru(x_seq)
            seq_emb = h[-1]
            static_emb = self.static_encoder(x_static)
            return self.head(torch.cat([seq_emb, static_emb], dim=1)).squeeze(-1)

    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            out = self.net(x)
            if out.shape[-1] != x.shape[-1]:
                out = out[..., : x.shape[-1]]
            return out + self.proj(x)

    class TCNClassifier(nn.Module):
        def __init__(self, seq_dim: int, static_dim: int):
            super().__init__()
            hidden = 128
            self.tcn = nn.Sequential(
                TemporalBlock(seq_dim, hidden, dilation=1, dropout=args.dropout),
                TemporalBlock(hidden, hidden, dilation=2, dropout=args.dropout),
                TemporalBlock(hidden, hidden, dilation=4, dropout=args.dropout),
            )
            self.static_encoder = StaticEncoder(static_dim, args.static_hidden, args.static_out)
            self.head = nn.Sequential(
                nn.Linear(hidden + args.static_out, args.fusion_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x_seq, x_static):
            seq_emb = self.tcn(x_seq.transpose(1, 2)).mean(dim=-1)
            static_emb = self.static_encoder(x_static)
            return self.head(torch.cat([seq_emb, static_emb], dim=1)).squeeze(-1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 64):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerClassifier(nn.Module):
        def __init__(self, seq_dim: int, static_dim: int):
            super().__init__()
            d_model = 128
            self.input_proj = nn.Linear(seq_dim, d_model)
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.static_encoder = StaticEncoder(static_dim, args.static_hidden, args.static_out)
            self.head = nn.Sequential(
                nn.Linear(d_model + args.static_out, args.fusion_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x_seq, x_static):
            seq_tokens = self.pos(self.input_proj(x_seq))
            seq_emb = self.encoder(seq_tokens).mean(dim=1)
            static_emb = self.static_encoder(x_static)
            return self.head(torch.cat([seq_emb, static_emb], dim=1)).squeeze(-1)

    if model_name == "gru":
        model = GRUClassifier(seq_dim, static_dim)
    elif model_name == "tcn":
        model = TCNClassifier(seq_dim, static_dim)
    elif model_name == "transformer":
        model = TransformerClassifier(seq_dim, static_dim)
    else:
        raise ValueError(model_name)
    model = model.to(torch_device)

    if args.task == "ia_failure":
        criterion, criterion_unweighted, raw_pos_weight, pos_weight, imbalance_strategy = bce_with_optional_pos_weight(
            nn, args, data["train"]["y"], torch_device
        )
        maximize = True
        best_metric_name = "val_auprc"
    else:
        criterion = nn.HuberLoss()
        criterion_unweighted = None
        raw_pos_weight = None
        pos_weight = None
        imbalance_strategy = None
        maximize = False
        best_metric_name = "val_mae_hours"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, scheduler_type = build_neural_scheduler(optimizer, args, args.task)
    batch_size = args.batch_size or 256
    sampler = None
    shuffle = True
    if args.task == "ia_failure" and args.sampling_strategy == "weighted":
        y_np = data["train"]["y"].astype(int)
        class_counts = np.bincount(y_np, minlength=2).astype(float)
        weights = 1.0 / np.maximum(class_counts[y_np], 1.0)
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    loader = DataLoader(
        TensorDataset(X_seq_train, X_static_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )
    history = []
    best_value = -float("inf") if maximize else float("inf")
    best_epoch = -1
    patience = 0

    def predict_numpy(seq_tensor, static_tensor) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start_idx in range(0, len(seq_tensor), batch_size * 4):
                seq_batch = seq_tensor[start_idx : start_idx + batch_size * 4].to(torch_device)
                static_batch = static_tensor[start_idx : start_idx + batch_size * 4].to(torch_device)
                outputs.append(model(seq_batch, static_batch).detach().cpu().numpy())
        return np.concatenate(outputs)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_unweighted_bce = 0.0
        total_n = 0
        train_logits_epoch = []
        train_y_epoch = []
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for batch_idx, (xb_seq, xb_static, yb) in enumerate(loader, start=1):
            xb_seq = xb_seq.to(torch_device)
            xb_static = xb_static.to(torch_device)
            yb = yb.to(torch_device)
            pred = model(xb_seq, xb_static)
            loss = criterion(pred, yb)
            (loss / max(1, args.grad_accum_steps)).backward()
            if batch_idx % args.grad_accum_steps == 0 or batch_idx == num_batches:
                if args.gradient_clip and args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * len(yb)
            if criterion_unweighted is not None:
                unweighted_loss = criterion_unweighted(pred, yb)
                total_unweighted_bce += float(unweighted_loss.detach().cpu()) * len(yb)
                train_logits_epoch.append(pred.detach().cpu().numpy())
                train_y_epoch.append(yb.detach().cpu().numpy())
            total_n += len(yb)
        train_loss = total_loss / max(1, total_n)
        train_bce_unweighted = total_unweighted_bce / max(1, total_n) if criterion_unweighted is not None else None
        val_raw = predict_numpy(X_seq_val, X_static_val)
        with torch.no_grad():
            val_loss = float(
                criterion(
                    torch.tensor(val_raw, dtype=torch.float32, device=torch_device),
                    torch.tensor(y_val_np, dtype=torch.float32, device=torch_device),
                ).detach().cpu()
            )
            val_bce_unweighted = None
            if criterion_unweighted is not None:
                val_bce_unweighted = float(
                    criterion_unweighted(
                        torch.tensor(val_raw, dtype=torch.float32, device=torch_device),
                        torch.tensor(y_val_np, dtype=torch.float32, device=torch_device),
                    ).detach().cpu()
                )
        if args.task == "ia_failure":
            val_score = sigmoid(val_raw)
            threshold, _ = best_f1_threshold(y_val_np, val_score)
            metrics = classification_metrics(y_val_np, val_score, threshold)
            train_score = sigmoid(np.concatenate(train_logits_epoch)) if train_logits_epoch else np.array([])
            train_y_diag = np.concatenate(train_y_epoch) if train_y_epoch else np.array([])
            train_metrics = train_epoch_metrics(train_y_diag, train_score) if len(train_score) else {}
            monitor = metrics["auprc"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auprc": metrics["auprc"],
                "val_auroc": metrics["auroc"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_brier": metrics["brier"],
                "val_bce": metrics["bce"],
                "val_ece": metrics["ece"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_bce_unweighted": train_bce_unweighted,
                "val_bce_unweighted": val_bce_unweighted,
                "pos_weight_raw": raw_pos_weight,
                "pos_weight_effective": pos_weight,
            }
            row.update(train_metrics)
        else:
            hours_true = data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val_np))).to_numpy(dtype=float)
            metrics = regression_metrics(y_val_np, val_raw, hours_true)
            monitor = metrics["mae_hours"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae_hours": metrics["mae_hours"],
                "val_rmse_hours": metrics["rmse_hours"],
                "val_median_ae_hours": metrics["median_ae_hours"],
                "val_log_mae": metrics["log_mae"],
                "val_log_rmse": metrics["log_rmse"],
                "val_r2": metrics["r2"],
                "val_spearman": metrics["spearman"],
                "val_pearson": metrics["pearson"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        if scheduler is not None:
            step_neural_scheduler(scheduler, scheduler_type, monitor)
            row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        improved = monitor > best_value if maximize else monitor < best_value
        if improved:
            best_value = monitor
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "metric": best_value,
                    "seq_dim": int(seq_dim),
                    "static_dim": int(static_dim),
                },
                out_dir / "best_checkpoint.pt",
            )
        else:
            patience += 1
        torch.save(
            {
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "metric": monitor,
                "seq_dim": int(seq_dim),
                "static_dim": int(static_dim),
            },
            out_dir / "last_checkpoint.pt",
        )
        if args.task == "ia_failure":
            print(f"{model_name} epoch {epoch}: train_loss={train_loss:.4f} val_auprc={metrics['auprc']:.4f} val_auroc={metrics['auroc']:.4f}", flush=True)
        else:
            print(f"{model_name} epoch {epoch}: train_loss={train_loss:.4f} val_mae_hours={metrics['mae_hours']:.4f} val_rmse_hours={metrics['rmse_hours']:.4f}", flush=True)
        if patience >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, {best_metric_name}={best_value:.6f}")
            break

    checkpoint = torch.load(out_dir / "best_checkpoint.pt", map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_raw = predict_numpy(X_seq_val, X_static_val)
    test_raw = predict_numpy(X_seq_test, X_static_test)
    if args.task == "ia_failure":
        val_score = sigmoid(val_raw)
        test_score = sigmoid(test_raw)
        threshold, _ = best_f1_threshold(y_val_np, val_score)
        val_metrics = classification_metrics(y_val_np, val_score, threshold)
        test_metrics = classification_metrics(y_test_np, test_score, threshold)
        pred_val = classification_predictions(data["val"]["index"], y_val_np, val_score, threshold, model_name, args.seed)
        pred_test = classification_predictions(data["test"]["index"], y_test_np, test_score, threshold, model_name, args.seed)
        if args.save_train_predictions:
            train_raw = predict_numpy(X_seq_train, X_static_train)
            train_score = sigmoid(train_raw)
            classification_predictions(data["train"]["index"], data["train"]["y"].astype(np.float32), train_score, threshold, model_name, args.seed).to_parquet(
                out_dir / "predictions_train.parquet", index=False
            )
        extra = {
            "class_imbalance_strategy": imbalance_strategy,
            "raw_pos_weight_or_scale_pos_weight": raw_pos_weight,
            "pos_weight_or_scale_pos_weight": pos_weight,
            "disable_pos_weight": bool(args.disable_pos_weight),
            "pos_weight_scale": float(args.pos_weight_scale),
            "loss_type": args.loss_type,
            "label_smoothing": float(args.label_smoothing),
            "focal_gamma": float(args.focal_gamma),
            "lr_scheduler_type": scheduler_type,
            "best_threshold_from_val": threshold,
            "best_epoch": int(best_epoch),
            "seq_dim": int(seq_dim),
            "static_dim": int(static_dim),
        }
        extra.update(score_logit_diagnostics(val_score, "val"))
        extra.update(score_logit_diagnostics(test_score, "test"))
    else:
        val_hours = data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val_np))).to_numpy(dtype=float)
        test_hours = data["test"]["index"].get("containment_hours", pd.Series(np.expm1(y_test_np))).to_numpy(dtype=float)
        val_metrics = regression_metrics(y_val_np, val_raw, val_hours)
        test_metrics = regression_metrics(y_test_np, test_raw, test_hours)
        pred_val = regression_predictions(data["val"]["index"], y_val_np, val_raw, model_name, args.seed)
        pred_test = regression_predictions(data["test"]["index"], y_test_np, test_raw, model_name, args.seed)
        extra = {
            "class_imbalance_strategy": None,
            "pos_weight_or_scale_pos_weight": None,
            "lr_scheduler_type": scheduler_type,
            "best_epoch": int(best_epoch),
            "seq_dim": int(seq_dim),
            "static_dim": int(static_dim),
        }

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    plot_training_curves(history_df, out_dir, args.task, best_epoch=best_epoch)
    pred_val.to_parquet(out_dir / "predictions_val.parquet", index=False)
    pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)
    runtime = time.time() - start
    metrics_payload = base_metrics_payload(args, model_name, cache, out_dir, data, runtime, device)
    metrics_payload.update(extra)
    metrics_payload.update({f"val_{k}": v for k, v in val_metrics.items()})
    metrics_payload.update({f"test_{k}": v for k, v in test_metrics.items()})
    write_json(out_dir / "metrics.json", metrics_payload)


def build_patch_model(model_name: str, input_shape: tuple[int, ...], args, torch_device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvNormAct(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
            super().__init__()
            padding = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride)
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.skip = (
                nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
                if in_ch != out_ch or stride != 1
                else nn.Identity()
            )

        def forward(self, x):
            return F.relu(self.conv2(self.conv1(x)) + self.skip(x), inplace=True)

    class BottleneckBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            mid = max(out_ch // 4, 16)
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.skip = (
                nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
                if in_ch != out_ch or stride != 1
                else nn.Identity()
            )

        def forward(self, x):
            return F.relu(self.net(x) + self.skip(x), inplace=True)

    def make_layer(block, in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [block(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    class ResNetUNetClassifier(nn.Module):
        def __init__(self, in_channels: int, variant: str):
            super().__init__()
            base = 32
            if variant == "resnet18_unet":
                block, counts = ResidualBlock, [2, 2, 2, 2]
            else:
                block, counts = BottleneckBlock, [3, 4, 6, 3]
            self.stem = ConvNormAct(in_channels, base)
            self.enc1 = make_layer(block, base, base, counts[0], stride=1)
            self.enc2 = make_layer(block, base, base * 2, counts[1], stride=2)
            self.enc3 = make_layer(block, base * 2, base * 4, counts[2], stride=2)
            self.enc4 = make_layer(block, base * 4, base * 8, counts[3], stride=2)
            self.dec3 = ConvNormAct(base * 8 + base * 4, base * 4)
            self.dec2 = ConvNormAct(base * 4 + base * 2, base * 2)
            self.dec1 = ConvNormAct(base * 2 + base, base)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base, args.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x):
            s0 = self.stem(x)
            s1 = self.enc1(s0)
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            z = self.enc4(s3)
            z = F.interpolate(z, size=s3.shape[-2:], mode="bilinear", align_corners=False)
            z = self.dec3(torch.cat([z, s3], dim=1))
            z = F.interpolate(z, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            z = self.dec2(torch.cat([z, s2], dim=1))
            z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            z = self.dec1(torch.cat([z, s1], dim=1))
            return self.head(z).squeeze(-1)

    class WindowAttentionBlock(nn.Module):
        """Lightweight Swin-style local window attention for small wildfire patches."""

        def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7, dropout: float = 0.0):
            super().__init__()
            self.window_size = window_size
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            ws = self.window_size
            pad_h = (ws - h % ws) % ws
            pad_w = (ws - w % ws) % ws
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            hp, wp = x.shape[-2:]
            tokens = x.permute(0, 2, 3, 1).contiguous()
            windows = tokens.view(b, hp // ws, ws, wp // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
            attn_in = self.norm1(windows)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            windows = windows + attn_out
            windows = windows + self.mlp(self.norm2(windows))
            tokens = windows.view(b, hp // ws, wp // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
            out = tokens.permute(0, 3, 1, 2).contiguous()
            return out[:, :, :h, :w]

    class SwinUNetClassifier(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            base = 48
            self.stem = ConvNormAct(in_channels, base)
            self.enc1 = nn.Sequential(WindowAttentionBlock(base, num_heads=4, window_size=7, dropout=args.dropout), ConvNormAct(base, base))
            self.down = ConvNormAct(base, base * 2, stride=2)
            self.enc2 = nn.Sequential(WindowAttentionBlock(base * 2, num_heads=4, window_size=5, dropout=args.dropout), ConvNormAct(base * 2, base * 2))
            self.bottleneck = nn.Sequential(ConvNormAct(base * 2, base * 2), WindowAttentionBlock(base * 2, num_heads=4, window_size=5, dropout=args.dropout))
            self.fuse = ConvNormAct(base * 3, base)
            self.dec_attn = WindowAttentionBlock(base, num_heads=4, window_size=7, dropout=args.dropout)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base, args.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x):
            s1 = self.enc1(self.stem(x))
            z = self.enc2(self.down(s1))
            z = self.bottleneck(z)
            z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            z = self.fuse(torch.cat([z, s1], dim=1))
            z = self.dec_attn(z)
            return self.head(z).squeeze(-1)

    class MixFFN(nn.Module):
        def __init__(self, dim: int, dropout: float):
            super().__init__()
            self.fc1 = nn.Conv2d(dim, dim * 2, kernel_size=1)
            self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2)
            self.fc2 = nn.Conv2d(dim * 2, dim, kernel_size=1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.fc2(self.drop(F.gelu(self.dwconv(self.fc1(x)))))

    class SpatialTransformerBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, dropout: float):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.BatchNorm2d(dim)
            self.ffn = MixFFN(dim, dropout)

        def forward(self, x):
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            attn_in = self.norm1(tokens)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            x = (tokens + attn_out).transpose(1, 2).reshape(b, c, h, w)
            x = x + self.ffn(self.norm2(x))
            return x

    class SegFormerClassifier(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            dims = [48, 96, 160]
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.ReLU(inplace=True),
                SpatialTransformerBlock(dims[0], num_heads=4, dropout=args.dropout),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[1]),
                nn.ReLU(inplace=True),
                SpatialTransformerBlock(dims[1], num_heads=4, dropout=args.dropout),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[2]),
                nn.ReLU(inplace=True),
                SpatialTransformerBlock(dims[2], num_heads=4, dropout=args.dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dims[2], args.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x):
            return self.head(self.stage3(self.stage2(self.stage1(x)))).squeeze(-1)

    class ConvLSTMCell(nn.Module):
        def __init__(self, in_ch: int, hidden_ch: int):
            super().__init__()
            self.hidden_ch = hidden_ch
            self.gates = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 4, kernel_size=3, padding=1)

        def forward(self, x, h, c):
            gates = self.gates(torch.cat([x, h], dim=1))
            i, f, o, g = torch.chunk(gates, 4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            return h, c

    class ConvLSTMClassifier(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            hidden = 64
            self.cell = ConvLSTMCell(in_ch, hidden)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, args.fusion_hidden), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.fusion_hidden, 1))

        def forward(self, x):
            b, t, _, h, w = x.shape
            h_state = x.new_zeros((b, 64, h, w))
            c_state = x.new_zeros((b, 64, h, w))
            for step in range(t):
                h_state, c_state = self.cell(x[:, step], h_state, c_state)
            return self.head(h_state).squeeze(-1)

    class ConvGRUCell(nn.Module):
        def __init__(self, in_ch: int, hidden_ch: int):
            super().__init__()
            self.hidden_ch = hidden_ch
            self.reset_update = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 2, kernel_size=3, padding=1)
            self.out_gate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size=3, padding=1)

        def forward(self, x, h):
            z_r = self.reset_update(torch.cat([x, h], dim=1))
            z, r = torch.chunk(z_r, 2, dim=1)
            z, r = torch.sigmoid(z), torch.sigmoid(r)
            n = torch.tanh(self.out_gate(torch.cat([x, r * h], dim=1)))
            return (1.0 - z) * h + z * n

    class ConvGRUClassifier(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            hidden = 64
            self.cell = ConvGRUCell(in_ch, hidden)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, args.fusion_hidden), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.fusion_hidden, 1))

        def forward(self, x):
            b, t, _, h, w = x.shape
            h_state = x.new_zeros((b, 64, h, w))
            for step in range(t):
                h_state = self.cell(x[:, step], h_state)
            return self.head(h_state).squeeze(-1)

    class PredRNNV2Cell(nn.Module):
        def __init__(self, in_ch: int, hidden_ch: int):
            super().__init__()
            self.hidden_ch = hidden_ch
            self.x_proj = nn.Conv2d(in_ch, hidden_ch * 4, kernel_size=3, padding=1)
            self.h_proj = nn.Conv2d(hidden_ch, hidden_ch * 4, kernel_size=3, padding=1)
            self.m_proj = nn.Conv2d(hidden_ch, hidden_ch * 3, kernel_size=3, padding=1)
            self.fuse = nn.Conv2d(hidden_ch * 2, hidden_ch, kernel_size=1)

        def forward(self, x, h, c, m):
            xi, xf, xo, xg = torch.chunk(self.x_proj(x), 4, dim=1)
            hi, hf, ho, hg = torch.chunk(self.h_proj(h), 4, dim=1)
            mi, mf, mg = torch.chunk(self.m_proj(m), 3, dim=1)
            i = torch.sigmoid(xi + hi)
            f = torch.sigmoid(xf + hf + 1.0)
            g = torch.tanh(xg + hg)
            c = f * c + i * g
            i_m = torch.sigmoid(xi + mi)
            f_m = torch.sigmoid(xf + mf + 1.0)
            g_m = torch.tanh(xg + mg)
            m = f_m * m + i_m * g_m
            fused = self.fuse(torch.cat([c, m], dim=1))
            o = torch.sigmoid(xo + ho)
            h = o * torch.tanh(fused)
            return h, c, m

    class PredRNNV2Classifier(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            hidden = 64
            self.cell = PredRNNV2Cell(in_ch, hidden)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, args.fusion_hidden), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.fusion_hidden, 1))

        def forward(self, x):
            b, t, _, h, w = x.shape
            h_state = x.new_zeros((b, 64, h, w))
            c_state = x.new_zeros((b, 64, h, w))
            m_state = x.new_zeros((b, 64, h, w))
            for step in range(t):
                h_state, c_state, m_state = self.cell(x[:, step], h_state, c_state, m_state)
            return self.head(h_state).squeeze(-1)

    class BasicBlock3D(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int] = (1, 1, 1)):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            self.skip = (
                nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(out_ch))
                if in_ch != out_ch or stride != (1, 1, 1)
                else nn.Identity()
            )

        def forward(self, x):
            return F.relu(self.net(x) + self.skip(x), inplace=True)

    class ResNet3DClassifier(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            base = 32
            self.net = nn.Sequential(
                nn.Conv3d(in_ch, base, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(base),
                nn.ReLU(inplace=True),
                BasicBlock3D(base, base),
                BasicBlock3D(base, base * 2, stride=(1, 2, 2)),
                BasicBlock3D(base * 2, base * 4, stride=(1, 2, 2)),
                BasicBlock3D(base * 4, base * 8, stride=(1, 2, 2)),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(base * 8, args.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2)).squeeze(-1)

    class UTAEClassifier(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            hidden = 64
            self.encoder = nn.Sequential(
                ConvNormAct(in_ch, hidden),
                ResidualBlock(hidden, hidden),
                ConvNormAct(hidden, hidden * 2, stride=2),
                ResidualBlock(hidden * 2, hidden * 2),
            )
            self.attn = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.fuse = ConvNormAct(hidden * 2, hidden * 2)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden * 2, args.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x):
            b, t, c, h, w = x.shape
            feat = self.encoder(x.reshape(b * t, c, h, w))
            _, ch, fh, fw = feat.shape
            feat = feat.view(b, t, ch, fh, fw)
            desc = feat.mean(dim=(-2, -1))
            weights = torch.softmax(self.attn(desc).squeeze(-1), dim=1)
            agg = (feat * weights[:, :, None, None, None]).sum(dim=1)
            return self.head(self.fuse(agg)).squeeze(-1)

    class SwinLSTMClassifier(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            hidden = 48
            self.hidden = hidden
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                WindowAttentionBlock(hidden, num_heads=4, window_size=5, dropout=args.dropout),
            )
            self.cell = ConvLSTMCell(hidden, hidden)
            self.post_attn = WindowAttentionBlock(hidden, num_heads=4, window_size=5, dropout=args.dropout)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden, args.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(args.fusion_hidden, 1),
            )

        def forward(self, x):
            b, t, _, _, _ = x.shape
            first = self.stem(x[:, 0])
            h_state = x.new_zeros((b, self.hidden, first.shape[-2], first.shape[-1]))
            c_state = x.new_zeros((b, self.hidden, first.shape[-2], first.shape[-1]))
            h_state, c_state = self.cell(first, h_state, c_state)
            for step in range(t):
                if step == 0:
                    continue
                x_step = self.stem(x[:, step])
                h_state, c_state = self.cell(x_step, h_state, c_state)
            h_state = self.post_attn(h_state)
            return self.head(h_state).squeeze(-1)

    if len(input_shape) == 3:
        in_ch = int(input_shape[0])
        if model_name in {"resnet18_unet", "resnet50_unet"}:
            return ResNetUNetClassifier(in_ch, model_name).to(torch_device)
        if model_name == "swin_unet":
            return SwinUNetClassifier(in_ch).to(torch_device)
        if model_name == "segformer":
            return SegFormerClassifier(in_ch).to(torch_device)
    if len(input_shape) == 4:
        in_ch = int(input_shape[1])
        if model_name == "convlstm":
            return ConvLSTMClassifier(in_ch).to(torch_device)
        if model_name == "convgru":
            return ConvGRUClassifier(in_ch).to(torch_device)
        if model_name == "predrnn_v2":
            return PredRNNV2Classifier(in_ch).to(torch_device)
        if model_name == "resnet3d":
            return ResNet3DClassifier(in_ch).to(torch_device)
        if model_name == "utae":
            return UTAEClassifier(in_ch).to(torch_device)
        if model_name == "swinlstm":
            return SwinLSTMClassifier(in_ch).to(torch_device)
    raise ValueError(f"Model {model_name} does not match input shape {input_shape}.")


def train_patch_neural(args, data: dict[str, Any], out_dir: Path, cache: Path, device: dict[str, Any], model_name: str) -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch is required for patch neural models.") from exc

    start = time.time()
    torch_device = torch.device(device["device"] if str(device.get("device", "cpu")).startswith("cuda") else "cpu")
    X_train = data["train"]["X"]
    X_val = data["val"]["X"]
    X_test = data["test"]["X"]
    y_train_np = data["train"]["y"].astype(np.float32)
    y_val_np = data["val"]["y"].astype(np.float32)
    y_test_np = data["test"]["y"].astype(np.float32)
    channel_mean = None
    channel_std = None
    if args.standardize_channels:
        print("Computing train-only channel standardization stats...", flush=True)
        channel_mean, channel_std = compute_channel_standardization_stats(X_train, out_dir)
    model = build_patch_model(model_name, tuple(X_train.shape[1:]), args, torch_device)
    if args.task == "ia_failure":
        criterion, criterion_unweighted, raw_pos_weight, pos_weight, imbalance_strategy = bce_with_optional_pos_weight(
            nn, args, y_train_np, torch_device
        )
        maximize = True
        best_metric_name = "val_auprc"
    else:
        criterion = nn.HuberLoss()
        criterion_unweighted = None
        raw_pos_weight = None
        pos_weight = None
        imbalance_strategy = None
        maximize = False
        best_metric_name = "val_mae_hours"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, scheduler_type = build_neural_scheduler(optimizer, args, args.task)
    batch_size = args.batch_size or default_batch_size(canonical_representation_name(args.representation))
    sampling_prob = None
    if args.task == "ia_failure" and args.sampling_strategy == "weighted":
        class_counts = np.bincount(y_train_np.astype(int), minlength=2).astype(float)
        weights = 1.0 / np.maximum(class_counts[y_train_np.astype(int)], 1.0)
        sampling_prob = weights / weights.sum()
    history = []
    best_value = -float("inf") if maximize else float("inf")
    best_epoch = -1
    patience = 0

    def predict_numpy(X) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start_idx in range(0, len(X), batch_size * 2):
                batch_np = standardize_patch_batch(X[start_idx : start_idx + batch_size * 2], channel_mean, channel_std)
                batch = torch.tensor(batch_np, dtype=torch.float32, device=torch_device)
                outputs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outputs)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_unweighted_bce = 0.0
        total_n = 0
        if sampling_prob is None:
            order = np.random.permutation(len(y_train_np))
        else:
            order = np.random.choice(np.arange(len(y_train_np)), size=len(y_train_np), replace=True, p=sampling_prob)
        train_logits_epoch = []
        train_y_epoch = []
        optimizer.zero_grad(set_to_none=True)
        num_batches = int(math.ceil(len(order) / batch_size))
        for start_idx in range(0, len(order), batch_size):
            batch_number = start_idx // batch_size + 1
            batch_idx = order[start_idx : start_idx + batch_size]
            xb_np = standardize_patch_batch(X_train[batch_idx], channel_mean, channel_std)
            xb = torch.tensor(xb_np, dtype=torch.float32, device=torch_device)
            yb = torch.tensor(y_train_np[batch_idx], dtype=torch.float32, device=torch_device)
            pred = model(xb)
            loss = criterion(pred, yb)
            (loss / max(1, args.grad_accum_steps)).backward()
            if batch_number % args.grad_accum_steps == 0 or batch_number == num_batches:
                if args.gradient_clip and args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * len(yb)
            if criterion_unweighted is not None:
                unweighted_loss = criterion_unweighted(pred, yb)
                total_unweighted_bce += float(unweighted_loss.detach().cpu()) * len(yb)
                train_logits_epoch.append(pred.detach().cpu().numpy())
                train_y_epoch.append(yb.detach().cpu().numpy())
            total_n += len(yb)
        train_loss = total_loss / max(1, total_n)
        train_bce_unweighted = total_unweighted_bce / max(1, total_n) if criterion_unweighted is not None else None
        val_raw = predict_numpy(X_val)
        with torch.no_grad():
            val_raw_t = torch.tensor(val_raw, dtype=torch.float32, device=torch_device)
            y_val_t = torch.tensor(y_val_np, dtype=torch.float32, device=torch_device)
            val_loss = float(criterion(val_raw_t, y_val_t).detach().cpu())
            val_bce_unweighted = None
            if criterion_unweighted is not None:
                val_bce_unweighted = float(criterion_unweighted(val_raw_t, y_val_t).detach().cpu())
        if args.task == "ia_failure":
            val_score = sigmoid(val_raw)
            threshold, _ = best_f1_threshold(y_val_np, val_score)
            metrics = classification_metrics(y_val_np, val_score, threshold)
            train_score = sigmoid(np.concatenate(train_logits_epoch)) if train_logits_epoch else np.array([])
            train_y_diag = np.concatenate(train_y_epoch) if train_y_epoch else np.array([])
            train_metrics = train_epoch_metrics(train_y_diag, train_score) if len(train_score) else {}
            monitor = metrics["auprc"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auprc": metrics["auprc"],
                "val_auroc": metrics["auroc"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_brier": metrics["brier"],
                "val_bce": metrics["bce"],
                "val_ece": metrics["ece"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_bce_unweighted": train_bce_unweighted,
                "val_bce_unweighted": val_bce_unweighted,
                "pos_weight_raw": raw_pos_weight,
                "pos_weight_effective": pos_weight,
            }
            row.update(train_metrics)
        else:
            hours_true = data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val_np))).to_numpy(dtype=float)
            metrics = regression_metrics(y_val_np, val_raw, hours_true)
            monitor = metrics["mae_hours"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae_hours": metrics["mae_hours"],
                "val_rmse_hours": metrics["rmse_hours"],
                "val_median_ae_hours": metrics["median_ae_hours"],
                "val_log_mae": metrics["log_mae"],
                "val_log_rmse": metrics["log_rmse"],
                "val_r2": metrics["r2"],
                "val_spearman": metrics["spearman"],
                "val_pearson": metrics["pearson"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        if scheduler is not None:
            step_neural_scheduler(scheduler, scheduler_type, monitor)
            row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        improved = monitor > best_value if maximize else monitor < best_value
        if improved:
            best_value = monitor
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "metric": best_value,
                    "input_shape": list(X_train.shape[1:]),
                },
                out_dir / "best_checkpoint.pt",
            )
        else:
            patience += 1
        torch.save(
            {
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "metric": monitor,
                "input_shape": list(X_train.shape[1:]),
            },
            out_dir / "last_checkpoint.pt",
        )
        if args.task == "ia_failure":
            print(f"{model_name} epoch {epoch}: train_loss={train_loss:.4f} val_auprc={metrics['auprc']:.4f} val_auroc={metrics['auroc']:.4f}", flush=True)
        else:
            print(f"{model_name} epoch {epoch}: train_loss={train_loss:.4f} val_mae_hours={metrics['mae_hours']:.4f} val_rmse_hours={metrics['rmse_hours']:.4f}", flush=True)
        if patience >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, {best_metric_name}={best_value:.6f}")
            break

    checkpoint = torch.load(out_dir / "best_checkpoint.pt", map_location=torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_raw = predict_numpy(X_val)
    test_raw = predict_numpy(X_test)
    if args.task == "ia_failure":
        val_score = sigmoid(val_raw)
        test_score = sigmoid(test_raw)
        threshold, _ = best_f1_threshold(y_val_np, val_score)
        val_metrics = classification_metrics(y_val_np, val_score, threshold)
        test_metrics = classification_metrics(y_test_np, test_score, threshold)
        pred_val = classification_predictions(data["val"]["index"], y_val_np, val_score, threshold, model_name, args.seed)
        pred_test = classification_predictions(data["test"]["index"], y_test_np, test_score, threshold, model_name, args.seed)
        if args.save_train_predictions:
            train_raw = predict_numpy(X_train)
            train_score = sigmoid(train_raw)
            classification_predictions(data["train"]["index"], y_train_np, train_score, threshold, model_name, args.seed).to_parquet(
                out_dir / "predictions_train.parquet", index=False
            )
        extra = {
            "class_imbalance_strategy": imbalance_strategy,
            "raw_pos_weight_or_scale_pos_weight": raw_pos_weight,
            "pos_weight_or_scale_pos_weight": pos_weight,
            "disable_pos_weight": bool(args.disable_pos_weight),
            "pos_weight_scale": float(args.pos_weight_scale),
            "loss_type": args.loss_type,
            "label_smoothing": float(args.label_smoothing),
            "focal_gamma": float(args.focal_gamma),
            "lr_scheduler_type": scheduler_type,
            "best_threshold_from_val": threshold,
            "best_epoch": int(best_epoch),
            "input_shape": list(X_train.shape[1:]),
        }
        extra.update(score_logit_diagnostics(val_score, "val"))
        extra.update(score_logit_diagnostics(test_score, "test"))
    else:
        val_hours = data["val"]["index"].get("containment_hours", pd.Series(np.expm1(y_val_np))).to_numpy(dtype=float)
        test_hours = data["test"]["index"].get("containment_hours", pd.Series(np.expm1(y_test_np))).to_numpy(dtype=float)
        val_metrics = regression_metrics(y_val_np, val_raw, val_hours)
        test_metrics = regression_metrics(y_test_np, test_raw, test_hours)
        pred_val = regression_predictions(data["val"]["index"], y_val_np, val_raw, model_name, args.seed)
        pred_test = regression_predictions(data["test"]["index"], y_test_np, test_raw, model_name, args.seed)
        extra = {
            "class_imbalance_strategy": None,
            "pos_weight_or_scale_pos_weight": None,
            "lr_scheduler_type": scheduler_type,
            "best_epoch": int(best_epoch),
            "input_shape": list(X_train.shape[1:]),
        }

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    plot_training_curves(history_df, out_dir, args.task, best_epoch=best_epoch)
    pred_val.to_parquet(out_dir / "predictions_val.parquet", index=False)
    pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)
    runtime = time.time() - start
    metrics_payload = base_metrics_payload(args, model_name, cache, out_dir, data, runtime, device)
    metrics_payload.update(extra)
    metrics_payload.update({f"val_{k}": v for k, v in val_metrics.items()})
    metrics_payload.update({f"test_{k}": v for k, v in test_metrics.items()})
    write_json(out_dir / "metrics.json", metrics_payload)


def train_one(args, model_name: str) -> None:
    registry = MODEL_REGISTRY[model_name]
    if not registry["implemented"]:
        raise NotImplementedError(f"Model {model_name} is registered but not implemented in this first version.")
    representation = canonical_representation_name(args.representation)
    if representation not in registry["families"] and not (representation == "tabular" and model_name in {"logistic_regression", "xgboost", "mlp"}):
        raise ValueError(f"Model {model_name} does not support representation={args.representation}.")
    if representation not in {"tabular", "temporal", "spatial", "spatiotemporal"}:
        raise NotImplementedError("This train.py version supports tabular, temporal, spatial, and spatiotemporal models.")

    if args.gpu_id is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    set_seed(args.seed)
    device = device_info(args.device, None if os.environ.get("CUDA_VISIBLE_DEVICES") else args.gpu_id)
    cache = cache_dir(args.base_dir, args.task, args.representation, args.weather_days, args.input_protocol)
    if not cache.exists():
        raise FileNotFoundError(f"Input cache directory does not exist: {cache}")
    data = load_cache(cache, args.task, args.representation)
    data = apply_debug_sample_limits(data, args)
    effective_name = output_model_name(args.task, model_name)
    out_dir = run_dir(
        args.base_dir,
        args.task,
        args.experiment_type,
        args.ablation_name,
        args.run_tag,
        args.representation,
        args.weather_days,
        args.input_protocol,
        effective_name,
        args.seed,
    )
    prepare_run_dir(out_dir, args.overwrite)
    config = vars(args).copy()
    config["model_name"] = effective_name
    config["input_cache_dir"] = str(cache)
    config["output_dir"] = str(out_dir)
    config["created_at"] = created_at()
    config["effective_batch_size"] = int((args.batch_size or default_batch_size(args.representation)) * args.grad_accum_steps)
    write_json(out_dir / "config.json", config)
    save_common_artifacts(out_dir, cache)

    if model_name == "logistic_regression":
        train_logistic_or_ridge(args, data, out_dir, cache, device)
    elif model_name == "xgboost":
        train_xgboost(args, data, out_dir, cache, device)
    elif model_name == "mlp":
        train_mlp(args, data, out_dir, cache, device)
    elif model_name in {"gru", "tcn", "transformer"}:
        train_temporal_neural(args, data, out_dir, cache, device, model_name)
    elif model_name in {"resnet18_unet", "resnet50_unet", "swin_unet", "segformer", "convlstm", "convgru", "predrnn_v2", "resnet3d", "utae", "swinlstm"}:
        train_patch_neural(args, data, out_dir, cache, device, model_name)
    else:
        raise NotImplementedError(f"Model {model_name} is registered but not implemented in this first version.")
    print(f"Wrote experiment: {out_dir}")


def default_batch_size(representation: str) -> int:
    return {
        "tabular": 512,
        "temporal": 256,
        "sequence": 256,
        "spatial": 64,
        "spatiotemporal": 32,
    }[representation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wildfire Initial Attack benchmark models from model-ready caches.")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--task", choices=["ia_failure", "containment_time"], default="ia_failure")
    parser.add_argument("--experiment_type", choices=["smoke", "full", "ablation"], default="full")
    parser.add_argument("--ablation_name", default=None)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--representation", choices=["tabular", "temporal", "sequence", "spatial", "spatiotemporal"], default="tabular")
    parser.add_argument("--weather_days", choices=[1, 2, 3, 4, 5], type=int, default=5)
    parser.add_argument("--input_protocol", default="all")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY) + ["all"], default="xgboost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--lr_scheduler_type", choices=["none", "plateau", "cosine"], default="none")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--standardize_channels", action="store_true")
    parser.add_argument("--sampling_strategy", choices=["random", "weighted"], default="random")
    parser.add_argument("--disable_pos_weight", action="store_true")
    parser.add_argument("--pos_weight_scale", type=float, default=1.0)
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    parser.add_argument("--save_train_predictions", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--static_hidden", type=int, default=256)
    parser.add_argument("--static_out", type=int, default=128)
    parser.add_argument("--fusion_hidden", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.base_dir = ensure_project_path(Path(args.base_dir))
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.pos_weight_scale < 0:
        raise ValueError("--pos_weight_scale must be >= 0")
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("--label_smoothing must be in [0, 1)")
    if args.focal_gamma < 0:
        raise ValueError("--focal_gamma must be >= 0")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup_epochs must be >= 0")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min_lr_ratio must be in [0, 1]")
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.representation)
    return args


def main() -> None:
    args = parse_args()
    if args.model == "all":
        representation = canonical_representation_name(args.representation)
        if representation == "tabular":
            model_names = ["logistic_regression", "xgboost", "mlp"]
        else:
            model_names = [
                name
                for name, spec in MODEL_REGISTRY.items()
                if representation in spec["families"] and spec["implemented"]
            ]
            skipped = [
                name
                for name, spec in MODEL_REGISTRY.items()
                if representation in spec["families"] and not spec["implemented"]
            ]
            for name in skipped:
                print(f"Warning: skipping unimplemented registered model: {name}")
        for name in model_names:
            train_one(args, name)
    else:
        train_one(args, args.model)


if __name__ == "__main__":
    main()
