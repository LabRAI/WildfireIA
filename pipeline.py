from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def configure_proj_data() -> None:
    """Point pyproj/rasterio at the conda PROJ database when env vars are unset."""
    if os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"):
        return
    candidates = [
        Path(sys.prefix) / "share/proj",
        Path(sys.prefix) / "lib/python3.10/site-packages/pyproj/proj_dir/share/proj",
        Path(sys.prefix) / "lib/python3.10/site-packages/pyogrio/proj_data",
    ]
    for candidate in candidates:
        if (candidate / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(candidate)
            os.environ["PROJ_LIB"] = str(candidate)
            try:
                import pyproj

                pyproj.datadir.set_data_dir(str(candidate))
            except Exception:
                pass
            break


configure_proj_data()
from pyproj import Transformer
from scipy.spatial import cKDTree


ACRES_TO_HECTARES = 0.40468564224
EPSG_WGS84 = "EPSG:4326"
EPSG_5070 = "EPSG:5070"
CELL_SIZE_M = 375
PATCH_SIZE = 29
CENTER_CELL = 14
PATCH_RADIUS_M = 5000
DAILY_RELATIVE_DAYS = list(range(-4, 1))

FORBIDDEN_AS_FEATURES = [
    "fire_size_acres",
    "fire_size_ha",
    "ia_failure_label",
    "contain_dt",
    "containment_hours",
    "log_containment_hours",
    "log_fire_size_ha",
    "FIRE_SIZE",
    "FIRE_SIZE_CLASS",
    "CONT_DATE",
    "CONT_TIME",
    "MTBS_ID",
    "MTBS_FIRE_NAME",
]

GRIDMET_VARS = [
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
    "th",
]

GRIDMET_OUTPUT_VARS = [
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

DRIVABLE_ROADS = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "track",
    "road",
]

MAJOR_ROADS = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
]

LOCAL_ROADS = [
    "service",
    "track",
    "road",
    "unclassified",
    "residential",
    "living_street",
]

OSM_YEAR_FILES = {
    2016: "north-america-160101.osm.pbf",
    2017: "north-america-170101.osm.pbf",
    2018: "north-america-180101.osm.pbf",
    2019: "north-america-190101.osm.pbf",
    2020: "north-america-200101.osm.pbf",
}


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def out_path(output_dir: Path, name: str) -> Path:
    return output_dir / name


def path_exists(path: Path) -> bool:
    return path.exists()


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def save_parquet(df: pd.DataFrame, path: Path, overwrite: bool = True) -> None:
    if overwrite:
        remove_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(df):,} rows)")


def load_or_build(path: Path, overwrite: bool, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return pd.read_parquet(path)
    return builder()


def write_json(payload: dict, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Reusing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Wrote {path}")


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_arcgis_or_string_date(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    numeric = pd.to_numeric(x, errors="coerce")
    if pd.notna(numeric) and abs(float(numeric)) > 1e11:
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce").tz_convert(None)
    return pd.to_datetime(x, errors="coerce")


def parse_hhmm_time(x) -> tuple[int, int]:
    if pd.isna(x):
        return 0, 0
    text = str(x).strip()
    if not text:
        return 0, 0
    if "." in text:
        text = text.split(".", 1)[0]
    try:
        text = str(int(text)).zfill(4)
    except ValueError:
        return 0, 0
    if len(text) != 4:
        return 0, 0
    hour = int(text[:2])
    minute = int(text[2:])
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return 0, 0


def combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = date_series.map(parse_arcgis_or_string_date)
    times = time_series.map(parse_hhmm_time)
    hours = times.map(lambda value: value[0])
    minutes = times.map(lambda value: value[1])
    return dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def split_from_year(year: int) -> str:
    if 2016 <= year <= 2018:
        return "train"
    if year == 2019:
        return "val"
    if year == 2020:
        return "test"
    return "unknown"


def process_fpa_fod(base_dir: Path, output_dir: Path, start_year: int, end_year: int, overwrite: bool) -> pd.DataFrame:
    path = out_path(output_dir, f"fire_events_natural_{start_year}_{end_year}.parquet")

    def _build() -> pd.DataFrame:
        fpa_dir = base_dir / "fpafod"
        frames = []
        dtype = {
            "FOD_ID": "string",
            "FPA_ID": "string",
            "DISCOVERY_TIME": "string",
            "CONT_TIME": "string",
            "STATE": "string",
            "COUNTY": "string",
            "NWCG_CAUSE_CLASSIFICATION": "string",
            "NWCG_GENERAL_CAUSE": "string",
        }
        for year in range(start_year, end_year + 1):
            csv_path = fpa_dir / f"fpa_fod_conus_{year}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            frames.append(pd.read_csv(csv_path, dtype=dtype, low_memory=False))

        df = pd.concat(frames, ignore_index=True)
        df["FIRE_YEAR"] = pd.to_numeric(df["FIRE_YEAR"], errors="coerce")
        df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
        df["FIRE_SIZE"] = pd.to_numeric(df["FIRE_SIZE"], errors="coerce")

        required = ["FOD_ID", "LATITUDE", "LONGITUDE", "DISCOVERY_DATE", "FIRE_SIZE"]
        keep = df["FIRE_YEAR"].between(start_year, end_year)
        for col in required:
            keep &= df[col].notna()
        keep &= df["FIRE_SIZE"] > 0
        df = df.loc[keep].copy()

        df["NWCG_CAUSE_CLASSIFICATION_CLEAN"] = (
            df["NWCG_CAUSE_CLASSIFICATION"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df = df.loc[df["NWCG_CAUSE_CLASSIFICATION_CLEAN"] == "natural"].copy()

        discovery_dt = combine_date_time(df["DISCOVERY_DATE"], df["DISCOVERY_TIME"])
        contain_dt = combine_date_time(df["CONT_DATE"], df["CONT_TIME"])
        fire_size_acres = df["FIRE_SIZE"]
        fire_size_ha = fire_size_acres * ACRES_TO_HECTARES
        containment_hours = (contain_dt - discovery_dt).dt.total_seconds() / 3600.0
        containment_hours = containment_hours.mask((containment_hours < 0) | (containment_hours > 1440))

        discovery_doy_raw = pd.to_numeric(df.get("DISCOVERY_DOY"), errors="coerce")
        discovery_doy = discovery_doy_raw.fillna(discovery_dt.dt.dayofyear)

        events = pd.DataFrame(
            {
                "fire_id": df["FOD_ID"].astype(str),
                "year": df["FIRE_YEAR"].astype(int),
                "lat": df["LATITUDE"].astype(float),
                "lon": df["LONGITUDE"].astype(float),
                "state": df.get("STATE", pd.Series(pd.NA, index=df.index)).fillna("Unknown").astype(str),
                "county": df.get("COUNTY", pd.Series(pd.NA, index=df.index)).fillna("Unknown").astype(str),
                "discovery_dt": discovery_dt,
                "discovery_date": discovery_dt.dt.strftime("%Y-%m-%d"),
                "discovery_month": discovery_dt.dt.month.astype("Int64"),
                "discovery_doy": discovery_doy.astype("Int64"),
                "discovery_hour": discovery_dt.dt.hour.astype("Int64"),
                "cause_classification": df.get("NWCG_CAUSE_CLASSIFICATION", pd.Series(pd.NA, index=df.index))
                .fillna("Unknown")
                .astype(str),
                "general_cause": df.get("NWCG_GENERAL_CAUSE", pd.Series(pd.NA, index=df.index))
                .fillna("Unknown")
                .astype(str),
                "fire_size_acres": fire_size_acres.astype(float),
                "fire_size_ha": fire_size_ha.astype(float),
                "ia_failure_label": pd.Series(np.nan, index=df.index, dtype="float"),
                "contain_dt": contain_dt,
                "containment_hours": containment_hours.astype(float),
                "log_containment_hours": np.log1p(containment_hours),
                "log_fire_size_ha": np.log1p(fire_size_ha),
            }
        )
        events.loc[events["fire_size_ha"] <= 10, "ia_failure_label"] = 0
        events.loc[events["fire_size_ha"] >= 50, "ia_failure_label"] = 1
        events["split"] = events["year"].map(split_from_year)

        if events["fire_id"].duplicated().any():
            raise ValueError("FPA-FOD Natural fire_id values are not unique.")

        save_parquet(events, path, overwrite=True)
        print_fpa_sanity(events)
        return events

    return load_or_build(path, overwrite, _build)


def print_fpa_sanity(events: pd.DataFrame) -> None:
    print("\nFPA-FOD Natural sanity checks")
    print("Total Natural events:", len(events))
    print("Events per year:")
    print(events["year"].value_counts().sort_index().to_string())
    print("Split counts:")
    print(events["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("cause_classification:")
    print(events["cause_classification"].value_counts(dropna=False).to_string())
    print("general_cause:")
    print(events["general_cause"].value_counts(dropna=False).head(20).to_string())
    print("Task 1 label counts by split:")
    print(
        events.dropna(subset=["ia_failure_label"])
        .groupby("split")["ia_failure_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
        .to_string()
    )
    print("Valid containment_hours count by split:")
    print(
        events.groupby("split")["containment_hours"]
        .apply(lambda s: int(s.notna().sum()))
        .reindex(["train", "val", "test"])
        .to_string()
    )
    print("fire_size_ha summary:")
    print(events["fire_size_ha"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    print("containment_hours summary:")
    print(events["containment_hours"].dropna().describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())


def build_event_grid_375m_index(
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> Path:
    path = out_path(output_dir, f"event_grid_375m_index_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return path
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)

    to_5070 = Transformer.from_crs(EPSG_WGS84, EPSG_5070, always_xy=True)
    to_4326 = Transformer.from_crs(EPSG_5070, EPSG_WGS84, always_xy=True)
    cells = pd.MultiIndex.from_product(
        [range(PATCH_SIZE), range(PATCH_SIZE)], names=["row", "col"]
    ).to_frame(index=False)
    cells["cell_id"] = cells["row"] * PATCH_SIZE + cells["col"]
    cells["dx_m"] = (cells["col"] - CENTER_CELL) * CELL_SIZE_M
    cells["dy_m"] = (CENTER_CELL - cells["row"]) * CELL_SIZE_M
    cells["distance_to_ignition_m"] = np.sqrt(cells["dx_m"] ** 2 + cells["dy_m"] ** 2)

    for year in range(start_year, end_year + 1):
        fires = events.loc[events["year"] == year, ["fire_id", "year", "split", "lon", "lat"]].copy()
        if fires.empty:
            continue
        x0, y0 = to_5070.transform(fires["lon"].to_numpy(), fires["lat"].to_numpy())
        fire_base = fires[["fire_id", "year", "split"]].loc[fires.index.repeat(len(cells))].reset_index(drop=True)
        tiled = pd.concat([cells] * len(fires), ignore_index=True)
        x0_rep = np.repeat(x0, len(cells))
        y0_rep = np.repeat(y0, len(cells))
        x = x0_rep + tiled["dx_m"].to_numpy()
        y = y0_rep + tiled["dy_m"].to_numpy()
        lon, lat = to_4326.transform(x, y)
        grid = pd.concat([fire_base, tiled], axis=1)
        grid["x_5070"] = x
        grid["y_5070"] = y
        grid["lat"] = lat
        grid["lon"] = lon
        grid = grid[
            [
                "fire_id",
                "year",
                "split",
                "row",
                "col",
                "cell_id",
                "x_5070",
                "y_5070",
                "lat",
                "lon",
                "dx_m",
                "dy_m",
                "distance_to_ignition_m",
            ]
        ]
        year_dir = path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(year_dir / "part-0.parquet", index=False)
        print(f"Wrote grid index {year}: {len(grid):,} rows")
    return path


def get_firms_dir(base_dir: Path) -> Path | None:
    firms_dir = base_dir / "firms_combine"
    if not firms_dir.exists():
        firms_dir = base_dir / "combine"
    if not firms_dir.exists():
        print("Warning: no firms_combine or combine directory found.")
        return None
    return firms_dir


def clean_viirs_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    viirs = frame.copy()
    viirs.columns = [c.strip() for c in viirs.columns]
    required = {"latitude", "longitude", "acq_date"}
    if not required <= set(viirs.columns):
        return pd.DataFrame()
    viirs["acq_date"] = pd.to_datetime(viirs["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    viirs = viirs.loc[viirs["acq_date"].notna()].copy()
    if "type" in viirs.columns:
        viirs = viirs.loc[pd.to_numeric(viirs["type"], errors="coerce") == 0].copy()
    for col in ["latitude", "longitude", "frp", "bright_ti4", "bright_ti5"]:
        if col in viirs.columns:
            viirs[col] = pd.to_numeric(viirs[col], errors="coerce")
    viirs = viirs.dropna(subset=["latitude", "longitude"])
    dedup_cols = [
        c
        for c in [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
            "platform",
            "instrument",
            "frp",
            "bright_ti4",
            "bright_ti5",
        ]
        if c in viirs.columns
    ]
    if dedup_cols:
        viirs = viirs.drop_duplicates(subset=dedup_cols)
    viirs["year"] = pd.to_datetime(viirs["acq_date"]).dt.year
    return viirs


def read_viirs_for_date(base_dir: Path, date: str) -> pd.DataFrame:
    firms_dir = get_firms_dir(base_dir)
    if firms_dir is None:
        return pd.DataFrame()
    candidates = [firms_dir / f"{date}.csv"]
    if not candidates[0].exists():
        candidates = sorted(firms_dir.glob(f"{date}*.csv"))
    frames = []
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            frames.append(pd.read_csv(csv_path, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    return clean_viirs_frame(pd.concat(frames, ignore_index=True))


def _project_points(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(EPSG_WGS84, EPSG_5070, always_xy=True)
    return transformer.transform(lon, lat)


def _empty_viirs_event_features(events: pd.DataFrame) -> pd.DataFrame:
    out = events[["fire_id"]].copy()
    cols = {
        "has_viirs_detection_1km_D": 0,
        "viirs_count_500m_D": 0,
        "viirs_count_1km_D": 0,
        "viirs_sum_frp_1km_D": 0.0,
        "viirs_max_frp_1km_D": 0.0,
        "viirs_mean_frp_1km_D": 0.0,
        "viirs_max_bright_ti4_1km_D": 0.0,
        "viirs_mean_bright_ti4_1km_D": 0.0,
        "viirs_day_count_1km_D": 0,
        "viirs_night_count_1km_D": 0,
        "viirs_count_3km_D": 0,
        "viirs_sum_frp_3km_D": 0.0,
        "viirs_max_frp_3km_D": 0.0,
        "viirs_mean_frp_3km_D": 0.0,
        "viirs_max_bright_ti4_3km_D": 0.0,
        "viirs_mean_bright_ti4_3km_D": 0.0,
        "viirs_nearest_detection_distance_m_D": np.nan,
        "viirs_min_assigned_distance_m_D": np.nan,
        "viirs_num_assigned_detections_D": 0,
        "viirs_ambiguous_match_count_D": 0,
    }
    for col, value in cols.items():
        out[col] = value
    return out


def _assign_viirs_same_day_stream(
    events: pd.DataFrame,
    base_dir: Path,
) -> tuple[pd.DataFrame, int, pd.Series]:
    assignments = []
    multi_fire_within_1km = 0
    viirs_counts_by_year = {}
    event_dates = events.groupby("discovery_date")
    for i, (date, fires_day) in enumerate(event_dates, start=1):
        viirs_day = read_viirs_for_date(base_dir, str(date))
        if not viirs_day.empty:
            year = int(pd.to_datetime(date).year)
            viirs_counts_by_year[year] = viirs_counts_by_year.get(year, 0) + len(viirs_day)
        if fires_day.empty:
            continue
        fire_x, fire_y = _project_points(fires_day["lon"].to_numpy(), fires_day["lat"].to_numpy())
        fire_xy = np.column_stack([fire_x, fire_y])
        fire_tree = cKDTree(fire_xy)
        if len(fires_day) > 1:
            multi_fire_within_1km += len(fire_tree.query_pairs(1000))
        if viirs_day.empty:
            continue
        det_x, det_y = _project_points(viirs_day["longitude"].to_numpy(), viirs_day["latitude"].to_numpy())
        det_xy = np.column_stack([det_x, det_y])
        k = min(2, len(fires_day))
        distances, indices = fire_tree.query(det_xy, k=k)
        if k == 1:
            nearest_distance = distances
            nearest_idx = indices
            second_distance = np.full(len(viirs_day), np.inf)
        else:
            nearest_distance = distances[:, 0]
            nearest_idx = indices[:, 0]
            second_distance = distances[:, 1]
        within_3km = nearest_distance <= 3000
        if not np.any(within_3km):
            continue
        assigned = viirs_day.loc[within_3km].copy()
        fire_rows = fires_day.iloc[nearest_idx[within_3km]].reset_index(drop=True)
        assigned["fire_id"] = fire_rows["fire_id"].to_numpy()
        assigned["fire_x_5070"] = fire_x[nearest_idx[within_3km]]
        assigned["fire_y_5070"] = fire_y[nearest_idx[within_3km]]
        assigned["x_5070"] = det_x[within_3km]
        assigned["y_5070"] = det_y[within_3km]
        assigned["assigned_distance_m"] = nearest_distance[within_3km]
        assigned["second_nearest_distance_m"] = second_distance[within_3km]
        assigned["ambiguous_match"] = (
            assigned["second_nearest_distance_m"] - assigned["assigned_distance_m"] < 500
        )
        assignments.append(assigned)
        if i % 250 == 0:
            print(f"Processed VIIRS same-day matching for {i} discovery dates", flush=True)
    counts = pd.Series(viirs_counts_by_year, dtype="int64").sort_index()
    if assignments:
        return pd.concat(assignments, ignore_index=True), multi_fire_within_1km, counts
    return pd.DataFrame(), multi_fire_within_1km, counts


def process_viirs_event_features(
    base_dir: Path,
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"viirs_features_natural_{start_year}_{end_year}.parquet")

    def _build() -> pd.DataFrame:
        out = _empty_viirs_event_features(events)
        assignments, multi_fire_within_1km, viirs_counts = _assign_viirs_same_day_stream(events, base_dir)
        if viirs_counts.empty:
            print("Warning: no VIIRS detections available. Writing zero VIIRS event features.")
            save_parquet(out, path, overwrite=True)
            return out
        if assignments.empty:
            save_parquet(out, path, overwrite=True)
            return out

        grouped = assignments.groupby("fire_id")
        out = out.set_index("fire_id")
        for radius in [500, 1000, 3000]:
            suffix = f"{int(radius / 1000)}km" if radius >= 1000 else "500m"
            within = assignments.loc[assignments["assigned_distance_m"] <= radius].copy()
            g = within.groupby("fire_id")
            count = g.size()
            if radius == 500:
                out["viirs_count_500m_D"] = count
                continue
            out[f"viirs_count_{suffix}_D"] = count
            out[f"viirs_sum_frp_{suffix}_D"] = g["frp"].sum(min_count=1)
            out[f"viirs_max_frp_{suffix}_D"] = g["frp"].max()
            out[f"viirs_mean_frp_{suffix}_D"] = g["frp"].mean()
            out[f"viirs_max_bright_ti4_{suffix}_D"] = g["bright_ti4"].max()
            out[f"viirs_mean_bright_ti4_{suffix}_D"] = g["bright_ti4"].mean()
            if radius == 1000 and "daynight" in within.columns:
                out["viirs_day_count_1km_D"] = within.loc[within["daynight"] == "D"].groupby("fire_id").size()
                out["viirs_night_count_1km_D"] = within.loc[within["daynight"] == "N"].groupby("fire_id").size()

        out["has_viirs_detection_1km_D"] = (out["viirs_count_1km_D"].fillna(0) > 0).astype(int)
        out["viirs_nearest_detection_distance_m_D"] = grouped["assigned_distance_m"].min()
        out["viirs_min_assigned_distance_m_D"] = grouped["assigned_distance_m"].min()
        out["viirs_num_assigned_detections_D"] = grouped.size()
        out["viirs_ambiguous_match_count_D"] = grouped["ambiguous_match"].sum()
        out = out.reset_index()

        count_cols = [c for c in out.columns if "count" in c or c.startswith("has_") or c.startswith("viirs_num")]
        sum_cols = [c for c in out.columns if "_sum_" in c]
        stat_cols = [c for c in out.columns if "_max_" in c or "_mean_" in c]
        out[count_cols] = out[count_cols].fillna(0)
        out[sum_cols + stat_cols] = out[sum_cols + stat_cols].fillna(0.0)
        for c in count_cols:
            out[c] = out[c].astype(int)

        save_parquet(out, path, overwrite=True)
        print_viirs_sanity(events, viirs_counts, out, assignments, multi_fire_within_1km)
        return out

    return load_or_build(path, overwrite, _build)


def print_viirs_sanity(
    events: pd.DataFrame,
    viirs_counts: pd.Series,
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    multi_fire_within_1km: int,
) -> None:
    print("\nVIIRS sanity checks")
    print("Natural FPA-FOD events per year:")
    print(events["year"].value_counts().sort_index().to_string())
    print("VIIRS detections per year after type filtering on discovery dates:")
    print(viirs_counts.to_string())
    for radius in ["500m", "1km", "3km"]:
        col = f"viirs_count_{radius}_D"
        pct = (features[col] > 0).mean() * 100 if col in features else 0.0
        print(f"Events with VIIRS within {radius}: {pct:.2f}%")
    if "ia_failure_label" in events.columns:
        labeled = features.merge(events[["fire_id", "ia_failure_label", "fire_size_ha"]], on="fire_id", how="left")
        print("VIIRS 1km match rate by ia_failure_label:")
        print(labeled.dropna(subset=["ia_failure_label"]).groupby("ia_failure_label")["has_viirs_detection_1km_D"].mean().to_string())
        bins = pd.cut(
            labeled["fire_size_ha"],
            bins=[0, 1, 10, 50, 300, np.inf],
            labels=["0-1ha", "1-10ha", "10-50ha", "50-300ha", "300ha+"],
            include_lowest=True,
        )
        print("VIIRS 1km match rate by fire_size_ha bin:")
        print(labeled.groupby(bins, observed=False)["has_viirs_detection_1km_D"].mean().to_string())
    print("Nearest assigned VIIRS distance summary:")
    print(features["viirs_nearest_detection_distance_m_D"].dropna().describe().to_string())
    print("Ambiguous same-day matches:", int(assignments["ambiguous_match"].sum()) if not assignments.empty else 0)
    print("Same-day FPA-FOD fire pairs within 1km:", multi_fire_within_1km)


def process_viirs_patch_375m_D(
    base_dir: Path,
    events: pd.DataFrame,
    grid_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> Path:
    path = out_path(output_dir, f"event_viirs_patch_375m_D_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return path
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)
    assignments, _, _ = _assign_viirs_same_day_stream(events, base_dir)

    if not assignments.empty:
        dx = assignments["x_5070"].to_numpy() - assignments["fire_x_5070"].to_numpy()
        dy = assignments["y_5070"].to_numpy() - assignments["fire_y_5070"].to_numpy()
        assignments["col"] = np.floor((dx + (CENTER_CELL + 0.5) * CELL_SIZE_M) / CELL_SIZE_M).astype(int)
        assignments["row"] = np.floor(((CENTER_CELL + 0.5) * CELL_SIZE_M - dy) / CELL_SIZE_M).astype(int)
        assignments = assignments.loc[
            assignments["row"].between(0, PATCH_SIZE - 1) & assignments["col"].between(0, PATCH_SIZE - 1)
        ].copy()
        assignments["cell_id"] = assignments["row"] * PATCH_SIZE + assignments["col"]

    for year in range(start_year, end_year + 1):
        grid = pd.read_parquet(grid_path / f"year={year}")
        base = grid[["fire_id", "year", "split", "row", "col", "cell_id"]].copy()
        cols = {
            "viirs_cell_count_D": 0,
            "viirs_cell_sum_frp_D": 0.0,
            "viirs_cell_max_frp_D": 0.0,
            "viirs_cell_mean_frp_D": 0.0,
            "viirs_cell_max_bright_ti4_D": 0.0,
            "viirs_cell_mean_bright_ti4_D": 0.0,
            "viirs_cell_day_count_D": 0,
            "viirs_cell_night_count_D": 0,
            "viirs_cell_has_detection_D": 0,
        }
        for col, val in cols.items():
            base[col] = val
        if not assignments.empty:
            assn_year = assignments.loc[assignments["fire_id"].isin(base["fire_id"].unique())].copy()
            if not assn_year.empty:
                keys = ["fire_id", "row", "col", "cell_id"]
                grouped = assn_year.groupby(keys)
                agg = grouped.agg(
                    viirs_cell_count_D=("frp", "size"),
                    viirs_cell_sum_frp_D=("frp", "sum"),
                    viirs_cell_max_frp_D=("frp", "max"),
                    viirs_cell_mean_frp_D=("frp", "mean"),
                    viirs_cell_max_bright_ti4_D=("bright_ti4", "max"),
                    viirs_cell_mean_bright_ti4_D=("bright_ti4", "mean"),
                ).reset_index()
                if "daynight" in assn_year.columns:
                    day = assn_year.loc[assn_year["daynight"] == "D"].groupby(keys).size().rename("viirs_cell_day_count_D")
                    night = assn_year.loc[assn_year["daynight"] == "N"].groupby(keys).size().rename("viirs_cell_night_count_D")
                    agg = agg.merge(day.reset_index(), on=keys, how="left").merge(night.reset_index(), on=keys, how="left")
                agg["viirs_cell_has_detection_D"] = 1
                base = base[keys + ["year", "split"]].merge(agg, on=keys, how="left")
                for col, val in cols.items():
                    if col not in base.columns:
                        base[col] = val
                    else:
                        base[col] = base[col].fillna(val)
        year_dir = path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        base.to_parquet(year_dir / "part-0.parquet", index=False)
        print(f"Wrote VIIRS patch {year}: {len(base):,} rows")
    return path


def gridmet_file(base_dir: Path, var: str, year: int) -> Path:
    return base_dir / "gridmet" / var / f"{var}_{year}.nc"


def _sample_gridmet_var(base_dir: Path, var: str, sample: pd.DataFrame) -> np.ndarray:
    from netCDF4 import Dataset, num2date

    values = np.full(len(sample), np.nan, dtype=float)
    dates = pd.to_datetime(sample["date"])
    for year in sorted(dates.dt.year.dropna().unique()):
        nc_path = gridmet_file(base_dir, var, int(year))
        if not nc_path.exists():
            continue
        idx = np.where(dates.dt.year.to_numpy() == year)[0]
        if len(idx) == 0:
            continue
        ds = Dataset(nc_path)
        try:
            data_var = [name for name in ds.variables if name not in {"lat", "lon", "day", "time", "crs"}][0]
            lat = np.asarray(ds.variables["lat"][:])
            lon = np.asarray(ds.variables["lon"][:])
            time_name = "day" if "day" in ds.variables else "time"
            time_var = ds.variables[time_name]
            nc_dates = pd.to_datetime(
                num2date(
                    time_var[:],
                    time_var.units,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
            ).strftime("%Y-%m-%d")
            day_lookup = {date: pos for pos, date in enumerate(nc_dates)}
            sample_dates = dates.iloc[idx].dt.strftime("%Y-%m-%d").to_numpy()
            valid_day = np.array([date in day_lookup for date in sample_dates])
            if not valid_day.any():
                continue

            idx_valid = idx[valid_day]
            day_idx = np.array([day_lookup[date] for date in sample_dates[valid_day]], dtype=int)

            lat_values = sample["lat"].iloc[idx_valid].to_numpy()
            lon_values = sample["lon"].iloc[idx_valid].to_numpy()
            if lat[0] > lat[-1]:
                lat_asc = lat[::-1]
                lat_pos = np.searchsorted(lat_asc, lat_values)
                lat_pos = np.clip(lat_pos, 1, len(lat_asc) - 1)
                left = lat_asc[lat_pos - 1]
                right = lat_asc[lat_pos]
                lat_idx_asc = np.where(np.abs(lat_values - left) <= np.abs(lat_values - right), lat_pos - 1, lat_pos)
                lat_idx = len(lat) - 1 - lat_idx_asc
            else:
                lat_pos = np.searchsorted(lat, lat_values)
                lat_pos = np.clip(lat_pos, 1, len(lat) - 1)
                left = lat[lat_pos - 1]
                right = lat[lat_pos]
                lat_idx = np.where(np.abs(lat_values - left) <= np.abs(lat_values - right), lat_pos - 1, lat_pos)

            lon_pos = np.searchsorted(lon, lon_values)
            lon_pos = np.clip(lon_pos, 1, len(lon) - 1)
            left = lon[lon_pos - 1]
            right = lon[lon_pos]
            lon_idx = np.where(np.abs(lon_values - left) <= np.abs(lon_values - right), lon_pos - 1, lon_pos)

            data = ds.variables[data_var][:]
            fill_value = getattr(ds.variables[data_var], "_FillValue", None)
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            data = np.asarray(data, dtype=float)
            sampled = data[day_idx, lat_idx, lon_idx]
            if fill_value is not None:
                sampled = np.where(sampled == fill_value, np.nan, sampled)
            sampled = np.where(np.abs(sampled) >= 32767, np.nan, sampled)
            values[idx_valid] = sampled
        finally:
            ds.close()
    return values


def process_gridmet_daily_event_features(
    base_dir: Path,
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"gridmet_daily_event_features_natural_{start_year}_{end_year}.parquet")

    def _build() -> pd.DataFrame:
        rows = []
        for rel in DAILY_RELATIVE_DAYS:
            frame = events[["fire_id", "year", "split", "discovery_date", "lat", "lon"]].copy()
            frame["relative_day"] = rel
            frame["date"] = (pd.to_datetime(frame["discovery_date"]) + pd.to_timedelta(rel, unit="D")).dt.strftime("%Y-%m-%d")
            rows.append(frame)
        daily = pd.concat(rows, ignore_index=True)
        for var in GRIDMET_VARS:
            try:
                print(f"Sampling gridMET variable: {var}", flush=True)
                daily[var] = _sample_gridmet_var(base_dir, var, daily)
                print(f"Finished gridMET variable: {var}", flush=True)
            except Exception as exc:
                print(f"Warning: gridMET sampling failed for {var}: {exc}")
                daily[var] = np.nan
        if "th" in daily.columns:
            radians = np.deg2rad(daily["th"])
            daily["wind_dir_sin"] = np.sin(radians)
            daily["wind_dir_cos"] = np.cos(radians)
            daily = daily.drop(columns=["th"])
        daily = daily[["fire_id", "date", "relative_day"] + GRIDMET_OUTPUT_VARS]
        save_parquet(daily, path, overwrite=True)
        return daily

    return load_or_build(path, overwrite, _build)


def process_gridmet_aggregate_event_features(
    daily: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"gridmet_features_natural_{start_year}_{end_year}.parquet")

    def _build() -> pd.DataFrame:
        features = daily[["fire_id"]].drop_duplicates().copy()
        for var in GRIDMET_OUTPUT_VARS:
            if var not in daily.columns:
                continue
            day0 = daily.loc[daily["relative_day"] == 0].set_index("fire_id")[var]
            lag1 = daily.loc[daily["relative_day"] == -1].set_index("fire_id")[var]
            features[f"{var}_day0"] = features["fire_id"].map(day0)
            if var != "pr":
                features[f"{var}_lag1"] = features["fire_id"].map(lag1)
        for var in GRIDMET_OUTPUT_VARS:
            for window in [2, 3, 4, 5]:
                rel_days = list(range(-(window - 1), 1))
                mask = daily["relative_day"].isin(rel_days)
                if var == "pr":
                    agg = daily.loc[mask].groupby("fire_id")[var].sum(min_count=1)
                    col = f"pr_sum{window}"
                else:
                    agg = daily.loc[mask].groupby("fire_id")[var].mean()
                    col = f"{var}_mean{window}"
                features[col] = features["fire_id"].map(agg)
        save_parquet(features, path, overwrite=True)
        return features

    return load_or_build(path, overwrite, _build)


def process_gridmet_daily_patch_375m(
    grid_path: Path,
    daily_event: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> Path:
    path = out_path(output_dir, f"event_weather_daily_patch_375m_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return path
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        grid = pd.read_parquet(grid_path / f"year={year}")
        for rel in DAILY_RELATIVE_DAYS:
            daily_rel = daily_event.loc[daily_event["relative_day"] == rel].copy()
            merged = grid[["fire_id", "year", "split", "row", "col", "cell_id"]].merge(
                daily_rel.drop(columns=["relative_day"]), on="fire_id", how="left"
            )
            merged["relative_day"] = rel
            part_dir = path / f"year={year}" / f"relative_day={rel}"
            part_dir.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(part_dir / "part-0.parquet", index=False)
            print(f"Wrote weather daily patch year={year} relative_day={rel}: {len(merged):,} rows")
    return path


def process_gridmet_aggregate_patch_375m(
    grid_path: Path,
    event_weather: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> Path:
    path = out_path(output_dir, f"event_weather_aggregate_patch_375m_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return path
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        grid = pd.read_parquet(grid_path / f"year={year}")
        merged = grid[["fire_id", "year", "split", "row", "col", "cell_id"]].merge(
            event_weather, on="fire_id", how="left"
        )
        year_dir = path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(year_dir / "part-0.parquet", index=False)
        print(f"Wrote weather aggregate patch {year}: {len(merged):,} rows")
    return path


def _placeholder_event_table(events: pd.DataFrame, columns: dict[str, float | int], path: Path, overwrite: bool) -> pd.DataFrame:
    df = events[["fire_id"]].copy()
    for col, value in columns.items():
        df[col] = value
    save_parquet(df, path, overwrite=True)
    return df


def landfire_static_raster_paths(base_dir: Path) -> dict[str, Path]:
    return {
        # Surface fuel, canopy fuel, fuel disturbance, and fuel vegetation use LF2016
        # for the no-future-information main benchmark.
        "fbfm40": base_dir / "landfire/fuel/sc/Tif/LF2016_FBFM40_CONUS.tif",
        "cbd": base_dir / "landfire/fuel/cf/cbd/LF2016_CBD_CONUS/Tif/LF2016_CBD_CONUS.tif",
        "cbh": base_dir / "landfire/fuel/cf/cbh/LF2016_CBH_CONUS/Tif/LF2016_CBH_CONUS.tif",
        "cc": base_dir / "landfire/fuel/cf/cc/LF2016_CC_CONUS/Tif/LF2016_CC_CONUS.tif",
        "ch": base_dir / "landfire/fuel/cf/ch/LF2016_CH_CONUS/Tif/LF2016_CH_CONUS.tif",
        "fd": base_dir / "landfire/fuel/fd/LF2016_FDist_CONUS/Tif/LF2016_FDist_CONUS.tif",
        "fvt": base_dir / "landfire/fuel/fv/fvt/LF2016_FVT_CONUS/Tif/LF2016_FVT_CONUS.tif",
        "fvc": base_dir / "landfire/fuel/fv/fvc/LF2016_FVC_CONUS/Tif/LF2016_FVC_CONUS.tif",
        "fvh": base_dir / "landfire/fuel/fv/fvh/LF2016_FVH_CONUS/Tif/LF2016_FVH_CONUS.tif",
        # Existing vegetation also uses LF2016 for the main benchmark.
        "evt": base_dir / "landfire/vegetation/evt/LF2016_EVT_CONUS/Tif/LF2016_EVT_CONUS.tif",
        "evc": base_dir / "landfire/vegetation/evc/LF2016_EVC_CONUS/Tif/LF2016_EVC_CONUS.tif",
        "evh": base_dir / "landfire/vegetation/evh/LF2016_EVH_CONUS/Tif/LF2016_EVH_CONUS.tif",
        "elev": base_dir / "landfire/topography/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif",
        "slope": base_dir / "landfire/topography/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif",
        "aspect": base_dir / "landfire/topography/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif",
    }


def sample_raster_points(
    raster_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    coords_crs: str,
    chunk_size: int = 250_000,
) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    values = np.full(len(x), np.nan, dtype="float64")
    if not raster_path.exists():
        print(f"Warning: missing raster source {raster_path}")
        return values
    with rasterio.open(raster_path) as src:
        xs = np.asarray(x)
        ys = np.asarray(y)
        if src.crs and str(src.crs) != coords_crs:
            transformer = Transformer.from_crs(coords_crs, src.crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows, dtype="int64")
        cols = np.asarray(cols, dtype="int64")
        valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if not valid.any():
            return values

        block_h, block_w = src.block_shapes[0]
        n_block_cols = math.ceil(src.width / block_w)
        valid_idx = np.flatnonzero(valid)
        valid_rows = rows[valid_idx]
        valid_cols = cols[valid_idx]
        block_rows = valid_rows // block_h
        block_cols = valid_cols // block_w
        block_ids = block_rows * n_block_cols + block_cols
        order = np.argsort(block_ids)
        sorted_idx = valid_idx[order]
        sorted_rows = valid_rows[order]
        sorted_cols = valid_cols[order]
        sorted_block_rows = block_rows[order]
        sorted_block_cols = block_cols[order]
        sorted_block_ids = block_ids[order]
        nodata = src.nodata

        start = 0
        while start < len(sorted_idx):
            stop = start + 1
            block_id = sorted_block_ids[start]
            while stop < len(sorted_idx) and sorted_block_ids[stop] == block_id:
                stop += 1

            block_row = int(sorted_block_rows[start])
            block_col = int(sorted_block_cols[start])
            row_off = block_row * block_h
            col_off = block_col * block_w
            height = min(block_h, src.height - row_off)
            width = min(block_w, src.width - col_off)
            tile = src.read(1, window=Window(col_off, row_off, width, height))
            local_rows = sorted_rows[start:stop] - row_off
            local_cols = sorted_cols[start:stop] - col_off
            values[sorted_idx[start:stop]] = tile[local_rows, local_cols].astype("float64", copy=False)
            start = stop

        if nodata is not None:
            values = np.where(values == nodata, np.nan, values)
        values = np.where(np.abs(values) >= 32767, np.nan, values)
    return values


LANDFIRE_FUEL_PATCH_COLUMNS = ["fbfm40", "cbd", "cbh", "cc", "ch", "fd", "fvt", "fvc", "fvh"]
LANDFIRE_VEGETATION_PATCH_COLUMNS = ["evt", "evc", "evh"]
LANDFIRE_CATEGORICAL_PATCH_COLUMNS = ["fbfm40", "fd", "fvt", "fvc", "fvh", "evt", "evc", "evh"]
LANDFIRE_CONTINUOUS_PATCH_COLUMNS = ["cbd", "cbh", "cc", "ch"]


def mode_series(values: pd.Series):
    valid = values.dropna()
    if valid.empty:
        return np.nan
    counts = valid.value_counts()
    return counts.index[0]


def aggregate_mode_from_patch(patch: pd.DataFrame, value_col: str, radius_m: int) -> pd.Series:
    subset = patch.loc[patch["distance_to_ignition_m"] <= radius_m, ["fire_id", value_col]]
    return subset.groupby("fire_id")[value_col].agg(mode_series)


def aggregate_mean_from_patch(patch: pd.DataFrame, value_col: str, radius_m: int) -> pd.Series:
    subset = patch.loc[patch["distance_to_ignition_m"] <= radius_m, ["fire_id", value_col]]
    return subset.groupby("fire_id")[value_col].mean()


def aggregate_max_from_patch(patch: pd.DataFrame, value_col: str, radius_m: int) -> pd.Series:
    subset = patch.loc[patch["distance_to_ignition_m"] <= radius_m, ["fire_id", value_col]]
    return subset.groupby("fire_id")[value_col].max()


def read_static_patch_all(output_dir: Path, start_year: int, end_year: int, columns: list[str] | None = None) -> pd.DataFrame:
    path = out_path(output_dir, f"event_static_patch_375m_natural_{start_year}_{end_year}.parquet")
    frames = []
    for year in range(start_year, end_year + 1):
        year_path = path / f"year={year}"
        if not year_path.exists():
            continue
        frames.append(pd.read_parquet(year_path, columns=columns))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def process_landfire_event_features(
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"landfire_fuel_veg_features_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        return pd.read_parquet(path)
    patch = read_static_patch_all(
        output_dir,
        start_year,
        end_year,
        columns=[
            "fire_id",
            "row",
            "col",
            "distance_to_ignition_m",
            *LANDFIRE_FUEL_PATCH_COLUMNS,
            *LANDFIRE_VEGETATION_PATCH_COLUMNS,
        ],
    )
    if patch.empty:
        raise FileNotFoundError("Static patch is required before LANDfire event aggregation.")
    out = events[["fire_id"]].copy()
    center = patch.loc[(patch["row"] == CENTER_CELL) & (patch["col"] == CENTER_CELL)].set_index("fire_id")
    for col in [*LANDFIRE_FUEL_PATCH_COLUMNS, *LANDFIRE_VEGETATION_PATCH_COLUMNS]:
        out[f"{col}_point"] = out["fire_id"].map(center[col])
    for radius_m, suffix in [(1000, "1km"), (3000, "3km"), (5000, "5km")]:
        for col in LANDFIRE_CATEGORICAL_PATCH_COLUMNS:
            out[f"{col}_mode_{suffix}"] = out["fire_id"].map(aggregate_mode_from_patch(patch, col, radius_m))
        for col in LANDFIRE_CONTINUOUS_PATCH_COLUMNS:
            out[f"{col}_mean_{suffix}"] = out["fire_id"].map(aggregate_mean_from_patch(patch, col, radius_m))
    for col in LANDFIRE_CONTINUOUS_PATCH_COLUMNS:
        out[f"{col}_max_5km"] = out["fire_id"].map(aggregate_max_from_patch(patch, col, 5000))
    save_parquet(out, path, overwrite=True)
    return out


def process_topography_event_features(
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"topography_features_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        return pd.read_parquet(path)
    patch = read_static_patch_all(
        output_dir,
        start_year,
        end_year,
        columns=[
            "fire_id",
            "row",
            "col",
            "distance_to_ignition_m",
            "elev",
            "slope",
            "aspect_sin",
            "aspect_cos",
        ],
    )
    if patch.empty:
        raise FileNotFoundError("Static patch is required before topography event aggregation.")
    out = events[["fire_id"]].copy()
    center = patch.loc[(patch["row"] == CENTER_CELL) & (patch["col"] == CENTER_CELL)].set_index("fire_id")
    out["elev_point"] = out["fire_id"].map(center["elev"])
    out["slope_point"] = out["fire_id"].map(center["slope"])
    out["aspect_sin_point"] = out["fire_id"].map(center["aspect_sin"])
    out["aspect_cos_point"] = out["fire_id"].map(center["aspect_cos"])
    for radius_m, suffix in [(1000, "1km"), (3000, "3km"), (5000, "5km")]:
        out[f"elev_mean_{suffix}"] = out["fire_id"].map(aggregate_mean_from_patch(patch, "elev", radius_m))
        out[f"slope_mean_{suffix}"] = out["fire_id"].map(aggregate_mean_from_patch(patch, "slope", radius_m))
    out["slope_max_5km"] = out["fire_id"].map(aggregate_max_from_patch(patch, "slope", 5000))
    save_parquet(out, path, overwrite=True)
    return out


def find_worldpop_raster(base_dir: Path, source_year: int) -> Path | None:
    population_dir = base_dir / "population"
    exact = population_dir / f"usa_pop_{source_year}.tif"
    if exact.exists():
        return exact
    candidates = sorted(population_dir.glob(f"*{source_year}*.tif"))
    return candidates[0] if candidates else None


def worldpop_raster_map(base_dir: Path, start_year: int, end_year: int) -> dict[int, Path | None]:
    source_years = range(start_year - 1, end_year)
    return {source_year: find_worldpop_raster(base_dir, source_year) for source_year in source_years}


def warn_missing_worldpop_rasters(rasters: dict[int, Path | None]) -> None:
    missing = [year for year, path in rasters.items() if path is None]
    if missing:
        print(
            "Warning: missing required prior-year WorldPop raster(s): "
            f"{missing}. Population numeric features for fires using these "
            "source years will remain NaN. No future raster fallback will be used."
        )


def _sample_raster_points_optional(raster_path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    try:
        import rasterio
    except Exception as exc:
        raise RuntimeError(f"rasterio is required for WorldPop raster sampling: {exc}") from exc

    with rasterio.open(raster_path) as src:
        xs = lon
        ys = lat
        if src.crs and str(src.crs) not in {"EPSG:4326", "OGC:CRS84"}:
            transformer = Transformer.from_crs(EPSG_WGS84, src.crs, always_xy=True)
            xs, ys = transformer.transform(lon, lat)
        values = np.array([value[0] for value in src.sample(zip(xs, ys))], dtype=float)
        if src.nodata is not None:
            values = np.where(values == src.nodata, np.nan, values)
        return values


def process_population_event_features(
    base_dir: Path,
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"population_features_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        return pd.read_parquet(path)
    patch = read_static_patch_all(
        output_dir,
        start_year,
        end_year,
        columns=["fire_id", "row", "col", "distance_to_ignition_m", "population_source_year", "pop_density"],
    )
    if patch.empty:
        raise FileNotFoundError("Static patch is required before population event aggregation.")
    df = events[["fire_id"]].copy()
    center = patch.loc[(patch["row"] == CENTER_CELL) & (patch["col"] == CENTER_CELL)].set_index("fire_id")
    df["population_source"] = "WorldPop"
    df["population_source_year"] = df["fire_id"].map(center["population_source_year"])
    df["pop_density_point"] = df["fire_id"].map(center["pop_density"])
    for radius_m, suffix in [(1000, "1km"), (3000, "3km"), (5000, "5km")]:
        df[f"pop_density_mean_{suffix}"] = df["fire_id"].map(aggregate_mean_from_patch(patch, "pop_density", radius_m))
    df["pop_density_max_5km"] = df["fire_id"].map(aggregate_max_from_patch(patch, "pop_density", 5000))
    save_parquet(df, path, overwrite=True)
    return df


def process_static_patch_375m(
    base_dir: Path,
    grid_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> Path:
    path = out_path(output_dir, f"event_static_patch_375m_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return path
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)
    static_rasters = landfire_static_raster_paths(base_dir)
    missing_static = [str(p) for p in static_rasters.values() if not p.exists()]
    if missing_static:
        print("Warning: missing static raster source(s):")
        for item in missing_static:
            print(f"  {item}")
    pop_rasters = worldpop_raster_map(base_dir, start_year, end_year)
    warn_missing_worldpop_rasters(pop_rasters)

    for year in range(start_year, end_year + 1):
        grid = pd.read_parquet(grid_path / f"year={year}")
        patch = grid[
            [
                "fire_id",
                "year",
                "split",
                "row",
                "col",
                "cell_id",
                "x_5070",
                "y_5070",
                "lat",
                "lon",
                "distance_to_ignition_m",
            ]
        ].copy()
        print(f"Sampling static patch rasters for {year}: {len(patch):,} cells", flush=True)
        for raster_name in [*LANDFIRE_FUEL_PATCH_COLUMNS, *LANDFIRE_VEGETATION_PATCH_COLUMNS]:
            print(f"  {year}: sampling {raster_name}", flush=True)
            patch[raster_name] = sample_raster_points(
                static_rasters[raster_name],
                patch["x_5070"].to_numpy(),
                patch["y_5070"].to_numpy(),
                EPSG_5070,
            )
        print(f"  {year}: sampling elev", flush=True)
        patch["elev"] = sample_raster_points(static_rasters["elev"], patch["x_5070"].to_numpy(), patch["y_5070"].to_numpy(), EPSG_5070)
        print(f"  {year}: sampling slope", flush=True)
        patch["slope"] = sample_raster_points(static_rasters["slope"], patch["x_5070"].to_numpy(), patch["y_5070"].to_numpy(), EPSG_5070)
        print(f"  {year}: sampling aspect", flush=True)
        aspect = sample_raster_points(static_rasters["aspect"], patch["x_5070"].to_numpy(), patch["y_5070"].to_numpy(), EPSG_5070)
        aspect_rad = np.deg2rad(aspect)
        patch["aspect_sin"] = np.sin(aspect_rad)
        patch["aspect_cos"] = np.cos(aspect_rad)
        patch["population_source_year"] = year - 1
        pop_path = pop_rasters.get(year - 1)
        if pop_path is None:
            patch["pop_density"] = np.nan
        else:
            print(f"  {year}: sampling WorldPop {year - 1}", flush=True)
            patch["pop_density"] = sample_raster_points(pop_path, patch["lon"].to_numpy(), patch["lat"].to_numpy(), EPSG_WGS84)
        patch = patch[
            [
                "fire_id",
                "year",
                "split",
                "row",
                "col",
                "cell_id",
                "distance_to_ignition_m",
                *LANDFIRE_FUEL_PATCH_COLUMNS,
                *LANDFIRE_VEGETATION_PATCH_COLUMNS,
                "elev",
                "slope",
                "aspect_sin",
                "aspect_cos",
                "population_source_year",
                "pop_density",
            ]
        ]
        year_dir = path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        patch.to_parquet(year_dir / "part-0.parquet", index=False)
        print(f"Wrote static patch {year}: {len(patch):,} rows")
    return path


def build_or_load_osm_layers(base_dir: Path, events: pd.DataFrame, year: int, overwrite: bool):
    import geopandas as gpd

    cache_dir = base_dir / "data/cache/osm" / str(year)
    cache_dir.mkdir(parents=True, exist_ok=True)
    roads_path = cache_dir / "roads.parquet"
    stations_path = cache_dir / "fire_stations.parquet"
    if roads_path.exists() and stations_path.exists() and not overwrite:
        roads = gpd.read_parquet(roads_path)
        stations = gpd.read_parquet(stations_path)
        return roads, stations

    pbf_name = OSM_YEAR_FILES.get(year)
    pbf_path = base_dir / "osm" / pbf_name if pbf_name else None
    if pbf_path is None or not pbf_path.exists():
        print(f"Warning: missing OSM PBF for {year}: {pbf_path}")
        return gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84), gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84)

    year_events = events.loc[events["year"] == year]
    bbox = [
        float(year_events["lon"].min() - 1.0),
        float(year_events["lat"].min() - 1.0),
        float(year_events["lon"].max() + 1.0),
        float(year_events["lat"].max() + 1.0),
    ]
    ogr2ogr = shutil.which("ogr2ogr")
    if ogr2ogr:
        roads_raw = cache_dir / "roads_raw.gpkg"
        station_points_raw = cache_dir / "fire_stations_points_raw.gpkg"
        station_polygons_raw = cache_dir / "fire_stations_polygons_raw.gpkg"

        highway_where = "highway IN ({})".format(",".join(f"'{value}'" for value in DRIVABLE_ROADS))
        spat_args = [str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]
        if roads_raw.exists():
            print(f"Reusing raw OSM roads extract for {year}: {roads_raw}", flush=True)
        else:
            print(f"Extracting OSM roads for {year} with ogr2ogr bbox={bbox}", flush=True)
            run_cmd(
                [
                    ogr2ogr,
                    "--config",
                    "OGR_INTERLEAVED_READING",
                    "YES",
                    "-progress",
                    "-f",
                    "GPKG",
                    str(roads_raw),
                    str(pbf_path),
                    "lines",
                    "-nln",
                    "roads",
                    "-spat",
                    *spat_args,
                    "-where",
                    highway_where,
                    "-select",
                    "highway",
                    "-overwrite",
                ]
            )
        if station_points_raw.exists():
            print(f"Reusing raw OSM fire station point extract for {year}: {station_points_raw}", flush=True)
        else:
            print(f"Extracting OSM fire station points for {year} with ogr2ogr", flush=True)
            run_cmd(
                [
                    ogr2ogr,
                    "--config",
                    "OGR_INTERLEAVED_READING",
                    "YES",
                    "-progress",
                    "-f",
                    "GPKG",
                    str(station_points_raw),
                    str(pbf_path),
                    "points",
                    "-nln",
                    "points",
                    "-spat",
                    *spat_args,
                    "-where",
                    'other_tags LIKE \'%"amenity"=>"fire_station"%\'',
                    "-overwrite",
                ]
            )
        if station_polygons_raw.exists():
            print(f"Reusing raw OSM fire station polygon extract for {year}: {station_polygons_raw}", flush=True)
        else:
            print(f"Extracting OSM fire station polygons for {year} with ogr2ogr", flush=True)
            run_cmd(
                [
                    ogr2ogr,
                    "--config",
                    "OGR_INTERLEAVED_READING",
                    "YES",
                    "--config",
                    "OGR_GEOMETRY_ACCEPT_UNCLOSED_RING",
                    "NO",
                    "-progress",
                    "-f",
                    "GPKG",
                    str(station_polygons_raw),
                    str(pbf_path),
                    "multipolygons",
                    "-nln",
                    "polygons",
                    "-spat",
                    *spat_args,
                    "-where",
                    'amenity = \'fire_station\' OR other_tags LIKE \'%"amenity"=>"fire_station"%\'',
                    "-overwrite",
                ]
            )

        def _read_layer(path: Path, layer: str, columns: list[str] | None = None):
            if not path.exists():
                return gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84)
            try:
                frame = gpd.read_file(path, layer=layer, engine="pyogrio")
            except Exception:
                return gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84)
            if frame.empty:
                return gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84)
            if columns:
                keep = [col for col in columns if col in frame.columns] + ["geometry"]
                frame = frame[keep].copy()
            frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
            return frame.set_crs(EPSG_WGS84, allow_override=True) if frame.crs is None else frame.to_crs(EPSG_WGS84)

        roads = _read_layer(roads_raw, "roads", ["highway"])
        if "highway" not in roads.columns:
            roads["highway"] = None
        roads = roads.loc[roads["highway"].isin(DRIVABLE_ROADS), ["highway", "geometry"]].copy()

        station_points = _read_layer(station_points_raw, "points")
        station_polygons = _read_layer(station_polygons_raw, "polygons")
        station_frames = []
        if not station_points.empty:
            station_frames.append(
                gpd.GeoDataFrame(
                    {"amenity": ["fire_station"] * len(station_points)},
                    geometry=station_points.geometry,
                    crs=EPSG_WGS84,
                )
            )
        if not station_polygons.empty:
            projected = station_polygons.to_crs(EPSG_5070)
            projected["geometry"] = projected.geometry.centroid
            station_frames.append(
                gpd.GeoDataFrame(
                    {"amenity": ["fire_station"] * len(projected)},
                    geometry=projected.geometry,
                    crs=EPSG_5070,
                ).to_crs(EPSG_WGS84)
            )
        if station_frames:
            stations = pd.concat(station_frames, ignore_index=True)
            stations = gpd.GeoDataFrame(stations, geometry="geometry", crs=EPSG_WGS84)
            stations = stations.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
        else:
            stations = gpd.GeoDataFrame({"amenity": []}, geometry=[], crs=EPSG_WGS84)

        roads.to_parquet(roads_path, index=False)
        stations.to_parquet(stations_path, index=False)
        for raw_path in [roads_raw, station_points_raw, station_polygons_raw]:
            remove_path(raw_path)
        print(f"Cached OSM {year}: roads={len(roads):,}, fire_stations={len(stations):,}", flush=True)
        return roads, stations

    try:
        from pyrosm import OSM

        print(f"Extracting OSM layers for {year} with bbox={bbox}", flush=True)
        osm = OSM(str(pbf_path), bounding_box=bbox)
        roads = osm.get_data_by_custom_criteria(
            custom_filter={"highway": DRIVABLE_ROADS},
            osm_keys_to_keep=["highway"],
            filter_type="keep",
            tags_as_columns=["highway"],
            keep_nodes=False,
            keep_ways=True,
            keep_relations=False,
        )
        if roads is None or len(roads) == 0:
            roads = gpd.GeoDataFrame({"highway": []}, geometry=[], crs=EPSG_WGS84)
        else:
            roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty, ["highway", "geometry"]].copy()
            roads = roads.set_crs(EPSG_WGS84, allow_override=True) if roads.crs is None else roads.to_crs(EPSG_WGS84)

        stations = osm.get_data_by_custom_criteria(
            custom_filter={"amenity": ["fire_station"]},
            osm_keys_to_keep=["amenity"],
            filter_type="keep",
            tags_as_columns=["amenity"],
            keep_nodes=True,
            keep_ways=True,
            keep_relations=True,
        )
        if stations is None or len(stations) == 0:
            stations = gpd.GeoDataFrame({"amenity": []}, geometry=[], crs=EPSG_WGS84)
        else:
            stations = stations.loc[stations.geometry.notna() & ~stations.geometry.is_empty, ["amenity", "geometry"]].copy()
            stations = stations.set_crs(EPSG_WGS84, allow_override=True) if stations.crs is None else stations.to_crs(EPSG_WGS84)
            projected = stations.to_crs(EPSG_5070)
            projected["geometry"] = projected.geometry.centroid
            stations = projected.to_crs(EPSG_WGS84)

        roads.to_parquet(roads_path, index=False)
        stations.to_parquet(stations_path, index=False)
        print(f"Cached OSM {year}: roads={len(roads):,}, fire_stations={len(stations):,}", flush=True)
        return roads, stations
    except Exception as exc:
        print(f"Warning: OSM extraction failed for {year}: {exc}")
        return gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84), gpd.GeoDataFrame(geometry=[], crs=EPSG_WGS84)


def make_fire_points_gdf(events: pd.DataFrame):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        events[["fire_id", "year", "lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(events["lon"], events["lat"]),
        crs=EPSG_WGS84,
    )


def road_density_for_fires(fires_5070, roads_5070, radius_m: int) -> pd.Series:
    import geopandas as gpd

    if roads_5070.empty:
        return pd.Series(dtype=float)
    buffers = gpd.GeoDataFrame(
        {"fire_id": fires_5070["fire_id"].values},
        geometry=fires_5070.geometry.buffer(radius_m),
        crs=EPSG_5070,
    )
    joined = gpd.sjoin(roads_5070[["geometry"]], buffers, how="inner", predicate="intersects")
    if joined.empty:
        return pd.Series(dtype=float)
    joined = joined.reset_index(drop=True)
    # Intersect only candidate road/buffer pairs.
    lengths = []
    chunk_size = 100_000
    for start in range(0, len(joined), chunk_size):
        stop = min(start + chunk_size, len(joined))
        chunk = joined.iloc[start:stop]
        left = gpd.GeoSeries(chunk.geometry.values, crs=EPSG_5070)
        right = gpd.GeoSeries(buffers.geometry.iloc[chunk["index_right"].to_numpy()].values, crs=EPSG_5070)
        lengths.append(left.intersection(right, align=False).length.to_numpy())
    joined["length_m"] = np.concatenate(lengths)
    area_km2 = math.pi * (radius_m / 1000.0) ** 2
    return (joined.groupby("fire_id")["length_m"].sum() / 1000.0) / area_km2


def station_counts_for_fires(fires_5070, stations_5070, radius_m: int) -> pd.Series:
    import geopandas as gpd

    if stations_5070.empty:
        return pd.Series(dtype=int)
    buffers = gpd.GeoDataFrame(
        {"fire_id": fires_5070["fire_id"].values},
        geometry=fires_5070.geometry.buffer(radius_m),
        crs=EPSG_5070,
    )
    joined = gpd.sjoin(stations_5070[["geometry"]], buffers, how="inner", predicate="within")
    if joined.empty:
        return pd.Series(dtype=int)
    return joined.groupby("fire_id").size()


def process_osm_event_features(
    base_dir: Path,
    events: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"osm_access_features_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        return pd.read_parquet(path)
    import geopandas as gpd

    outputs = []
    for year in range(start_year, end_year + 1):
        fires = events.loc[events["year"] == year].copy()
        roads, stations = build_or_load_osm_layers(base_dir, events, year, overwrite=overwrite)
        out = fires[["fire_id", "year"]].copy()
        out["distance_to_nearest_drivable_road_m"] = np.nan
        out["road_density_1km_km_per_km2"] = 0.0
        out["road_density_3km_km_per_km2"] = 0.0
        out["road_density_5km_km_per_km2"] = 0.0
        out["nearest_fire_station_distance_km"] = np.nan
        out["fire_station_count_10km"] = 0
        out["fire_station_count_20km"] = 0
        out["fire_station_count_50km"] = 0
        fires_5070 = make_fire_points_gdf(fires).to_crs(EPSG_5070)
        if not roads.empty:
            roads_5070 = roads.to_crs(EPSG_5070)
            nearest = gpd.sjoin_nearest(
                fires_5070[["fire_id", "geometry"]],
                roads_5070[["geometry"]],
                how="left",
                distance_col="distance_m",
            ).groupby("fire_id")["distance_m"].min()
            out["distance_to_nearest_drivable_road_m"] = out["fire_id"].map(nearest)
            for radius_m, suffix in [(1000, "1km"), (3000, "3km"), (5000, "5km")]:
                density = road_density_for_fires(fires_5070, roads_5070, radius_m)
                out[f"road_density_{suffix}_km_per_km2"] = out["fire_id"].map(density).fillna(0.0)
        if not stations.empty:
            stations_5070 = stations.to_crs(EPSG_5070)
            nearest_st = gpd.sjoin_nearest(
                fires_5070[["fire_id", "geometry"]],
                stations_5070[["geometry"]],
                how="left",
                distance_col="distance_m",
            ).groupby("fire_id")["distance_m"].min() / 1000.0
            out["nearest_fire_station_distance_km"] = out["fire_id"].map(nearest_st)
            for radius_m, suffix in [(10_000, "10km"), (20_000, "20km"), (50_000, "50km")]:
                counts = station_counts_for_fires(fires_5070, stations_5070, radius_m)
                out[f"fire_station_count_{suffix}"] = out["fire_id"].map(counts).fillna(0).astype(int)
        outputs.append(out)
        print(f"Wrote in-memory OSM event features for {year}: roads={len(roads):,}, stations={len(stations):,}", flush=True)
    result = pd.concat(outputs, ignore_index=True)
    save_parquet(result, path, overwrite=True)
    return result


def nearest_distance_for_points(points_5070, targets_5070) -> np.ndarray:
    import geopandas as gpd

    distances = np.full(len(points_5070), np.nan, dtype="float64")
    if targets_5070.empty:
        return distances
    joined = gpd.sjoin_nearest(
        points_5070[["point_order", "geometry"]],
        targets_5070[["geometry"]],
        how="left",
        distance_col="distance_m",
    )
    nearest = joined.groupby("point_order")["distance_m"].min()
    distances[nearest.index.to_numpy(dtype=int)] = nearest.to_numpy(dtype=float)
    return distances


def road_length_in_cells(cells_5070, roads_5070) -> np.ndarray:
    import geopandas as gpd
    import shapely

    lengths = np.zeros(len(cells_5070), dtype="float64")
    if roads_5070.empty:
        return lengths
    half = CELL_SIZE_M / 2.0
    cell_polys = gpd.GeoDataFrame(
        {"cell_order": cells_5070["point_order"].to_numpy()},
        geometry=shapely.box(
            cells_5070.geometry.x.to_numpy() - half,
            cells_5070.geometry.y.to_numpy() - half,
            cells_5070.geometry.x.to_numpy() + half,
            cells_5070.geometry.y.to_numpy() + half,
        ),
        crs=EPSG_5070,
    )
    joined = gpd.sjoin(roads_5070[["geometry"]], cell_polys, how="inner", predicate="intersects")
    if joined.empty:
        return lengths
    joined = joined.reset_index(drop=True)
    chunk_size = 100_000
    pieces = []
    for start in range(0, len(joined), chunk_size):
        stop = min(start + chunk_size, len(joined))
        chunk = joined.iloc[start:stop]
        left = gpd.GeoSeries(chunk.geometry.values, crs=EPSG_5070)
        right = gpd.GeoSeries(cell_polys.geometry.iloc[chunk["index_right"].to_numpy()].values, crs=EPSG_5070)
        lengths_part = left.intersection(right, align=False).length.to_numpy()
        pieces.append(pd.DataFrame({"cell_order": chunk["cell_order"].to_numpy(), "length_m": lengths_part}))
    summed = pd.concat(pieces, ignore_index=True).groupby("cell_order")["length_m"].sum()
    lengths[summed.index.to_numpy(dtype=int)] = summed.to_numpy(dtype=float)
    return lengths


def process_osm_patch_375m(
    base_dir: Path,
    events: pd.DataFrame,
    grid_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> Path:
    path = out_path(output_dir, f"event_osm_patch_375m_natural_{start_year}_{end_year}.parquet")
    if path_exists(path) and not overwrite:
        print(f"Reusing {path}")
        return path
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)
    import geopandas as gpd

    for year in range(start_year, end_year + 1):
        grid = pd.read_parquet(grid_path / f"year={year}", columns=["fire_id", "year", "split", "row", "col", "cell_id", "x_5070", "y_5070"])
        roads, stations = build_or_load_osm_layers(base_dir, events, year, overwrite=False)
        patch_parts = []
        if roads.empty:
            roads_5070 = gpd.GeoDataFrame({"highway": []}, geometry=[], crs=EPSG_5070)
            major_5070 = roads_5070
            local_5070 = roads_5070
        else:
            roads_5070 = roads.to_crs(EPSG_5070)
            major_5070 = roads_5070.loc[roads_5070["highway"].isin(MAJOR_ROADS)].copy()
            local_5070 = roads_5070.loc[roads_5070["highway"].isin(LOCAL_ROADS)].copy()
        stations_5070 = stations.to_crs(EPSG_5070) if not stations.empty else gpd.GeoDataFrame(geometry=[], crs=EPSG_5070)
        chunk_size = 250_000
        for start in range(0, len(grid), chunk_size):
            stop = min(start + chunk_size, len(grid))
            chunk = grid.iloc[start:stop].copy().reset_index(drop=True)
            cells = gpd.GeoDataFrame(
                {"point_order": np.arange(len(chunk))},
                geometry=gpd.points_from_xy(chunk["x_5070"], chunk["y_5070"]),
                crs=EPSG_5070,
            )
            chunk["cell_distance_to_nearest_drivable_road_m"] = nearest_distance_for_points(cells, roads_5070)
            chunk["cell_distance_to_nearest_fire_station_m"] = nearest_distance_for_points(cells, stations_5070)
            chunk["cell_distance_to_nearest_major_road_m"] = nearest_distance_for_points(cells, major_5070)
            chunk["cell_distance_to_nearest_track_or_service_road_m"] = nearest_distance_for_points(cells, local_5070)
            chunk["cell_road_length_375m_m"] = road_length_in_cells(cells, roads_5070)
            chunk["cell_has_drivable_road"] = (chunk["cell_road_length_375m_m"] > 0).astype(int)
            patch_parts.append(
                chunk[
                    [
                        "fire_id",
                        "year",
                        "split",
                        "row",
                        "col",
                        "cell_id",
                        "cell_distance_to_nearest_drivable_road_m",
                        "cell_distance_to_nearest_fire_station_m",
                        "cell_road_length_375m_m",
                        "cell_has_drivable_road",
                        "cell_distance_to_nearest_major_road_m",
                        "cell_distance_to_nearest_track_or_service_road_m",
                    ]
                ]
            )
            if start and start % 1_000_000 == 0:
                print(f"Processed OSM patch {year}: {start:,}/{len(grid):,} cells", flush=True)
        patch = pd.concat(patch_parts, ignore_index=True)
        year_dir = path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        patch.to_parquet(year_dir / "part-0.parquet", index=False)
        print(f"Wrote OSM patch {year}: {len(patch):,} rows")
    return path


def merge_master_table(
    events: pd.DataFrame,
    event_tables: list[pd.DataFrame],
    output_dir: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> pd.DataFrame:
    path = out_path(output_dir, f"master_features_natural_{start_year}_{end_year}.parquet")

    def _build() -> pd.DataFrame:
        master = events.copy()
        if master["fire_id"].duplicated().any():
            raise ValueError("Duplicate fire_id in fire_events table.")
        for table in event_tables:
            if "fire_id" not in table.columns:
                raise ValueError("Event-level feature table missing fire_id.")
            if table["fire_id"].duplicated().any():
                raise ValueError("Event-level feature table has duplicated fire_id.")
            master = master.merge(table, on="fire_id", how="left")
        if len(master) != len(events):
            raise ValueError("Master row count changed during merges.")
        save_parquet(master, path, overwrite=True)
        return master

    return load_or_build(path, overwrite, _build)


def _columns_by_prefix(columns: list[str], prefixes: list[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(prefix) for prefix in prefixes)]


def write_feature_manifest(master: pd.DataFrame, output_dir: Path, start_year: int, end_year: int, overwrite: bool) -> None:
    cols = list(master.columns)
    manifest = {
        "metadata": [
            "lat",
            "lon",
            "state",
            "county",
            "discovery_month",
            "discovery_doy",
            "discovery_hour",
            "cause_classification",
            "general_cause",
        ],
        "fire_strict": [
            "has_viirs_detection_1km_D",
            "viirs_count_500m_D",
            "viirs_count_1km_D",
            "viirs_sum_frp_1km_D",
            "viirs_max_frp_1km_D",
            "viirs_mean_frp_1km_D",
            "viirs_max_bright_ti4_1km_D",
            "viirs_mean_bright_ti4_1km_D",
            "viirs_day_count_1km_D",
            "viirs_night_count_1km_D",
        ],
        "fire_wide": [
            "viirs_count_3km_D",
            "viirs_sum_frp_3km_D",
            "viirs_max_frp_3km_D",
            "viirs_mean_frp_3km_D",
            "viirs_max_bright_ti4_3km_D",
            "viirs_mean_bright_ti4_3km_D",
        ],
        "fire_diagnostics": [
            "viirs_nearest_detection_distance_m_D",
            "viirs_min_assigned_distance_m_D",
            "viirs_num_assigned_detections_D",
            "viirs_ambiguous_match_count_D",
        ],
        "weather_aggregate": _columns_by_prefix(cols, ["tmmx_", "tmmn_", "pr_", "rmax_", "rmin_", "sph_", "vpd_", "vs_", "srad_"]),
        "fire_danger_aggregate": _columns_by_prefix(cols, ["erc_", "bi_", "fm100_", "fm1000_", "wind_dir_"]),
        "fuel": _columns_by_prefix(cols, ["fbfm40_", "cbd_", "cbh_", "cc_", "ch_", "fd_", "fvt_", "fvc_", "fvh_"]),
        "vegetation": _columns_by_prefix(cols, ["evt_", "evc_", "evh_"]),
        "topography": _columns_by_prefix(cols, ["elev_", "slope_", "aspect_"]),
        "access": _columns_by_prefix(cols, ["distance_to_nearest_drivable_road", "road_density_", "nearest_fire_station", "fire_station_"]),
        "human": ["population_source_year"] + _columns_by_prefix(cols, ["pop_density_"]),
        "firms_viirs_role": "optional_discovery_day_thermal_feature_not_sample_filter",
        "events_without_viirs_are_retained": True,
        "default_training_note": (
            "FPA-FOD Natural wildfire events are the sample unit. FIRMS/VIIRS "
            "features are optional D-day thermal features. Events without VIIRS "
            "detections remain in the benchmark with zero-valued VIIRS count/FRP/"
            "brightness features."
        ),
        "daily_dynamic_tables": {
            "event_weather_daily": f"gridmet_daily_event_features_natural_{start_year}_{end_year}.parquet",
            "patch_weather_daily": f"event_weather_daily_patch_375m_natural_{start_year}_{end_year}.parquet",
        },
        "patch_tables": {
            "grid_index": f"event_grid_375m_index_natural_{start_year}_{end_year}.parquet",
            "static_patch": f"event_static_patch_375m_natural_{start_year}_{end_year}.parquet",
            "viirs_patch_D": f"event_viirs_patch_375m_D_natural_{start_year}_{end_year}.parquet",
            "weather_daily_patch": f"event_weather_daily_patch_375m_natural_{start_year}_{end_year}.parquet",
            "weather_aggregate_patch": f"event_weather_aggregate_patch_375m_natural_{start_year}_{end_year}.parquet",
            "osm_patch": f"event_osm_patch_375m_natural_{start_year}_{end_year}.parquet",
        },
        "forbidden_as_features": FORBIDDEN_AS_FEATURES,
    }
    write_json(manifest, output_dir / "feature_manifest_natural.json", overwrite)


def write_label_manifest(output_dir: Path, overwrite: bool) -> None:
    manifest = {
        "tasks": {
            "ia_failure": {
                "target_column": "ia_failure_label",
                "task_type": "binary_classification",
                "positive_label": 1,
                "negative_label": 0,
                "ignore_if_nan": True,
                "description": "Initial attack failure prediction. 0 if fire_size_ha <= 10, 1 if fire_size_ha >= 50, NaN for 10-50 ha.",
            },
            "containment_time": {
                "target_column": "log_containment_hours",
                "task_type": "regression",
                "raw_column": "containment_hours",
                "ignore_if_nan": True,
                "description": "Remaining time to containment prediction using log1p containment hours.",
            },
            "burned_area_optional": {
                "target_column": "log_fire_size_ha",
                "task_type": "regression",
                "raw_column": "fire_size_ha",
                "ignore_if_nan": True,
                "description": "Optional burned area target. Do not use this as an input feature for IA failure.",
            },
        },
        "forbidden_as_features": FORBIDDEN_AS_FEATURES,
    }
    write_json(manifest, output_dir / "label_manifest_natural.json", overwrite)


def write_patch_manifest(output_dir: Path, start_year: int, end_year: int, overwrite: bool) -> None:
    manifest = {
        "cell_size_m": CELL_SIZE_M,
        "crs": EPSG_5070,
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
        "patch_radius_m": PATCH_RADIUS_M,
        "center_cell": [CENTER_CELL, CENTER_CELL],
        "sample_unit": "FPA-FOD Natural wildfire event",
        "spatial_unit": "fire_id x 375m event-centered grid cell",
        "outputs": {
            "grid_index": f"event_grid_375m_index_natural_{start_year}_{end_year}.parquet",
            "static_patch": f"event_static_patch_375m_natural_{start_year}_{end_year}.parquet",
            "viirs_patch_D": f"event_viirs_patch_375m_D_natural_{start_year}_{end_year}.parquet",
            "weather_daily_patch": f"event_weather_daily_patch_375m_natural_{start_year}_{end_year}.parquet",
            "weather_aggregate_patch": f"event_weather_aggregate_patch_375m_natural_{start_year}_{end_year}.parquet",
            "osm_patch": f"event_osm_patch_375m_natural_{start_year}_{end_year}.parquet",
        },
    }
    write_json(manifest, output_dir / "event_patch_manifest_375m_natural.json", overwrite)


def write_temporal_protocol_manifest(output_dir: Path, overwrite: bool) -> None:
    manifest = {
        "prediction_time": "discovery day D",
        "weather_daily_range": {
            "min_relative_day": -4,
            "max_relative_day": 0,
            "description": "Daily weather is stored from D-4 to D. D is discovery day.",
        },
        "default_later_training_protocol": {
            "weather_input_days": 5,
            "weather_relative_days": [-4, -3, -2, -1, 0],
            "fire_input_days": 1,
            "fire_relative_days": [0],
            "fire_signal": "VIIRS discovery day D only",
        },
        "supported_later_ablation_protocols": {
            "weather_1d": [0],
            "weather_2d": [-1, 0],
            "weather_3d": [-2, -1, 0],
            "weather_4d": [-3, -2, -1, 0],
            "weather_5d": [-4, -3, -2, -1, 0],
        },
        "important_note": "pipeline.py only saves canonical daily data. dataloader.py later selects input_days and reshapes data into model-specific caches.",
    }
    write_json(manifest, output_dir / "temporal_protocol_manifest_natural.json", overwrite)


def count_parquet_rows(path: Path, columns: list[str] | None = None) -> int:
    if not path_exists(path):
        return 0
    try:
        import pyarrow.parquet as pq

        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        if files:
            return int(sum(pq.ParquetFile(file).metadata.num_rows for file in files))
        return len(pd.read_parquet(path, columns=columns))
    except Exception:
        return 0


def print_missing_rate_summary(name: str, df: pd.DataFrame, key_cols: set[str]) -> None:
    feature_cols = [c for c in df.columns if c not in key_cols]
    if not feature_cols:
        return
    print(f"\nMissing-rate summary: {name}")
    print(df[feature_cols].isna().mean().sort_values(ascending=False).head(30).to_string())


def print_sanity_checks(
    base_dir: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    events: pd.DataFrame,
    master: pd.DataFrame,
    viirs_features: pd.DataFrame,
    gridmet_daily: pd.DataFrame,
    patch_paths: dict[str, Path],
    event_tables: dict[str, pd.DataFrame],
) -> None:
    print("\n================ Phase 1 Compact Report ================")
    print("1. FPA-FOD Natural event count:", len(events))
    print("1b. Events retained after VIIRS processing:", viirs_features["fire_id"].nunique())
    if "viirs_num_assigned_detections_D" in viirs_features.columns:
        no_viirs = int((viirs_features["viirs_num_assigned_detections_D"] == 0).sum())
        print(f"1c. Events without D-day VIIRS detections retained: {no_viirs:,}")
    print("2. Event count by year:")
    print(events["year"].value_counts().sort_index().to_string())
    print("3. Split count:")
    print(events["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("4. Task 1 label count by split:")
    print(
        events.dropna(subset=["ia_failure_label"])
        .groupby("split")["ia_failure_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
        .to_string()
    )
    print("5. Valid containment label count by split:")
    print(
        events.groupby("split")["log_containment_hours"]
        .apply(lambda s: int(s.notna().sum()))
        .reindex(["train", "val", "test"])
        .to_string()
    )
    expected_grid_rows = len(events) * PATCH_SIZE * PATCH_SIZE
    grid_rows = count_parquet_rows(patch_paths["grid_index"], columns=["fire_id"])
    print(f"6. event_grid_375m rows: {grid_rows:,}; expected: {expected_grid_rows:,}")
    expected_daily_rows = len(events) * len(DAILY_RELATIVE_DAYS)
    print(f"7. gridmet_daily_event rows: {len(gridmet_daily):,}; expected: {expected_daily_rows:,}")
    expected_weather_patch_rows = expected_grid_rows * len(DAILY_RELATIVE_DAYS)
    weather_patch_rows = count_parquet_rows(patch_paths["weather_daily_patch"], columns=["fire_id"])
    print(f"8. event_weather_daily_patch rows: {weather_patch_rows:,}; expected: {expected_weather_patch_rows:,}")
    print("9. VIIRS detection match rates:")
    for radius in ["500m", "1km", "3km"]:
        col = f"viirs_count_{radius}_D"
        if col in viirs_features:
            print(f"   {radius}: {(viirs_features[col] > 0).mean() * 100:.2f}%")
    print("10. Ambiguous VIIRS match count:", int(viirs_features["viirs_ambiguous_match_count_D"].sum()))
    population = event_tables.get("population")
    if population is not None and "population_source_year" in population.columns:
        print("10b. WorldPop population_source_year counts:")
        print(population["population_source_year"].value_counts(dropna=False).sort_index().to_string())
    osm_access = event_tables.get("osm_access")
    if osm_access is not None:
        road_valid = osm_access["distance_to_nearest_drivable_road_m"].notna().mean() * 100
        station_valid = osm_access["nearest_fire_station_distance_km"].notna().mean() * 100
        print(f"10c. Events with valid nearest road distance: {road_valid:.2f}%")
        print(f"10d. Events with valid nearest fire station distance: {station_valid:.2f}%")
    print("11. Event-level feature group missing rates:")
    for name, table in event_tables.items():
        print_missing_rate_summary(name, table, {"fire_id", "year"})
    print("12. 375m patch table missing-rate row counts:")
    for name, path in patch_paths.items():
        rows = count_parquet_rows(path, columns=["fire_id"])
        print(f"   {name}: {rows:,} rows at {path}")
    print("13. Master table shape:", master.shape)
    source_missing = []
    source_checks = [
        *landfire_static_raster_paths(base_dir).values(),
        base_dir / "landfire/topography/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif",
        base_dir / "landfire/topography/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif",
        base_dir / "landfire/topography/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif",
    ]
    source_checks.extend(base_dir / "population" / f"usa_pop_{year}.tif" for year in range(start_year - 1, end_year))
    source_checks.extend(base_dir / "osm" / OSM_YEAR_FILES[year] for year in range(start_year, end_year + 1))
    for source in source_checks:
        if not source.exists():
            source_missing.append(str(source))
    print("14. Missing source files:")
    if source_missing:
        for source in source_missing:
            print(f"   {source}")
    else:
        print("   none detected for requested static/access sources")
    placeholder_like = []
    for name, table in event_tables.items():
        feature_cols = [c for c in table.columns if c not in {"fire_id", "year", "population_source", "population_source_year"}]
        if feature_cols and table[feature_cols].isna().all().all():
            placeholder_like.append(name)
    print("15. Placeholder-like all-missing event outputs:")
    print("   " + (", ".join(placeholder_like) if placeholder_like else "none detected"))
    print("16. Output file paths:")
    for p in sorted(output_dir.iterdir()):
        print(f"   {p}")
    print("========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 canonical preprocessing pipeline.")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--start_year", type=int, default=2016)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument(
        "--output_dir",
        default="./data/cache/raw_feature_tables",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stages",
        default=None,
        help=(
            "Comma-separated stages to run. Supported: fpa_fod,grid,viirs,gridmet,"
            "landfire,topography,population,osm,static_patch,master,manifests,validate"
        ),
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = ensure_output_dir(Path(args.output_dir).resolve())
    project_root = Path(".").resolve()
    if not str(base_dir).startswith(str(project_root)):
        raise ValueError("base_dir must stay under .")
    if not str(output_dir).startswith(str(project_root)):
        raise ValueError("output_dir must stay under .")

    supported_stages = {
        "fpa_fod",
        "grid",
        "viirs",
        "gridmet",
        "landfire",
        "topography",
        "population",
        "osm",
        "static_patch",
        "master",
        "manifests",
        "validate",
    }
    all_stages = supported_stages - {"validate"}
    if args.stages:
        stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
        unknown = stages - supported_stages
        if unknown:
            raise ValueError(f"Unknown stage(s): {sorted(unknown)}")
    else:
        stages = set(all_stages)

    def stage_requested(stage: str) -> bool:
        return stage in stages

    fire_events_path = out_path(output_dir, f"fire_events_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("fpa_fod"):
        events = process_fpa_fod(base_dir, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        if not fire_events_path.exists():
            raise FileNotFoundError(f"Missing prerequisite fire events table: {fire_events_path}")
        events = pd.read_parquet(fire_events_path)
        print(f"Loaded existing fire events: {fire_events_path} ({len(events):,} rows)")

    grid_path = out_path(output_dir, f"event_grid_375m_index_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("grid"):
        grid_path = build_event_grid_375m_index(events, output_dir, args.start_year, args.end_year, args.overwrite)
    elif not grid_path.exists():
        raise FileNotFoundError(f"Missing prerequisite grid index: {grid_path}")
    else:
        print(f"Loaded existing grid index path: {grid_path}")

    viirs_path = out_path(output_dir, f"viirs_features_natural_{args.start_year}_{args.end_year}.parquet")
    viirs_patch = out_path(output_dir, f"event_viirs_patch_375m_D_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("viirs"):
        viirs_features = process_viirs_event_features(base_dir, events, output_dir, args.start_year, args.end_year, args.overwrite)
        viirs_patch = process_viirs_patch_375m_D(base_dir, events, grid_path, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        viirs_features = pd.read_parquet(viirs_path)

    gridmet_daily_path = out_path(output_dir, f"gridmet_daily_event_features_natural_{args.start_year}_{args.end_year}.parquet")
    gridmet_features_path = out_path(output_dir, f"gridmet_features_natural_{args.start_year}_{args.end_year}.parquet")
    weather_daily_patch = out_path(output_dir, f"event_weather_daily_patch_375m_natural_{args.start_year}_{args.end_year}.parquet")
    weather_agg_patch = out_path(output_dir, f"event_weather_aggregate_patch_375m_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("gridmet"):
        gridmet_daily = process_gridmet_daily_event_features(base_dir, events, output_dir, args.start_year, args.end_year, args.overwrite)
        gridmet_features = process_gridmet_aggregate_event_features(gridmet_daily, output_dir, args.start_year, args.end_year, args.overwrite)
        weather_daily_patch = process_gridmet_daily_patch_375m(grid_path, gridmet_daily, output_dir, args.start_year, args.end_year, args.overwrite)
        weather_agg_patch = process_gridmet_aggregate_patch_375m(grid_path, gridmet_features, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        gridmet_daily = pd.read_parquet(gridmet_daily_path)
        gridmet_features = pd.read_parquet(gridmet_features_path)

    static_patch = out_path(output_dir, f"event_static_patch_375m_natural_{args.start_year}_{args.end_year}.parquet")
    needs_static_for_events = any(stage_requested(stage) for stage in ["landfire", "topography", "population"])
    if stage_requested("static_patch") or (needs_static_for_events and not static_patch.exists()):
        static_patch = process_static_patch_375m(base_dir, grid_path, output_dir, args.start_year, args.end_year, args.overwrite)
    elif not static_patch.exists():
        raise FileNotFoundError(f"Missing prerequisite static patch: {static_patch}")
    else:
        print(f"Loaded existing static patch path: {static_patch}")

    landfire_path = out_path(output_dir, f"landfire_fuel_veg_features_natural_{args.start_year}_{args.end_year}.parquet")
    topo_path = out_path(output_dir, f"topography_features_natural_{args.start_year}_{args.end_year}.parquet")
    pop_path = out_path(output_dir, f"population_features_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("landfire"):
        landfire_features = process_landfire_event_features(events, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        landfire_features = pd.read_parquet(landfire_path)
    if stage_requested("topography"):
        topo_features = process_topography_event_features(events, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        topo_features = pd.read_parquet(topo_path)
    if stage_requested("population"):
        pop_features = process_population_event_features(base_dir, events, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        pop_features = pd.read_parquet(pop_path)

    osm_path = out_path(output_dir, f"osm_access_features_natural_{args.start_year}_{args.end_year}.parquet")
    osm_patch = out_path(output_dir, f"event_osm_patch_375m_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("osm"):
        osm_features = process_osm_event_features(base_dir, events, output_dir, args.start_year, args.end_year, args.overwrite)
        osm_patch = process_osm_patch_375m(base_dir, events, grid_path, output_dir, args.start_year, args.end_year, args.overwrite)
    else:
        osm_features = pd.read_parquet(osm_path)

    event_tables = {
        "viirs": viirs_features,
        "gridmet": gridmet_features,
        "landfire_fuel_veg": landfire_features,
        "topography": topo_features,
        "osm_access": osm_features,
        "population": pop_features,
    }
    master_path = out_path(output_dir, f"master_features_natural_{args.start_year}_{args.end_year}.parquet")
    if stage_requested("master"):
        master = merge_master_table(
            events,
            list(event_tables.values()),
            output_dir,
            args.start_year,
            args.end_year,
            args.overwrite,
        )
        forbidden_master_cols = {"row", "col", "cell_id", "relative_day"}
        leaked = sorted(forbidden_master_cols & set(master.columns))
        if leaked:
            raise ValueError(f"Master table accidentally contains patch/long columns: {leaked}")
    else:
        master = pd.read_parquet(master_path)

    if stage_requested("manifests"):
        write_feature_manifest(master, output_dir, args.start_year, args.end_year, args.overwrite)
        write_label_manifest(output_dir, args.overwrite)
        write_patch_manifest(output_dir, args.start_year, args.end_year, args.overwrite)
        write_temporal_protocol_manifest(output_dir, args.overwrite)

    patch_paths = {
        "grid_index": grid_path,
        "static_patch": static_patch,
        "viirs_patch_D": viirs_patch,
        "weather_daily_patch": weather_daily_patch,
        "weather_aggregate_patch": weather_agg_patch,
        "osm_patch": osm_patch,
    }
    if stage_requested("validate") or args.stages is not None:
        print_sanity_checks(
        base_dir,
        output_dir,
        args.start_year,
        args.end_year,
        events,
        master,
        viirs_features,
        gridmet_daily,
        patch_paths,
        event_tables,
        )


if __name__ == "__main__":
    main()
