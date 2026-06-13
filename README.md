# WildfireIA

Official implementation for **WildfireIA**, a benchmark for predicting whether a
wildfire will escape initial attack from public information available at fire
discovery time.

Code is maintained by the Responsible AI Lab at Florida State University.

Dataset release:

<https://huggingface.co/datasets/WildfireIA/Anonymous-WildfireIA>

The dataset repository contains canonical benchmark tables and Croissant
metadata. It does not contain model-ready caches. After cloning this code
repository, place the canonical tables at the path below and regenerate caches
with `dataloader.py`.

## Quick Start

Clone this repository:

```bash
git clone https://github.com/LabRAI/WildfireIA.git
cd WildfireIA
```

Install Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch GPU wheels depend on the local CUDA version. If the default `torch`
installation is not compatible with your system, install PyTorch following the
official PyTorch instructions, then rerun `pip install -r requirements.txt`.

Download the Hugging Face dataset into a temporary folder. The most portable
method is `huggingface_hub`, which downloads Git LFS files without requiring a
system `git-lfs` installation:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="WildfireIA/Anonymous-WildfireIA",
    repo_type="dataset",
    local_dir="hf_data",
)
PY
```

If `git-lfs` is already installed, this equivalent command also works:

```bash
git clone https://huggingface.co/datasets/WildfireIA/Anonymous-WildfireIA hf_data
```

Copy the canonical tables into this repository:

```bash
mkdir -p data/canonical/raw_feature_tables
rsync -a hf_data/data/canonical/raw_feature_tables/ data/canonical/raw_feature_tables/
```

The expected path is:

```text
data/canonical/raw_feature_tables/
```

## Canonical Data and Input Contracts

The Hugging Face release contains canonical tables, not prebuilt training
caches. The main canonical files are:

```text
fire_events_natural_2016_2020.parquet
master_features_natural_2016_2020.parquet
gridmet_features_natural_2016_2020.parquet
gridmet_daily_event_features_natural_2016_2020.parquet
viirs_features_natural_2016_2020.parquet
landfire_fuel_veg_features_natural_2016_2020.parquet
topography_features_natural_2016_2020.parquet
osm_access_features_natural_2016_2020.parquet
population_features_natural_2016_2020.parquet
event_*_patch_375m_*.parquet
```

The manifests in the same folder define the data contract:

```text
feature_manifest_natural.json
label_manifest_natural.json
temporal_protocol_manifest_natural.json
event_patch_manifest_375m_natural.json
```

These files list feature groups, forbidden target/leakage columns, task labels,
weather-day windows, and the event-centered 375 m patch geometry. The patch
contract is a 29 x 29 grid centered on each FPA-FOD Natural wildfire event,
with 375 m cells in EPSG:5070.

`dataloader.py` converts these canonical tables into model-ready caches. The
supported `--input_protocol` values are:

```text
metadata
firms
weather
fuel
vegetation
topography
access
human
metadata_vegetation
metadata_fuel
metadata_topography
metadata_access
metadata_human
all
all_without_fire
all_without_weather
all_without_vegetation
all_without_fuel
all_without_topography
all_without_access
all_without_human
```

For the official full-input setting, the generated Task 1 cache shapes are:

```text
tabular:        X_train.npy        [22576, 6029]
temporal:       X_seq_train.npy    [22576, 5, 15]
                X_static_train.npy [22576, 5940]
spatial:        X_train.npy        [22576, 121, 29, 29]
spatiotemporal: X_train.npy        [22576, 5, 47, 29, 29]
```

Each cache directory also contains `metadata.json`, feature/channel names,
`sample_index_{split}.parquet`, `fire_id_{split}.npy`, and `y_{split}.npy`.

Generate Task 1 model-ready caches:

```bash
python dataloader.py \
  --base_dir . \
  --canonical_dir data/canonical/raw_feature_tables \
  --output_dir data/cache/model_ready \
  --task ia_failure \
  --representation all \
  --weather_days 5 \
  --input_protocol all \
  --overwrite
```

Run one baseline:

```bash
python train.py \
  --base_dir . \
  --task ia_failure \
  --experiment_type smoke \
  --representation tabular \
  --weather_days 5 \
  --input_protocol all \
  --model xgboost \
  --seed 553371 \
  --overwrite
```

The output is written to:

```text
experiments/ia_failure/smoke/tabular/weather5_all/xgboost_seed553371/
```

Important output files include:

```text
config.json
metrics.json
predictions_val.parquet
predictions_test.parquet
```

## Main Experiment Pattern

Experiment outputs follow this directory format:

```text
experiments/{task}/{experiment_type}/{representation}/weather{days}_{protocol}/{model}_seed{seed}/
```

Example full-input XGBoost run:

```bash
python train.py \
  --base_dir . \
  --task ia_failure \
  --experiment_type full \
  --representation tabular \
  --weather_days 5 \
  --input_protocol all \
  --model xgboost \
  --seed 553371 \
  --overwrite
```

Example spatial neural baseline:

```bash
python train.py \
  --base_dir . \
  --task ia_failure \
  --experiment_type full \
  --representation spatial \
  --weather_days 5 \
  --input_protocol all \
  --model swin_unet \
  --seed 553371 \
  --max_epochs 100 \
  --batch_size 64 \
  --early_stop_patience 15 \
  --sampling_strategy weighted \
  --standardize_channels \
  --overwrite
```

For the containment-duration task, first generate caches:

```bash
python dataloader.py \
  --base_dir . \
  --canonical_dir data/canonical/raw_feature_tables \
  --output_dir data/cache/model_ready \
  --task containment_time \
  --representation all \
  --weather_days 5 \
  --input_protocol all \
  --overwrite
```

Then run a containment-duration model:

```bash
python train.py \
  --base_dir . \
  --task containment_time \
  --experiment_type full \
  --representation tabular \
  --weather_days 5 \
  --input_protocol all \
  --model xgboost \
  --seed 553371 \
  --overwrite
```

## Summarizing Results

After full experiments finish:

```bash
python summarize_task1_full_all_seeds.py
python summarize_task2_full_all_seeds.py
```

Summary CSV and Markdown files are written under:

```text
results/
```

## Scripts

- `pipeline.py`: optional raw-data canonicalization script. Users who download
  the canonical tables from Hugging Face do not need to run it.
- `dataloader.py`: converts canonical tables into model-ready caches.
- `train.py`: trains tabular, temporal, spatial, and spatiotemporal baselines.
- `summarize_*.py`: summarizes full-test and ablation experiment outputs.

## Supported Models

- Tabular: `logistic_regression`, `xgboost`, `mlp`
- Temporal: `gru`, `tcn`, `transformer`
- Spatial: `resnet18_unet`, `resnet50_unet`, `swin_unet`, `segformer`
- Spatiotemporal: `convlstm`, `convgru`, `predrnn_v2`, `utae`, `swinlstm`,
  `resnet3d`

## License

This code is released under the MIT License.
