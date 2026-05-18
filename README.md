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

Download the Hugging Face dataset into a temporary folder:

```bash
git lfs install
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

Generate model-ready caches:

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

## Scripts

- `pipeline.py`: optional raw-data canonicalization script. Users who download
  the canonical tables from Hugging Face do not need to run it.
- `dataloader.py`: converts canonical tables into model-ready caches.
- `train.py`: trains tabular, temporal, spatial, and spatiotemporal baselines.

## Supported Models

- Tabular: `logistic_regression`, `xgboost`, `mlp`
- Temporal: `gru`, `tcn`, `transformer`
- Spatial: `resnet18_unet`, `resnet50_unet`, `swin_unet`, `segformer`
- Spatiotemporal: `convlstm`, `convgru`, `predrnn_v2`, `utae`, `swinlstm`,
  `resnet3d`

## License

This code is released under the MIT License.
