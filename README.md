# Garbage Detection

Waste image classification on the *Recyclable and Household Waste Classification* dataset.
The 30 dataset categories are mapped to a recyclability target (`recyclable` / `organic` /
`non_recyclable`). The pipeline covers data validation,
deduplication, leakage-safe splitting, frozen-backbone feature extraction, and a linear
probe over six ImageNet/DINOv2 backbones.

## Requirements

Python 3.12. Dependencies are pinned in `requirements.txt` (PyTorch, timm, scikit-learn,
pandas, pyarrow, matplotlib).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Data

Download the dataset from Kaggle and place it under `data/raw/`:

```
data/raw/<class>/<default|real_world>/*.png
```

30 classes, 2 domains, 250 images each (15,000 PNG, 256x256 RGB).
See [`data/README.md`](data/README.md) for the download and directory-naming steps.

## Usage

```bash
.venv/bin/python scripts/prepare_data.py    # validate, deduplicate, split
.venv/bin/python scripts/run_probe.py       # extract features, fit linear probes
.venv/bin/python scripts/make_report.py     # figures and REPORT.md
```

All three accept `--config` and `--stage` to re-run a single step from cached outputs;
`run_probe.py` also accepts `--only <backbone,...>` and `--workers`.

Settings live in `config.yaml`: paths, input schema, deduplication thresholds, split
protocols, backbone registry, and the 30-class to target mapping.

## Evaluation protocols

Splits are grouped by duplicate cluster and stratified by class.

| Protocol | Train | Test |
| --- | --- | --- |
| `mixed` | both domains | both domains |
| `d2r` | `default` | `real_world` |
| `r2d` | `real_world` | `default` |

## Results

Frozen backbone with a logistic-regression head, macro-F1:

| Backbone | dim | mixed | d2r | r2d |
| --- | ---: | ---: | ---: | ---: |
| dinov2_vits14 | 384 | **0.928** | 0.831 | 0.880 |
| convnext_t_22k | 768 | 0.921 | **0.854** | 0.897 |
| vit_s16_21k | 384 | 0.920 | 0.799 | 0.876 |
| effnet_b0_ns | 1280 | 0.910 | 0.806 | 0.854 |
| mobilenetv3_l | 1280 | 0.841 | 0.742 | 0.818 |
| resnet18 | 512 | 0.838 | 0.702 | 0.799 |


## Project structure

```
config.yaml            configuration for every stage
pipeline/
  common/              config loading, 30-class taxonomy
  data/                manifest, quality checks, splitting, cleaning
  features/            backbone registry, frozen feature extraction
  train/               linear probe, metrics
  report/              figures
scripts/               prepare_data.py, run_probe.py, make_report.py
data/raw/              dataset (not tracked)
data/processed/        manifest and split tables (not tracked)
artifacts/             features (not tracked), figures, probe results
REPORT.md              generated report
```

## Dataset citation

Alistair King, *Recyclable and Household Waste Classification*, Kaggle, 2024.
<https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification>
