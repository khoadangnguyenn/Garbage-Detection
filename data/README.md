# Data

`raw/` and `processed/` are not tracked. This file covers how to obtain `raw/`;
`processed/` is produced by `scripts/prepare_data.py`.

## Source

Alistair King, *Recyclable and Household Waste Classification* v1.0 (2024-05-18), Kaggle.
<https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification>

15,000 PNG images, 256x256 RGB, across 30 categories. Each category has a `default`
(studio) and a `real_world` folder with 250 images each. Free for educational, research
and non-commercial use. Original notes: `references/dataset_README.txt`.

## Download

```bash
kaggle datasets download -d alistairking/recyclable-and-household-waste-classification
unzip recyclable-and-household-waste-classification.zip -d /tmp/rhwc
```

The archive stores categories as `images/<Title Case Name>/`. The pipeline expects
snake_case directories directly under `data/raw/`, with no `images/` level:

```bash
mkdir -p data/raw
for d in /tmp/rhwc/images/*/; do
  cp -r "$d" "data/raw/$(basename "$d" | tr 'A-Z ' 'a-z_')"
done
```

Verify:

```bash
ls data/raw | wc -l                                  # 30
ls data/raw/plastic_water_bottles/default | wc -l    # 250
```

`prepare_data.py` validates size, mode, format, domains and per-class counts against
`config.yaml : schema` and fails if the layout is wrong.

## Layout

```
data/raw/<class>/<default|real_world>/Image_<n>.png
data/processed/
  manifest.parquet          one row per image with hashes, domain and labels
  quality_report.json       duplicate, conflict and domain-overlap counts
  label_conflicts.parquet   identical files appearing in several classes
  manifest_clean.parquet    one row per duplicate cluster (9,341 rows)
  splits.parquet            uid to fold, all protocols
  splits_clean.parquet      uid to fold on the cleaned manifest
```
