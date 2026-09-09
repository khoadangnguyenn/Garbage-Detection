#!/usr/bin/env python
"""Data pipeline: raw images -> manifest.parquet -> quality_report -> splits.parquet

    python scripts/prepare_data.py                  # full run  (~20s / 15k imgs)
    python scripts/prepare_data.py --stage split    # reuse cached manifest
"""
import argparse

import pandas as pd

from pipeline.common.config import Config
from pipeline.data import clean, manifest, quality, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", choices=["all", "manifest", "quality", "split", "clean"], default="all")
    args = ap.parse_args()

    cfg = Config(args.config)
    print(f"config {cfg.hash()}  scheme={cfg.task['scheme']}  targets={cfg.target_classes}")

    if args.stage in ("all", "manifest"):
        df = manifest.build(cfg)
    else:
        df = pd.read_parquet(cfg.paths["manifest"])

    if args.stage in ("all", "quality"):
        quality.build(df, cfg)
    if args.stage in ("all", "split"):
        split.build(df, cfg)
    if args.stage in ("all", "clean"):
        clean.build(df, cfg)

    print("\ndone.")


if __name__ == "__main__":
    main()
