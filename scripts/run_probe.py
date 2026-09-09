#!/usr/bin/env python
"""Frozen-backbone bake-off: extract feature caches, then linear-probe every
backbone × {mixed, d2r, r2d}.

    python scripts/run_probe.py                       # features (all) + probe
    python scripts/run_probe.py --stage features      # just build the caches
    python scripts/run_probe.py --stage probe         # just probe existing caches
    python scripts/run_probe.py --only convnext_t_22k,dinov2_vits14
"""
import argparse

from pipeline.common.config import Config
from pipeline.features import extract
from pipeline.train import probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", choices=["all", "features", "probe"], default="all")
    ap.add_argument("--only", default="", help="comma-separated backbone keys")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = Config(args.config)
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or None
    bs = cfg.model.get("batch_size", 64)

    if args.stage in ("all", "features"):
        extract.run(cfg, keys=keys, batch_size=bs, workers=args.workers)
    if args.stage in ("all", "probe"):
        probe.run(cfg, keys=keys, C=cfg.model.get("probe_C", 1.0))


if __name__ == "__main__":
    main()
