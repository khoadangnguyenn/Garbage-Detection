"""Stage 1: one validated row per image -> manifest.parquet."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from tqdm import tqdm

from pipeline.common.taxonomy import FINE_CLASSES, MATERIAL_GROUP


def phash(img: Image.Image, size: int = 8, hi: int = 32) -> int:
    g = np.asarray(img.convert("L").resize((hi, hi), Image.Resampling.LANCZOS), dtype=np.float64)
    d = dctn(g, norm="ortho")[:size, :size].flatten()
    bits = d[1:] > np.median(d[1:])
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _one(path: Path, schema: dict) -> dict:
    fine, domain = path.parent.parent.name, path.parent.name
    rec = dict(
        uid=f"{fine}/{domain}/{path.name}", path=str(path),
        fine_label=fine, material_group=MATERIAL_GROUP.get(fine, "UNKNOWN"),
        domain=domain, width=-1, height=-1, mode="", fmt="",
        ok=False, issues="", sha256="", phash=-1,
    )
    try:
        rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            im.load()
            rec.update(width=im.width, height=im.height, mode=im.mode, fmt=im.format or "")
            rec["phash"] = phash(im, schema["_phash_size"])
        issues = []
        if [rec["width"], rec["height"]] != list(schema["size"]):
            issues.append(f"size={rec['width']}x{rec['height']}")
        if rec["mode"] != schema["mode"]:
            issues.append(f"mode={rec['mode']}")
        if rec["fmt"] != schema["format"]:
            issues.append(f"fmt={rec['fmt']}")
        if domain not in schema["domains"]:
            issues.append(f"domain={domain}")
        if fine not in MATERIAL_GROUP:
            issues.append("unknown_class")
        rec["issues"] = ";".join(issues)
        rec["ok"] = not issues
    except Exception as e:  # noqa: BLE001 -- record every failure instead of raising
        rec["issues"] = f"read_error:{type(e).__name__}:{e}"
    return rec


def build(cfg) -> pd.DataFrame:
    schema = dict(cfg.schema, _phash_size=cfg.dedup["phash_size"])
    files = sorted(cfg.paths["raw"].glob("*/*/*.png"))
    if not files:
        raise SystemExit(f"no images under {cfg.paths['raw']}  (see data/README.md)")

    df = pd.DataFrame(_one(p, schema) for p in tqdm(files, desc="scan", unit="img"))
    df["target"] = df["fine_label"].map(cfg.target_map)
    df["label_id"] = df["fine_label"].map({c: i for i, c in enumerate(FINE_CLASSES)})

    cfg.paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.paths["manifest"], index=False)
    _report(df, cfg)
    return df


def _report(df: pd.DataFrame, cfg):
    n = len(df)
    print(f"\n── manifest ─ {n} images ─ config {cfg.hash()}")
    bad = df[~df.ok]
    print(f"valid: {df.ok.sum()}/{n}" + ("" if bad.empty else f"   BAD: {len(bad)}"))
    if not bad.empty:
        print(bad.issues.value_counts().head(10).to_string())
    piv = df.pivot_table(index="fine_label", columns="domain", values="uid", aggfunc="count", fill_value=0)
    off = piv[(piv != cfg.schema["images_per_class_per_domain"]).any(axis=1)]
    print(f"class count: all {len(piv)} = {cfg.schema['images_per_class_per_domain']}/domain"
          if off.empty else "class count OFF:\n" + off.to_string())
    print("target balance:")
    print(df.groupby("target").agg(n=("uid", "size"), n_fine=("fine_label", "nunique")).to_string())
