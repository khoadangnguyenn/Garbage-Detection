"""Stage 4: one row per duplicate group, dropping target-ambiguous groups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.data.split import assign_groups
from pipeline.data.split import build as _build_splits


def build(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df[df.ok].copy().reset_index(drop=True)
    df["group_id"] = assign_groups(df, cfg.dedup["phash_max_hamming"])

    g = df.groupby("group_id")
    agg = g.agg(n_copies=("uid", "size"), n_domain=("domain", "nunique"),
                n_fine=("fine_label", "nunique"), n_target=("target", "nunique"))
    rep = df.sort_values(["group_id", "domain", "uid"]).groupby("group_id").head(1).set_index("group_id")
    clean = rep.join(agg)
    clean["fine_label"] = g.fine_label.agg(lambda s: s.mode().iloc[0])
    clean["target"] = g.target.agg(lambda s: s.mode().iloc[0])
    clean["fine_label_ambiguous"] = clean.n_fine > 1
    clean["domain"] = np.where(clean.n_domain > 1, "both", clean["domain"])

    dropped = clean[clean.n_target > 1]
    clean = clean[clean.n_target == 1].reset_index().drop(columns=["n_domain", "n_target", "issues", "ok"])
    clean.to_parquet(cfg.paths["manifest_clean"], index=False)

    print(f"\n── clean ─ {len(df)} raw → {df.group_id.nunique()} groups → {len(clean)} clean rows "
          f"({len(dropped)} target-ambiguous dropped)")
    print(f"  fine_label-ambiguous kept: {int(clean.fine_label_ambiguous.sum())}")
    print(f"  domain: {clean.domain.value_counts().to_dict()}")
    print(f"  target: {clean.target.value_counts().to_dict()}")

    cln = clean.copy()
    cln["group_id"] = np.arange(len(cln))
    cln["ok"] = True
    _build_splits(cln, cfg, out_path=cfg.paths["splits_clean"], regroup=False)
    return clean
