"""Stage 2: measure (not fix) duplicates, label conflicts and domain leakage."""
from __future__ import annotations

import json

import pandas as pd


def build(df: pd.DataFrame, cfg) -> dict:
    df = df[df.ok].copy()
    n = len(df)

    grp = df.groupby("sha256")
    dup_rows = grp.filter(lambda d: len(d) > 1)
    agg = grp.agg(size=("uid", "size"), n_domain=("domain", "nunique"),
                  n_label=("fine_label", "nunique"), n_target=("target", "nunique"))
    multi = agg[agg["size"] > 1]

    conflict_hashes = multi[multi.n_label > 1].index
    conflicts = (df[df.sha256.isin(conflict_hashes)]
                 .sort_values(["sha256", "fine_label"])
                 [["sha256", "uid", "fine_label", "material_group", "target", "domain"]])
    conflicts.to_parquet(cfg.paths["processed"] / "label_conflicts.parquet", index=False)

    per_class = (df.groupby("fine_label")
                 .apply(lambda d: int(d.sha256.duplicated().sum()), include_groups=False)
                 .sort_values(ascending=False))

    redundant = len(dup_rows) - dup_rows.sha256.nunique()
    report = {
        "config_hash": cfg.hash(),
        "n_images": n,
        "exact_dup": {
            "rows_in_dup_groups": int(len(dup_rows)),
            "dup_groups": int(dup_rows.sha256.nunique()),
            "redundant_copies": int(redundant),
            "redundant_pct": round(redundant / n * 100, 1),
        },
        "cross_domain_dup_groups": int((multi.n_domain > 1).sum()),
        "label_conflicts": {
            "groups": int((multi.n_label > 1).sum()),
            "rows": int(len(conflicts)),
            "target_flipping_groups": int((multi.n_target > 1).sum()),
        },
        "per_class_redundant_copies": per_class.to_dict(),
    }
    (cfg.paths["processed"] / "quality_report.json").write_text(json.dumps(report, indent=2))

    e, lc = report["exact_dup"], report["label_conflicts"]
    print(f"\n── quality ─ {n} images ─ config {report['config_hash']}")
    print(f"exact byte-dupes : {e['rows_in_dup_groups']} rows / {e['dup_groups']} groups / "
          f"{e['redundant_copies']} redundant ({e['redundant_pct']}%)")
    print(f"cross-domain dupes: {report['cross_domain_dup_groups']} groups")
    print(f"label conflicts  : {lc['groups']} groups / {lc['rows']} rows  "
          f"(target-flipping: {lc['target_flipping_groups']})")
    return report
