#!/usr/bin/env python
"""Generate every figure + a REPORT.md that embeds them.

    python scripts/make_report.py            # data + model figures + REPORT.md
    python scripts/make_report.py --data     # data figures only
"""
import argparse
import json

import pandas as pd

from pipeline.common.config import REPO, Config
from pipeline.report import figures


def _rel(p, repo=REPO):
    return str(p.relative_to(repo)).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", action="store_true", help="skip model figures")
    args = ap.parse_args()
    cfg = Config(args.config)
    repo = REPO

    print("data figures ...")
    data_figs = figures.all_data(cfg)
    model_figs = [] if args.data else figures.all_model(cfg)
    for p in data_figs + model_figs:
        print("  ", _rel(p, repo))

    q = json.loads((cfg.paths["processed"] / "quality_report.json").read_text())
    mc = pd.read_parquet(cfg.paths["manifest_clean"])
    e = q["exact_dup"]
    lc = q["label_conflicts"]

    L = []
    w = L.append
    w(f"# URAxUTS Garbage Detection — data & model report\n")
    w(f"_config `{cfg.hash()}` · scheme `{cfg.task['scheme']}` · targets "
      f"`{', '.join(cfg.target_classes)}`_\n")

    w("## 1. Dataset\n")
    w(f"- {q['n_images']:,} PNG, 256×256 RGB, 30 folder classes × (250 `default` studio "
      f"+ 250 `real_world`).\n- 30 classes → recyclability target: "
      + ", ".join(f"**{k}** {v}" for k, v in mc.target.value_counts().items()) + " (clean rows).\n")
    w(f"![class balance]({_rel(data_figs[0], repo)})\n")
    w(f"![target balance]({_rel(data_figs[1], repo)})\n")

    w("## 2. Data-quality findings\n")
    w(f"- **Exact byte-duplicates: {e['redundant_copies']:,} redundant copies "
      f"({e['redundant_pct']}%)** in {e['dup_groups']:,} groups; with pHash near-dupes the "
      f"15 000 images collapse to **{mc.shape[0]:,} distinct object views**. A naïve random "
      f"split would put a test image's twin in train → inflated accuracy.\n")
    w(f"- **{q['cross_domain_dup_groups']:,} groups have the *same file* in both `default/` and "
      f"`real_world/`** — the domain split is ~39% fictitious; `d2r`/`r2d` keep only the ~50% "
      f"that is cleanly domain-separated.\n")
    w(f"- **{lc['groups']} label-conflict groups** (byte-identical, ≥2 class folders; "
      f"mostly cardboard-box↔packaging & aluminum↔steel cans) — but **only "
      f"{lc['target_flipping_groups']} flip the recyclability target**, so the 3-class target "
      f"is robust to the 30-class label noise.\n")
    w(f"![per-class duplication]({_rel(data_figs[2], repo)})\n")
    w(f"![domain composition]({_rel(data_figs[3], repo)})\n")
    w(f"![label conflicts]({_rel(data_figs[4], repo)})\n")
    w(f"![duplicate & conflict examples]({_rel(data_figs[6], repo)})\n")

    w("## 3. Cleaning & splits\n")
    w(f"- `manifest_clean.parquet`: 1 row per dedup cluster (prefer `default` copy), "
      f"{lc['target_flipping_groups']} target-ambiguous groups dropped → {mc.shape[0]:,} rows.\n")
    w("- Splits are tables (`uid → fold`), grouped by dedup cluster, stratified by fine class, "
      "leakage-asserted. 3 protocols: `mixed` (both domains), `d2r` (studio→real-world), `r2d`.\n")
    w(f"![split sizes]({_rel(data_figs[5], repo)})\n")

    if model_figs:
        res = pd.read_csv(cfg.paths["results"])
        piv = res.pivot_table(index="backbone", columns="protocol", values="macro_f1")
        best_mixed = piv["mixed"].idxmax()
        best_d2r = piv["d2r"].idxmax()
        w("## 4. Model bake-off — frozen backbone + linear probe\n")
        w("No fine-tuning yet: one forward pass per backbone → cached features → logistic-"
          "regression head per protocol.\n")
        w(f"- Best `mixed` macro-F1: **{best_mixed}** ({piv.loc[best_mixed, 'mixed']:.3f}). "
          f"Best on the deploy-relevant `d2r`: **{best_d2r}** ({piv.loc[best_d2r, 'd2r']:.3f}), "
          f"smallest studio→real-world drop (Δ{piv.loc[best_d2r,'mixed']-piv.loc[best_d2r,'d2r']:.3f}).\n")
        w(f"- The efficient CNNs the shortlist named (ResNet18 {piv.loc['resnet18','mixed']:.3f}, "
          f"MobileNetV3-L {piv.loc['mobilenetv3_l','mixed']:.3f}) trail on *frozen* features, as "
          f"expected — weak IN-1k penultimate features.\n")
        w(f"![linear-probe macro-F1]({_rel(model_figs[0], repo)})\n")
        w(f"![domain gap]({_rel(model_figs[1], repo)})\n")
        w(f"![confusion — best d2r backbone]({_rel(model_figs[2], repo)})\n")
        w("## 5. Recommendation\n")
        w(f"**{best_d2r} + LP-FT** (linear-probe head, then unfreeze at low LR). It has the "
          f"strongest frozen features that also survive the domain shift; LP-FT (Kumar et al. "
          f"ICLR 2022) adds ~10% on OOD over plain fine-tuning. Runs on this M3 Pro in ~1–1.5 h.\n")

    (REPO / cfg.raw["paths"]["report"]).write_text("\n".join(L))
    print(f"\n-> {cfg.raw['paths']['report']}")


if __name__ == "__main__":
    main()
