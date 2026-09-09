"""Report figures -> artifacts/figures/*.png. Captions live in REPORT.md, not on the plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

TARGET_COLOR = {"recyclable": "#2e7d32", "non_recyclable": "#c62828", "organic": "#f9a825"}
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p)
    plt.close(fig)
    return p


def class_balance(m, mc, out):
    order = m.groupby("fine_label").size().sort_values().index
    raw = m.groupby("fine_label").size().reindex(order)
    clean = mc.groupby("fine_label").size().reindex(order).fillna(0)
    colors = [TARGET_COLOR[t] for t in m.drop_duplicates("fine_label").set_index("fine_label")
              .target.reindex(order)]
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(order))
    ax.barh(y + 0.2, raw, 0.4, color=colors, alpha=0.4, label="raw (500)")
    ax.barh(y - 0.2, clean, 0.4, color=colors, label="clean")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("images")
    ax.set_title("Per-class count: raw vs clean")
    ax.legend(loc="lower right")
    return _save(fig, out, "01_class_balance.png")


def target_balance(m, mc, out):
    cats = ["recyclable", "non_recyclable", "organic"]
    raw, clean = m.target.value_counts().reindex(cats), mc.target.value_counts().reindex(cats)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(cats))
    ax.bar(x - 0.2, raw, 0.4, label="raw", color=[TARGET_COLOR[c] for c in cats], alpha=0.4)
    ax.bar(x + 0.2, clean, 0.4, label="clean", color=[TARGET_COLOR[c] for c in cats])
    ax.bar_label(ax.containers[0], fontsize=8)
    ax.bar_label(ax.containers[1], fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("images")
    ax.set_title("Target balance (3-class)")
    ax.legend()
    return _save(fig, out, "02_target_balance.png")


def dup_per_class(q, out):
    pct = (pd.Series(q["per_class_redundant_copies"]).sort_values() / 500 * 100)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(pct.index, pct.values, color="#c62828")
    ax.set_xlabel("redundant byte-identical copies  (% of 500)")
    ax.set_title(f"Per-class duplication  —  {q['exact_dup']['redundant_pct']}% dataset-wide")
    return _save(fig, out, "03_dup_per_class.png")


def domain_composition(mc, q, out):
    comp = mc.domain.value_counts().reindex(["default", "real_world", "both"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar_label(ax.bar(comp.index, comp.values, color=["#1565c0", "#6a1b9a", "#8d6e63"]))
    ax.set_ylabel("dedup groups")
    ax.set_title(f"Clean-group domain — {q['cross_domain_dup_groups']} 'both' (same file, both folders)")
    return _save(fig, out, "04_domain_composition.png")


def label_conflicts(lc, out):
    pairs = (lc.groupby("sha256").fine_label.apply(lambda s: " ↔ ".join(sorted(s.unique())))
             .value_counts().head(12).sort_values())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar_label(ax.barh(pairs.index, pairs.values, color="#f9a825"))
    ax.set_xlabel("byte-identical images filed under both labels")
    ax.set_title(f"Label conflicts — {lc.sha256.nunique()} groups (6 flip the target)")
    return _save(fig, out, "05_label_conflicts.png")


def split_sizes(sp_raw, sp_clean, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, sp, title in [(axes[0], sp_raw, "raw"), (axes[1], sp_clean, "clean")]:
        protos = ["mixed", "d2r", "r2d"]
        tr = [(sp[p] == "train").sum() for p in protos]
        va = [(sp[p] == "val").sum() for p in protos]
        te = [(sp[p] == "test").sum() for p in protos]
        ax.bar(protos, tr, label="train", color="#1565c0")
        ax.bar(protos, va, bottom=tr, label="val", color="#90caf9")
        ax.bar(protos, te, bottom=np.add(tr, va), label="test", color="#ef9a9a")
        for i, p in enumerate(protos):
            used = tr[i] + va[i] + te[i]
            ax.text(i, used, f"{used / len(sp) * 100:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
    axes[0].set_ylabel("rows")
    axes[1].legend(loc="upper right")
    fig.suptitle("Split sizes  (d2r/r2d drop cross-domain groups)")
    return _save(fig, out, "06_split_sizes.png")


def dup_examples(m, out, n_dup=3, n_conf=3):
    dup_h = (m[m.sha256.duplicated(keep=False)].groupby("sha256")
             .filter(lambda d: d.fine_label.nunique() == 1).sha256.drop_duplicates().head(n_dup))
    conf_h = (m.groupby("sha256").filter(lambda d: d.fine_label.nunique() > 1)
              .sha256.drop_duplicates().head(n_conf))
    rows = [("EXACT DUP", h) for h in dup_h] + [("CONFLICT", h) for h in conf_h]
    fig, axes = plt.subplots(len(rows), 3, figsize=(5.5, 1.7 * len(rows)))
    for r, (kind, h) in enumerate(rows):
        g = m[m.sha256 == h].head(3)
        for c in range(3):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(g):
                rec = g.iloc[c]
                ax.imshow(Image.open(rec.path))
                cap = f"{rec.domain}" if kind == "EXACT DUP" else f"{rec.fine_label}\n{rec.domain}"
                ax.set_title(cap, fontsize=7, color="#c62828" if kind == "CONFLICT" else "black")
        axes[r, 0].text(-0.1, 0.5, kind, rotation=90, transform=axes[r, 0].transAxes,
                        va="center", ha="center", fontsize=8, weight="bold")
    fig.suptitle("Same bytes, different folder / label")
    fig.tight_layout()
    return _save(fig, out, "07_dup_examples.png")


def probe_macrof1(res, out):
    piv = res.pivot_table(index="backbone", columns="protocol",
                          values="macro_f1")[["mixed", "r2d", "d2r"]].sort_values("d2r", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.65, vmax=0.95, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    for i in range(len(piv)):
        for j in range(3):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Linear-probe macro-F1 (frozen backbone)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out, "08_probe_macrof1.png")


def domain_gap(res, out):
    piv = res.pivot_table(index="backbone", columns="protocol", values="macro_f1").sort_values("d2r")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(piv))
    ax.barh(y + 0.2, piv["mixed"], 0.4, label="mixed", color="#90caf9")
    ax.barh(y - 0.2, piv["d2r"], 0.4, label="d2r (studio→real)", color="#1565c0")
    for i in range(len(piv)):
        ax.text(piv["mixed"].iloc[i] + .004, y[i] + 0.2,
                f"Δ{piv['mixed'].iloc[i] - piv['d2r'].iloc[i]:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index)
    ax.set_xlim(0.65, 0.97)
    ax.set_xlabel("macro-F1")
    ax.set_title("Studio → real-world gap (smaller Δ = more deployable)")
    ax.legend(loc="lower right")
    return _save(fig, out, "09_domain_gap.png")


def confusion_best(res, conf, out):
    labels = conf["labels"]
    best = res[res.protocol == "d2r"].sort_values("macro_f1").iloc[-1].backbone
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, proto in zip(axes, ["mixed", "d2r"]):
        cm = np.array(conf["confusions"][best][proto], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l[:8] for l in labels], rotation=20)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l[:8] for l in labels])
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
        f1 = res[(res.backbone == best) & (res.protocol == proto)].macro_f1.iloc[0]
        ax.set_title(f"{proto}  (macro-F1 {f1:.3f})")
        ax.set_ylabel("true")
        ax.set_xlabel("pred")
    fig.suptitle(f"Confusion — {best}")
    return _save(fig, out, "10_confusion_best.png")


def all_data(cfg):
    out = cfg.paths["figures_dir"]
    m = pd.read_parquet(cfg.paths["manifest"])
    m = m[m.ok]
    mc = pd.read_parquet(cfg.paths["manifest_clean"])
    q = json.loads((cfg.paths["processed"] / "quality_report.json").read_text())
    lc = pd.read_parquet(cfg.paths["processed"] / "label_conflicts.parquet")
    return [
        class_balance(m, mc, out), target_balance(m, mc, out), dup_per_class(q, out),
        domain_composition(mc, q, out), label_conflicts(lc, out),
        split_sizes(pd.read_parquet(cfg.paths["splits"]), pd.read_parquet(cfg.paths["splits_clean"]), out),
        dup_examples(m, out),
    ]


def all_model(cfg):
    out = cfg.paths["figures_dir"]
    if not cfg.paths["results"].exists():
        return []
    res = pd.read_csv(cfg.paths["results"])
    conf = json.loads((cfg.paths["results"].parent / "probe_confusion.json").read_text())
    return [probe_macrof1(res, out), domain_gap(res, out), confusion_best(res, conf, out)]
