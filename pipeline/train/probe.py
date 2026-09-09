"""Linear probe: frozen features with a logistic-regression head, per protocol."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.features.backbones import BACKBONES
from pipeline.train.metrics import confusion, score

PROTOCOLS = ["mixed", "d2r", "r2d"]


def _load_features(path):
    d = np.load(path, allow_pickle=True)
    return {u: i for i, u in enumerate(d["uid"])}, d["X"]


def _probe_one(X, uid2row, splits, proto, labels, seed, C):
    s = splits.dropna(subset=[proto])
    tr, te = s[s[proto] == "train"], s[s[proto] == "test"]
    Xtr = X[[uid2row[u] for u in tr.uid]]
    Xte = X[[uid2row[u] for u in te.uid]]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed),
    )
    clf.fit(Xtr, tr.target.to_numpy())
    pred = clf.predict(Xte)
    yte = te.target.to_numpy()
    m = score(yte, pred, labels)
    m.update(n_train=len(tr), n_test=len(te))
    return m, (yte, pred), confusion_matrix(yte, pred, labels=labels).tolist()


def run(cfg, keys=None, C=1.0, clean=False):
    keys = keys or list(BACKBONES)
    splits = pd.read_parquet(cfg.paths["splits_clean"] if clean else cfg.paths["splits"])
    labels = cfg.target_classes
    rows, best, confusions = [], None, {}

    for key in keys:
        fpath = cfg.paths["features_dir"] / f"{key}.npz"
        if not fpath.exists():
            print(f"[{key}] no features cache, skip")
            continue
        uid2row, X = _load_features(fpath)
        for proto in PROTOCOLS:
            m, yp, cm = _probe_one(X, uid2row, splits, proto, labels, cfg.seed, C)
            rows.append(dict(backbone=key, feat_dim=X.shape[1], protocol=proto, **m))
            confusions.setdefault(key, {})[proto] = cm
            print(f"{key:<15} {proto:<6} acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} "
                  f"binAcc={m['bin_acc']:.3f} recycRecall={m['bin_recyc_recall']:.3f}")
            if proto == "mixed" and (best is None or m["macro_f1"] > best[1]):
                best = (key, m["macro_f1"], yp)

    res = pd.DataFrame(rows)
    cfg.paths["results"].parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(cfg.paths["results"], index=False)
    (cfg.paths["results"].parent / "probe_confusion.json").write_text(
        json.dumps({"labels": labels, "confusions": confusions}, indent=2))

    piv = res.pivot_table(index="backbone", columns="protocol", values="macro_f1")
    piv["mixed→d2r drop"] = (piv["mixed"] - piv["d2r"]).round(3)
    print(f"\nmacro-F1 by backbone × protocol:\n{piv.round(3).to_string()}")
    if best:
        key, f1, (yt, yp) = best
        print(f"\nbest on mixed: {key} ({f1:.3f})\n{confusion(yt, yp, labels)}")
    return res
