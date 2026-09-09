"""Stage 3: duplicate grouping and leakage-safe splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class _DSU:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _popcount64(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.uint64)
    m1, m2, m4 = np.uint64(0x5555555555555555), np.uint64(0x3333333333333333), np.uint64(0x0F0F0F0F0F0F0F0F)
    a = a - ((a >> np.uint64(1)) & m1)
    a = (a & m2) + ((a >> np.uint64(2)) & m2)
    a = (a + (a >> np.uint64(4))) & m4
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def assign_groups(df: pd.DataFrame, phash_max_hamming: int) -> pd.Series:
    dsu = _DSU(df.uid.tolist())
    for _, g in df.groupby("sha256"):
        u = g.uid.tolist()
        for x in u[1:]:
            dsu.union(u[0], x)
    for _, g in df.groupby("fine_label"):        # pHash compared only within a class
        uids, h = g.uid.to_numpy(), g.phash.to_numpy(dtype=np.uint64)
        for i in range(len(uids)):
            d = _popcount64(np.bitwise_xor(h[i], h[i + 1:]))
            for j in np.nonzero(d <= phash_max_hamming)[0]:
                dsu.union(uids[i], uids[i + 1 + j])
    root = {u: dsu.find(u) for u in df.uid}
    codes = {r: k for k, r in enumerate(sorted(set(root.values())))}
    return df.uid.map(lambda u: codes[root[u]]).rename("group_id")


def _fold(df, y_col, seed, n_splits):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(skf.split(np.arange(len(df)), df[y_col].to_numpy(), df["group_id"].to_numpy()))


def _split_mixed(df, cfg):
    r, y = cfg.split["protocols"]["mixed"], cfg.split["stratify_by"]
    tr, te = _fold(df, y, cfg.seed, max(2, round(1 / r["test"])))
    sub = df.iloc[tr].reset_index(drop=True)
    _, va = _fold(sub, y, cfg.seed + 1, max(2, round(1 / (r["val"] / (r["train"] + r["val"])))))
    out = pd.Series("train", index=df.uid, name="mixed")
    out[df.iloc[te].uid] = "test"
    out[sub.iloc[va].uid] = "val"
    return out


def _split_domain(df, cfg, train_dom, test_dom, val_frac):
    span = df.groupby("group_id").domain.nunique()
    cross = set(span[span > 1].index)             # same object in both domains -> exclude
    keep = df[~df.group_id.isin(cross)]
    pool = keep[keep.domain == train_dom].reset_index(drop=True)
    tr, va = _fold(pool, cfg.split["stratify_by"], cfg.seed, max(2, round(1 / val_frac)))
    out = pd.Series(pd.NA, index=df.uid, name=f"{train_dom[0]}2{test_dom[0]}", dtype="object")
    out[pool.iloc[tr].uid] = "train"
    out[pool.iloc[va].uid] = "val"
    out[keep[keep.domain == test_dom].uid] = "test"
    return out, len(cross)


def build(df: pd.DataFrame, cfg, out_path=None, regroup: bool = True) -> pd.DataFrame:
    out_path = out_path or cfg.paths["splits"]
    df = df[df.ok].copy().reset_index(drop=True)
    if regroup:
        df["group_id"] = assign_groups(df, cfg.dedup["phash_max_hamming"])

    mixed = _split_mixed(df, cfg)
    d2r, x1 = _split_domain(df, cfg, "default", "real_world",
                            cfg.split["protocols"]["d2r"]["val_frac_of_default"])
    r2d, x2 = _split_domain(df, cfg, "real_world", "default",
                            cfg.split["protocols"]["r2d"]["val_frac_of_real"])

    out = df[["uid", "fine_label", "target", "domain", "group_id"]].copy()
    out["mixed"], out["d2r"], out["r2d"] = out.uid.map(mixed), out.uid.map(d2r), out.uid.map(r2d)
    out.to_parquet(out_path, index=False)
    _report(out, df, cfg, x1, x2, out_path.name)
    return out


def _report(out, df, cfg, cross_d2r, cross_r2d, name):
    ng = df.group_id.nunique()
    dup = len(df) - ng
    print(f"\n── splits ─ {name}")
    print(f"dedup: {len(df)} -> {ng} groups  ({dup} folded, {dup / len(df) * 100:.1f}%)")
    for col in ("mixed", "d2r", "r2d"):
        g = out.dropna(subset=[col]).groupby("group_id")[col].nunique()
        assert (g <= 1).all(), f"LEAK in {col}: {(g > 1).sum()} groups span folds"
    print(f"leakage check: PASS   |   cross-domain groups dropped: d2r={cross_d2r} r2d={cross_r2d}")
    for col in ("mixed", "d2r", "r2d"):
        s = out[col].dropna()
        cnt = s.value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict()
        bal = out.dropna(subset=[col]).groupby(col).target.value_counts(normalize=True).unstack().round(3)
        print(f"\n[{col}]  {cnt}\n{bal.to_string()}")
