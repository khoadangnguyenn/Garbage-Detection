"""Frozen-backbone feature extraction with an on-disk cache."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pipeline.features.backbones import BACKBONES


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _ImgDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def _build(name: str, model_kwargs: dict, device: str):
    model = timm.create_model(name, pretrained=True, num_classes=0, **(model_kwargs or {}))
    model.eval().to(device)
    dc = resolve_model_data_config(model)
    dc["input_size"], dc["crop_pct"] = (3, 224, 224), 1.0
    return model, create_transform(**dc, is_training=False)


def extract_one(cfg, key: str, device: str, batch_size: int, workers: int) -> Path:
    out_dir = cfg.paths["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.npz"
    if out.exists():
        print(f"[{key}] cache hit")
        return out

    df = pd.read_parquet(cfg.paths["manifest"])
    df = df[df.ok].reset_index(drop=True)
    model, tf = _build(BACKBONES[key]["name"], BACKBONES[key].get("model_kwargs"), device)
    dl = DataLoader(_ImgDS(df.path.tolist(), tf), batch_size=batch_size,
                    num_workers=workers, shuffle=False, pin_memory=(device == "cuda"))

    feats, t0 = [], time.time()
    with torch.inference_mode():
        for xb in tqdm(dl, desc=key, unit="batch"):
            feats.append(model(xb.to(device)).float().cpu().numpy())
    X = np.concatenate(feats).astype(np.float32)
    np.savez(out, uid=df.uid.to_numpy(), X=X)
    print(f"[{key}] {X.shape} in {time.time() - t0:.0f}s")
    return out


def run(cfg, keys=None, batch_size=64, workers=4):
    device = pick_device()
    keys = keys or list(BACKBONES)
    print(f"device={device}  backbones={keys}")
    return {k: extract_one(cfg, k, device, batch_size, workers) for k in keys}
