"""Load config.yaml + derive the recyclability target map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .taxonomy import FINE_CLASSES

REPO = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self, path: str | Path = REPO / "config.yaml"):
        self.path = Path(path)
        self.raw = yaml.safe_load(self.path.read_text())
        self.seed = self.raw["seed"]
        self.paths = {k: (REPO / v) for k, v in self.raw["paths"].items()}
        self.schema = self.raw["schema"]
        self.dedup = self.raw["dedup"]
        self.split = self.raw["split"]
        self.task = self.raw["task"]
        self.model = self.raw.get("model", {})
        self._build_target_map()

    def _build_target_map(self):
        lm: dict[str, list[str]] = self.task["label_map"]
        covered = [c for cls in lm.values() for c in cls]
        missing = set(FINE_CLASSES) - set(covered)
        dupe = sorted({c for c in covered if covered.count(c) > 1})
        if missing:
            raise ValueError(f"label_map missing classes: {sorted(missing)}")
        if dupe:
            raise ValueError(f"label_map has classes in >1 bucket: {dupe}")

        three = {c: bucket for bucket, cls in lm.items() for c in cls}
        if self.task["scheme"] == "binary":
            self.target_map = {c: ("recyclable" if b == "recyclable" else "non_recyclable")
                               for c, b in three.items()}
        else:
            self.target_map = three
        self.target_classes = sorted(set(self.target_map.values()))

    def hash(self) -> str:
        """Short deterministic id of the config -> stamp artifacts with it."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:12]
