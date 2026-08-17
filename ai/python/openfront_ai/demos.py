"""Load expert JSONL demos and provide batch helpers for BC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .env import (
    COARSE_H,
    COARSE_W,
    GLOBAL_C,
    GLOBAL_H,
    GLOBAL_W,
    LOCAL_C,
    LOCAL_H,
    LOCAL_W,
    NUM_ACTION_TYPES,
    NUM_BUILD,
    NUM_CELL,
    NUM_FRAC,
    NUM_TARGET,
    VECTOR_DIM,
    _parse_mask,
    _reshape_obs,
)


# Pre-100-bin demos used these five spend labels (indices 0..4).
_LEGACY_TROOP_FRACS = (0.1, 0.2, 0.35, 0.5, 0.75)


def _migrate_troop_frac(idx: int, bins: int) -> int:
    """Map demo troopFrac index onto current NUM_FRAC vocabulary."""
    if bins == NUM_FRAC:
        return max(0, min(NUM_FRAC - 1, idx))
    if bins == 5 and 0 <= idx < len(_LEGACY_TROOP_FRACS):
        v = _LEGACY_TROOP_FRACS[idx]
        return max(0, min(NUM_FRAC - 1, int(round(v * 100)) - 1))
    return max(0, min(NUM_FRAC - 1, idx))


def load_jsonl_demos(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            obs = _reshape_obs(raw["obs"])
            mask = _parse_mask(raw.get("mask"))
            a = raw["action"]
            bins = int(raw.get("troopFracBins", 5))
            # Legacy 1v1 demos used targetPlayer 0=TN / 1=enemy — keep slot 1 valid.
            target = int(a["targetPlayer"])
            if target > 0:
                target = max(1, min(NUM_TARGET - 1, target))
            action = (
                int(a["actionType"]),
                target,
                int(a["cellX"]),
                int(a["cellY"]),
                _migrate_troop_frac(int(a["troopFrac"]), bins),
                int(a["buildType"]),
            )
            rows.append(
                {
                    "obs": obs,
                    "mask": mask,
                    "action": action,
                    "reward": float(raw.get("reward", 0.0)),
                    "done": bool(raw.get("done", False)),
                    "macro": int(raw.get("macroGoal", -1)),
                }
            )
    return rows


def empty_batch(n: int) -> dict[str, np.ndarray]:
    return {
        "global": np.zeros((n, GLOBAL_C, GLOBAL_H, GLOBAL_W), dtype=np.float32),
        "local": np.zeros((n, LOCAL_C, LOCAL_H, LOCAL_W), dtype=np.float32),
        "vector": np.zeros((n, VECTOR_DIM), dtype=np.float32),
        "action": np.zeros((n, 6), dtype=np.int64),
        "actionType_mask": np.zeros((n, NUM_ACTION_TYPES), dtype=bool),
        "targetPlayer_mask": np.ones((n, NUM_TARGET), dtype=bool),
        "cell_mask": np.ones((n, NUM_CELL), dtype=bool),
        "troopFrac_mask": np.ones((n, NUM_FRAC), dtype=bool),
        "buildType_mask": np.zeros((n, NUM_BUILD), dtype=bool),
        "reward": np.zeros(n, dtype=np.float32),
        "done": np.zeros(n, dtype=bool),
        "macro": np.full(n, -1, dtype=np.int64),
    }


def stack_transitions(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    n = len(rows)
    b = empty_batch(n)
    for i, r in enumerate(rows):
        b["global"][i] = r["obs"]["global"]
        b["local"][i] = r["obs"]["local"]
        b["vector"][i] = r["obs"]["vector"]
        b["action"][i] = r["action"]
        m = r["mask"]
        b["actionType_mask"][i] = m["actionType"]
        b["targetPlayer_mask"][i] = m["targetPlayer"]
        b["cell_mask"][i] = m["cell"]
        b["troopFrac_mask"][i] = m["troopFrac"]
        b["buildType_mask"][i] = m["buildType"]
        b["reward"][i] = r.get("reward", 0.0)
        b["done"][i] = r.get("done", False)
        b["macro"][i] = r.get("macro", -1)
    return b
