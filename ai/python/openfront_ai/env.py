"""Gymnasium-style env wrapping the TypeScript GameEnv via stdin/stdout JSON-RPC."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repo root

GLOBAL_C, GLOBAL_H, GLOBAL_W = 12, 64, 128
LOCAL_C, LOCAL_H, LOCAL_W = 8, 64, 64
VECTOR_DIM = 64
NUM_ACTION_TYPES = 10
NUM_TARGET = 2
COARSE_W, COARSE_H = 32, 16
NUM_FRAC = 5
NUM_BUILD = 10


@dataclass
class StepResult:
    obs: dict
    reward: float
    done: bool
    info: dict


class TsRpcClient:
    """One persistent `npx tsx ai/ts/rpc_server.ts` process."""

    def __init__(self, map_name: str = "plains", stride: int = 20):
        self.map_name = map_name
        self.stride = stride
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0

    def start(self) -> None:
        if self.proc is not None:
            return
        cmd = [
            "npx",
            "tsx",
            str(ROOT / "ai/ts/rpc_server.ts"),
            "--map",
            self.map_name,
            "--stride",
            str(self.stride),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Skip any non-JSON preamble
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("rpc died before ready")
            line = line.strip()
            if not line.startswith("{"):
                continue
            ready = json.loads(line)
            if ready.get("ready"):
                break
            raise RuntimeError(f"rpc not ready: {ready}")

    def _readline_json(self) -> dict:
        assert self.proc and self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("rpc died")
            line = line.strip()
            if not line.startswith("{"):
                continue
            return json.loads(line)

    def call(self, cmd: str, **kwargs) -> Any:
        self.start()
        assert self.proc and self.proc.stdin
        self._id += 1
        msg = {"id": self._id, "cmd": cmd, **kwargs}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        resp = self._readline_json()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "rpc error"))
        return resp["result"]

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.call("quit")
        except Exception:
            pass
        self.proc.kill()
        self.proc = None


def _reshape_obs(raw: dict) -> dict:
    g = np.asarray(raw["global"], dtype=np.float32).reshape(GLOBAL_C, GLOBAL_H, GLOBAL_W)
    l = np.asarray(raw["local"], dtype=np.float32).reshape(LOCAL_C, LOCAL_H, LOCAL_W)
    v = np.asarray(raw["vector"], dtype=np.float32)
    return {"global": g, "local": l, "vector": v}


class OpenFrontEnv:
    """Single-env wrapper. Uses TS RPC when available; otherwise a tiny stub."""

    def __init__(
        self,
        map_dir: str = "",
        seed: int = 0,
        difficulty: str = "impossible",
        use_ts: bool = True,
        map_name: str = "plains",
        stride: int = 20,
    ):
        self.map_dir = map_dir
        self.seed = seed
        self.difficulty = difficulty
        self.map_name = map_name
        self.stride = stride
        self._client: Optional[TsRpcClient] = None
        self._use_ts = use_ts
        self._stub_tick = 0
        self._stub_tiles = 50
        self._stub_troops = 25000.0
        self._stub_gold = 0
        self._last_mask = np.zeros(NUM_ACTION_TYPES, dtype=bool)
        self._last_mask[0] = True
        self._last_mask[2] = True

    def _ensure(self) -> None:
        if not self._use_ts:
            return
        if self._client is None:
            self._client = TsRpcClient(self.map_name, self.stride)
            try:
                self._client.start()
            except Exception as e:
                print(f"[openfront_ai] TS RPC unavailable ({e}); using stub", file=sys.stderr)
                self._use_ts = False
                self._client = None

    def reset(self) -> dict:
        self._ensure()
        if self._client is not None:
            result = self._client.call("reset", seed=self.seed)
            self._last_mask = np.asarray(result["mask"]["actionType"], dtype=bool)
            return _reshape_obs(result["obs"])
        self._stub_tick = 0
        self._stub_tiles = 50
        self._stub_troops = 25000.0
        self._stub_gold = 0
        return self._stub_obs()

    def _stub_obs(self) -> dict:
        rng = np.random.default_rng(self.seed + self._stub_tick)
        return {
            "global": rng.random((GLOBAL_C, GLOBAL_H, GLOBAL_W), dtype=np.float32) * 0.01,
            "local": rng.random((LOCAL_C, LOCAL_H, LOCAL_W), dtype=np.float32) * 0.01,
            "vector": np.array(
                [
                    math.log1p(self._stub_gold),
                    100.0,
                    self._stub_troops / 100000.0,
                    self._stub_tiles / 10000.0,
                ]
                + [0.0] * (VECTOR_DIM - 4),
                dtype=np.float32,
            ),
        }

    def step(self, action: Tuple[int, int, int, int, int, int]) -> StepResult:
        atype, target, cx, cy, frac, build = action
        self._ensure()
        if self._client is not None:
            result = self._client.call(
                "step",
                action={
                    "actionType": int(atype),
                    "targetPlayer": int(target),
                    "cellX": int(cx),
                    "cellY": int(cy),
                    "troopFrac": int(frac),
                    "buildType": int(build),
                },
            )
            self._last_mask = np.asarray(result["mask"]["actionType"], dtype=bool)
            info = result["info"]
            return StepResult(
                _reshape_obs(result["obs"]),
                float(info["reward"]),
                bool(info["done"]),
                info,
            )
        self._stub_tick += 5
        if atype == 2:
            self._stub_tiles += 3
            self._stub_troops *= 0.98
        self._stub_gold += 500
        self._stub_troops += 50
        done = self._stub_tick > 2000 or self._stub_tiles > 8000
        reward = (5.0 if atype == 2 else 0.0) + 0.05
        if done:
            reward += 10.0 if self._stub_tiles > 4000 else -10.0
        return StepResult(self._stub_obs(), reward, done, {"tick": self._stub_tick, "stub": True})

    def legal_action_types(self) -> np.ndarray:
        return self._last_mask.copy()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def make_vec_env(n: int, map_name: str = "plains", seed: int = 0) -> list[OpenFrontEnv]:
    return [
        OpenFrontEnv(map_name=map_name, seed=seed + i, use_ts=True) for i in range(n)
    ]
