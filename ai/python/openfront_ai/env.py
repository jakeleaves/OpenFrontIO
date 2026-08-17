"""Gymnasium-style env wrapping the TypeScript GameEnv via stdin/stdout JSON-RPC.

Training uses one Node `rpc_pool.ts` process with N worker_threads and an optional
binary obs side-channel on an inherited pipe (OPENFRONT_OBS_FD). Viz keeps using
`rpc_server.ts` (single env).
"""

from __future__ import annotations

import base64
import json
import math
import os
import selectors
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repo root
AI_ROOT = Path(__file__).resolve().parents[2]  # ai/
LOG_DIR = AI_ROOT / "fixtures" / "logs"

GLOBAL_C, GLOBAL_H, GLOBAL_W = 12, 64, 128
LOCAL_C, LOCAL_H, LOCAL_W = 8, 64, 64
VECTOR_DIM = 64
NUM_ACTION_TYPES = 10
MAX_NATIONS = 72  # must match ai/ts/types.ts
NUM_TARGET = 1 + MAX_NATIONS  # TN + nation slots
COARSE_W, COARSE_H = 32, 16
NUM_FRAC = 100  # 1% .. 100% (must match ai/ts/types.ts TROOP_FRACS)
NUM_BUILD = 10
NUM_CELL = COARSE_H * COARSE_W

DEFAULT_STRIDE = 5
DEFAULT_RESET_TIMEOUT_S = 120.0
DEFAULT_STEP_TIMEOUT_S = 60.0
MAX_RPC_RESTARTS = 5

_FLOAT_COUNT = (
    GLOBAL_C * GLOBAL_H * GLOBAL_W + LOCAL_C * LOCAL_H * LOCAL_W + VECTOR_DIM
)
FLOAT_BYTES = _FLOAT_COUNT * 4


def _mask_nbytes(n: int) -> int:
    return (n + 7) // 8


MASK_BYTES = (
    _mask_nbytes(NUM_ACTION_TYPES)
    + _mask_nbytes(NUM_TARGET)
    + _mask_nbytes(NUM_CELL)
    + _mask_nbytes(NUM_FRAC)
    + _mask_nbytes(NUM_BUILD)
)
FRAME_BYTES = FLOAT_BYTES + MASK_BYTES


class RpcError(RuntimeError):
    """Base class for recoverable RPC failures."""


class RpcTimeout(RpcError):
    """Child process did not respond within the deadline."""


class RpcDied(RpcError):
    """Child process exited or closed stdout mid-call."""


@dataclass
class StepResult:
    obs: dict
    reward: float
    done: bool
    info: dict
    mask: dict
    timing: dict = field(default_factory=dict)


def _f32_from_b64(s: str, shape: tuple[int, ...]) -> np.ndarray:
    buf = base64.b64decode(s)
    arr = np.frombuffer(buf, dtype=np.float32)
    return arr.reshape(shape).copy()


def _reshape_obs(raw: dict) -> dict:
    """Accept base64 Float32 (encoding=f32b64) or legacy JSON float arrays."""
    encoding = raw.get("encoding")
    if encoding == "f32b64" or isinstance(raw.get("global"), str):
        g = _f32_from_b64(raw["global"], (GLOBAL_C, GLOBAL_H, GLOBAL_W))
        l = _f32_from_b64(raw["local"], (LOCAL_C, LOCAL_H, LOCAL_W))
        v = _f32_from_b64(raw["vector"], (VECTOR_DIM,))
        if v.shape[0] < VECTOR_DIM:
            pad = np.zeros(VECTOR_DIM, dtype=np.float32)
            pad[: v.shape[0]] = v
            v = pad
        return {"global": g, "local": l, "vector": v}

    g = np.asarray(raw["global"], dtype=np.float32).reshape(
        GLOBAL_C, GLOBAL_H, GLOBAL_W
    )
    l = np.asarray(raw["local"], dtype=np.float32).reshape(LOCAL_C, LOCAL_H, LOCAL_W)
    v = np.asarray(raw["vector"], dtype=np.float32)
    if v.shape[0] < VECTOR_DIM:
        pad = np.zeros(VECTOR_DIM, dtype=np.float32)
        pad[: v.shape[0]] = v
        v = pad
    return {"global": g, "local": l, "vector": v}


def _parse_mask(raw: dict | None) -> dict:
    if not raw:
        m = {
            "actionType": np.zeros(NUM_ACTION_TYPES, dtype=bool),
            "targetPlayer": np.ones(NUM_TARGET, dtype=bool),
            "cell": np.ones(NUM_CELL, dtype=bool),
            "troopFrac": np.ones(NUM_FRAC, dtype=bool),
            "buildType": np.zeros(NUM_BUILD, dtype=bool),
        }
        m["actionType"][0] = True
        return m

    def _fit(arr: Any, n: int, fill: bool) -> np.ndarray:
        a = np.asarray(arr, dtype=bool).reshape(-1)
        out = np.full(n, fill, dtype=bool)
        out[: min(n, a.shape[0])] = a[:n]
        return out

    return {
        "actionType": _fit(raw["actionType"], NUM_ACTION_TYPES, False),
        # Legacy 1v1 masks had length 2; pad remaining nation slots as illegal.
        "targetPlayer": _fit(
            raw.get("targetPlayer", [True, True]), NUM_TARGET, False
        ),
        "cell": _fit(raw.get("cell", [True] * NUM_CELL), NUM_CELL, True),
        "troopFrac": _fit(
            raw.get("troopFrac", [True] * NUM_FRAC), NUM_FRAC, True
        ),
        "buildType": _fit(
            raw.get("buildType", [False] * NUM_BUILD), NUM_BUILD, False
        ),
    }


def _unpack_bits(buf: bytes | memoryview, offset: int, n: int) -> tuple[np.ndarray, int]:
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        out[i] = bool(buf[offset + (i >> 3)] & (1 << (i & 7)))
    return out, offset + _mask_nbytes(n)


def decode_obs_mask_frame(data: bytes) -> tuple[dict, dict]:
    """Decode a packed fd-3 frame into obs dict + mask dict."""
    if len(data) < FRAME_BYTES:
        raise RpcError(f"short obs frame: {len(data)} < {FRAME_BYTES}")
    f32 = np.frombuffer(data, dtype=np.float32, count=_FLOAT_COUNT)
    g_n = GLOBAL_C * GLOBAL_H * GLOBAL_W
    l_n = LOCAL_C * LOCAL_H * LOCAL_W
    g = f32[:g_n].reshape(GLOBAL_C, GLOBAL_H, GLOBAL_W).copy()
    l = f32[g_n : g_n + l_n].reshape(LOCAL_C, LOCAL_H, LOCAL_W).copy()
    v = f32[g_n + l_n : g_n + l_n + VECTOR_DIM].copy()
    mv = memoryview(data)
    o = FLOAT_BYTES
    action_type, o = _unpack_bits(mv, o, NUM_ACTION_TYPES)
    target_player, o = _unpack_bits(mv, o, NUM_TARGET)
    cell, o = _unpack_bits(mv, o, NUM_CELL)
    troop_frac, o = _unpack_bits(mv, o, NUM_FRAC)
    build_type, o = _unpack_bits(mv, o, NUM_BUILD)
    obs = {"global": g, "local": l, "vector": v}
    mask = {
        "actionType": action_type,
        "targetPlayer": target_player,
        "cell": cell,
        "troopFrac": troop_frac,
        "buildType": build_type,
    }
    return obs, mask


def pack_obs_mask_frame(obs: dict, mask: dict) -> bytes:
    """Pack obs+mask for tests (mirrors ai/ts/wire_codec.ts)."""
    parts = [
        np.asarray(obs["global"], dtype=np.float32).ravel().tobytes(),
        np.asarray(obs["local"], dtype=np.float32).ravel().tobytes(),
        np.asarray(obs["vector"], dtype=np.float32).ravel().tobytes(),
    ]
    out = bytearray(b"".join(parts))
    if len(out) != FLOAT_BYTES:
        raise ValueError(f"float payload {len(out)} != {FLOAT_BYTES}")

    def append_bits(bits: np.ndarray, n: int) -> None:
        nbytes = _mask_nbytes(n)
        buf = bytearray(nbytes)
        arr = np.asarray(bits, dtype=bool).reshape(-1)
        for i in range(min(n, arr.shape[0])):
            if arr[i]:
                buf[i >> 3] |= 1 << (i & 7)
        out.extend(buf)

    append_bits(mask["actionType"], NUM_ACTION_TYPES)
    append_bits(mask["targetPlayer"], NUM_TARGET)
    append_bits(mask["cell"], NUM_CELL)
    append_bits(mask["troopFrac"], NUM_FRAC)
    append_bits(mask["buildType"], NUM_BUILD)
    return bytes(out)


def _tail_file(path: Path, max_bytes: int = 4096) -> str:
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _action_dict(action: Tuple[int, int, int, int, int, int]) -> dict:
    atype, target, cx, cy, frac, build = action
    return {
        "actionType": int(atype),
        "targetPlayer": int(target),
        "cellX": int(cx),
        "cellY": int(cy),
        "troopFrac": int(frac),
        "buildType": int(build),
    }


class TsRpcClient:
    """Legacy single-env `rpc_server.ts` client (viz / tests). Prefer TsRpcPoolClient."""

    def __init__(
        self,
        map_name: str = "world",
        stride: int = DEFAULT_STRIDE,
        difficulty: str = "Impossible",
        nations: int = MAX_NATIONS,
        bots: int = 0,
        max_ticks: int = 30_000,
        reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
        step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
    ):
        self.map_name = map_name
        self.stride = stride
        self.difficulty = difficulty
        self.nations = nations
        self.bots = bots
        self.max_ticks = max_ticks
        self.reset_timeout_s = reset_timeout_s
        self.step_timeout_s = step_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._stderr_path: Optional[Path] = None
        self._stderr_fh: Any = None
        self._pending = False

    def start(self) -> None:
        if self.proc is not None:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._stderr_path = LOG_DIR / f"rpc_{os.getpid()}_{id(self)}.log"
        self._stderr_fh = open(self._stderr_path, "ab", buffering=0)
        cmd = [
            "npx",
            "tsx",
            str(ROOT / "ai/ts/rpc_server.ts"),
            "--map",
            self.map_name,
            "--stride",
            str(self.stride),
            "--difficulty",
            self.difficulty,
            "--nations",
            str(self.nations),
            "--bots",
            str(self.bots),
            "--max-ticks",
            str(self.max_ticks),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            text=True,
            bufsize=1,
        )
        ready = self._readline_json(timeout_s=self.reset_timeout_s)
        if not ready.get("ready"):
            raise RpcDied(f"rpc not ready: {ready}")

    def _kill_child(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self.proc = None
        self._pending = False
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None

    def _attach_error(self, exc: Exception) -> Exception:
        tail = (
            _tail_file(self._stderr_path) if self._stderr_path is not None else ""
        )
        if not tail:
            return exc
        msg = f"{exc} | rpc stderr tail:\n{tail}"
        if isinstance(exc, RpcTimeout):
            return RpcTimeout(msg)
        if isinstance(exc, RpcDied):
            return RpcDied(msg)
        return RpcError(msg)

    def _readline_json(self, timeout_s: float) -> dict:
        assert self.proc and self.proc.stdout
        deadline = time.monotonic() + timeout_s
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_child()
                    raise self._attach_error(RpcTimeout(f"rpc timeout after {timeout_s}s"))
                events = sel.select(timeout=min(remaining, 1.0))
                if not events:
                    if self.proc.poll() is not None:
                        raise self._attach_error(
                            RpcDied(f"rpc exited {self.proc.returncode}")
                        )
                    continue
                line = self.proc.stdout.readline()
                if not line:
                    raise self._attach_error(RpcDied("rpc died"))
                line = line.strip()
                if not line.startswith("{"):
                    continue
                return json.loads(line)
        finally:
            sel.close()

    def _timeout_for(self, cmd: str) -> float:
        if cmd in ("reset", "viz_reset", "reset_batch"):
            return self.reset_timeout_s
        return self.step_timeout_s

    def send(self, cmd: str, **kwargs) -> int:
        """Write one request without waiting for the response (for VecEnv pipelining)."""
        self.start()
        assert self.proc and self.proc.stdin
        if self._pending:
            raise RpcError("send() called while a response is still pending")
        self._id += 1
        msg = {"id": self._id, "cmd": cmd, **kwargs}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        self._pending = True
        return self._id

    def recv(self, cmd: str = "step") -> Any:
        """Read the response for the last send()."""
        if not self._pending:
            raise RpcError("recv() with no pending send")
        try:
            resp = self._readline_json(timeout_s=self._timeout_for(cmd))
        except (RpcTimeout, RpcDied) as e:
            self._pending = False
            raise
        self._pending = False
        if not resp.get("ok"):
            raise RpcError(resp.get("error", "rpc error"))
        return resp["result"]

    def call(self, cmd: str, **kwargs) -> Any:
        self.send(cmd, **kwargs)
        return self.recv(cmd=cmd)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if not self._pending:
                self.call("quit")
        except Exception:
            pass
        self._kill_child()


class TsRpcPoolClient:
    """One `rpc_pool.ts` process with N worker_thread slots + optional fd-3 obs."""

    def __init__(
        self,
        slots: int = 1,
        map_name: str = "onion",
        stride: int = DEFAULT_STRIDE,
        difficulty: str = "Easy",
        nations: int = 3,
        bots: int = 2,
        max_ticks: int = 30_000,
        reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
        step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
    ):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = slots
        self.map_name = map_name
        self.stride = stride
        self.difficulty = difficulty
        self.nations = nations
        self.bots = bots
        self.max_ticks = max_ticks
        self.reset_timeout_s = reset_timeout_s
        self.step_timeout_s = step_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._stderr_path: Optional[Path] = None
        self._stderr_fh: Any = None
        self._obs_r: Optional[int] = None
        self._binary = False
        self._pending = 0

    def start(self) -> None:
        if self.proc is not None:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._stderr_path = LOG_DIR / f"rpc_pool_{os.getpid()}_{id(self)}.log"
        self._stderr_fh = open(self._stderr_path, "ab", buffering=0)

        obs_r, obs_w = os.pipe()
        os.set_inheritable(obs_w, True)
        env = os.environ.copy()
        env["OPENFRONT_OBS_FD"] = str(obs_w)

        cmd = [
            "node",
            "--import",
            str(ROOT / "node_modules/tsx/dist/loader.mjs"),
            str(ROOT / "ai/ts/rpc_pool.ts"),
            "--slots",
            str(self.slots),
            "--map",
            self.map_name,
            "--stride",
            str(self.stride),
            "--difficulty",
            self.difficulty,
            "--nations",
            str(self.nations),
            "--bots",
            str(self.bots),
            "--max-ticks",
            str(self.max_ticks),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            text=True,
            bufsize=1,
            env=env,
            pass_fds=(obs_w,),
            close_fds=True,
        )
        os.close(obs_w)
        self._obs_r = obs_r
        ready = self._readline_json(timeout_s=self.reset_timeout_s)
        if not ready.get("ready"):
            raise RpcDied(f"rpc pool not ready: {ready}")
        self._binary = bool(ready.get("binaryFd"))

    def _kill_child(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self.proc = None
        self._pending = 0
        if self._obs_r is not None:
            try:
                os.close(self._obs_r)
            except Exception:
                pass
            self._obs_r = None
        self._binary = False
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None

    def _attach_error(self, exc: Exception) -> Exception:
        tail = (
            _tail_file(self._stderr_path) if self._stderr_path is not None else ""
        )
        if not tail:
            return exc
        msg = f"{exc} | rpc stderr tail:\n{tail}"
        if isinstance(exc, RpcTimeout):
            return RpcTimeout(msg)
        if isinstance(exc, RpcDied):
            return RpcDied(msg)
        return RpcError(msg)

    def _readline_json(self, timeout_s: float) -> dict:
        assert self.proc and self.proc.stdout
        deadline = time.monotonic() + timeout_s
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_child()
                    raise self._attach_error(RpcTimeout(f"rpc timeout after {timeout_s}s"))
                events = sel.select(timeout=min(remaining, 1.0))
                if not events:
                    if self.proc.poll() is not None:
                        raise self._attach_error(
                            RpcDied(f"rpc exited {self.proc.returncode}")
                        )
                    continue
                line = self.proc.stdout.readline()
                if not line:
                    raise self._attach_error(RpcDied("rpc died"))
                line = line.strip()
                if not line.startswith("{"):
                    continue
                return json.loads(line)
        finally:
            sel.close()

    def _read_exact(self, n: int, timeout_s: float) -> bytes:
        assert self._obs_r is not None
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        sel = selectors.DefaultSelector()
        sel.register(self._obs_r, selectors.EVENT_READ)
        try:
            while len(buf) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_child()
                    raise self._attach_error(
                        RpcTimeout(f"obs fd timeout after {timeout_s}s")
                    )
                events = sel.select(timeout=min(remaining, 1.0))
                if not events:
                    if self.proc is not None and self.proc.poll() is not None:
                        raise self._attach_error(
                            RpcDied(f"rpc exited {self.proc.returncode}")
                        )
                    continue
                chunk = os.read(self._obs_r, n - len(buf))
                if not chunk:
                    raise self._attach_error(RpcDied("obs fd closed"))
                buf.extend(chunk)
        finally:
            sel.close()
        return bytes(buf)

    def _read_frame(self, timeout_s: float) -> tuple[dict, dict]:
        header = self._read_exact(4, timeout_s)
        (nbytes,) = struct.unpack("<I", header)
        if nbytes <= 0 or nbytes > FRAME_BYTES * 4:
            raise RpcError(f"bad obs frame size {nbytes}")
        payload = self._read_exact(nbytes, timeout_s)
        return decode_obs_mask_frame(payload)

    def _timeout_for(self, cmd: str) -> float:
        if cmd in ("reset", "reset_batch", "configure"):
            return self.reset_timeout_s
        return self.step_timeout_s

    def send(self, cmd: str, **kwargs) -> int:
        self.start()
        assert self.proc and self.proc.stdin
        self._id += 1
        msg = {"id": self._id, "cmd": cmd, **kwargs}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        self._pending += 1
        return self._id

    def recv_raw(self, cmd: str = "step") -> dict:
        if self._pending <= 0:
            raise RpcError("recv() with no pending send")
        try:
            resp = self._readline_json(timeout_s=self._timeout_for(cmd))
        except (RpcTimeout, RpcDied):
            self._pending = max(0, self._pending - 1)
            raise
        self._pending = max(0, self._pending - 1)
        if not resp.get("ok"):
            raise RpcError(resp.get("error", "rpc error"))
        return resp

    def recv_slot_result(self, cmd: str = "step") -> tuple[dict, dict, dict, dict]:
        """Return (obs, mask, info, timing) for a single-slot reset/step."""
        resp = self.recv_raw(cmd=cmd)
        result = resp.get("result") or {}
        timeout = self._timeout_for(cmd)
        if self._binary and result.get("encoding") == "fd3":
            obs, mask = self._read_frame(timeout)
        elif "obs" in result:
            obs = _reshape_obs(result["obs"])
            mask = _parse_mask(result.get("mask"))
        else:
            raise RpcError(f"missing obs in result keys={list(result)}")
        info = dict(result.get("info") or {})
        timing = dict(result.get("timing") or {})
        return obs, mask, info, timing

    def recv_batch_results(
        self, n: int, cmd: str = "step_batch"
    ) -> list[tuple[dict, dict, dict, dict]]:
        resp = self.recv_raw(cmd=cmd)
        results = resp.get("results")
        if not isinstance(results, list) or len(results) != n:
            raise RpcError(f"batch results length mismatch: {results!r}")
        timeout = self._timeout_for(cmd)
        out: list[tuple[dict, dict, dict, dict]] = []
        for result in results:
            result = result or {}
            if self._binary and result.get("encoding") == "fd3":
                obs, mask = self._read_frame(timeout)
            elif "obs" in result:
                obs = _reshape_obs(result["obs"])
                mask = _parse_mask(result.get("mask"))
            else:
                raise RpcError(f"missing obs in batch result keys={list(result)}")
            info = dict(result.get("info") or {})
            timing = dict(result.get("timing") or {})
            out.append((obs, mask, info, timing))
        return out

    def call(self, cmd: str, **kwargs) -> Any:
        self.send(cmd, **kwargs)
        resp = self.recv_raw(cmd=cmd)
        if "results" in resp:
            return resp["results"]
        return resp.get("result")

    def configure(
        self,
        *,
        slot: int | None = None,
        map_name: str | None = None,
        nations: int | None = None,
        bots: int | None = None,
        difficulty: str | None = None,
    ) -> Any:
        payload: dict[str, Any] = {}
        if slot is not None:
            payload["slot"] = slot
        if map_name is not None:
            payload["map"] = map_name
            self.map_name = map_name
        if nations is not None:
            payload["nations"] = int(nations)
            self.nations = int(nations)
        if bots is not None:
            payload["bots"] = int(bots)
            self.bots = int(bots)
        if difficulty is not None:
            payload["difficulty"] = difficulty
            self.difficulty = difficulty
        return self.call("configure", **payload)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self._pending == 0:
                self.call("quit")
        except Exception:
            pass
        self._kill_child()


class OpenFrontEnv:
    """Single-env wrapper. Uses a 1-slot RPC pool when available; stub if use_ts=False."""

    def __init__(
        self,
        map_dir: str = "",
        seed: int = 0,
        difficulty: str = "Impossible",
        use_ts: bool = True,
        map_name: str = "world",
        stride: int = DEFAULT_STRIDE,
        nations: int = MAX_NATIONS,
        bots: int = 0,
        max_ticks: int = 30_000,
        reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
        step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
        max_restarts: int = MAX_RPC_RESTARTS,
        pool: TsRpcPoolClient | None = None,
        slot: int = 0,
    ):
        self.map_dir = map_dir
        self.seed = seed
        self.difficulty = difficulty
        self.map_name = map_name
        self.stride = stride
        self.nations = nations
        self.bots = bots
        self.max_ticks = max_ticks
        self.reset_timeout_s = reset_timeout_s
        self.step_timeout_s = step_timeout_s
        self.max_restarts = max_restarts
        self._pool: Optional[TsRpcPoolClient] = pool
        self._owns_pool = pool is None
        self._slot = slot
        self._use_ts = use_ts
        self._restart_count = 0
        self._stub_tick = 0
        self._stub_tiles = 50
        self._stub_troops = 25000.0
        self._stub_gold = 0
        self._last_mask = _parse_mask(None)
        self.last_timing: dict = {}

    def _make_pool(self) -> TsRpcPoolClient:
        return TsRpcPoolClient(
            slots=1,
            map_name=self.map_name,
            stride=self.stride,
            difficulty=self.difficulty,
            nations=self.nations,
            bots=self.bots,
            max_ticks=self.max_ticks,
            reset_timeout_s=self.reset_timeout_s,
            step_timeout_s=self.step_timeout_s,
        )

    def _ensure(self) -> None:
        if not self._use_ts:
            return
        if self._pool is None:
            self._pool = self._make_pool()
            self._owns_pool = True
            self._pool.start()

    def _restart_and_recover(self, seed: int | None = None) -> None:
        self._restart_count += 1
        if self._restart_count > self.max_restarts:
            raise RpcError(
                f"RPC restart budget exhausted ({self.max_restarts}); giving up"
            )
        print(
            f"[openfront_ai] RPC restart {self._restart_count}/{self.max_restarts}",
            file=sys.stderr,
            flush=True,
        )
        if self._pool is not None and self._owns_pool:
            self._pool.close()
            self._pool = None
        if self._owns_pool:
            self._pool = self._make_pool()
            self._pool.start()
        elif self._pool is not None:
            # Shared pool: reconfigure this slot after supervisor restart by owner.
            self._pool.configure(
                slot=self._slot,
                map_name=self.map_name,
                nations=self.nations,
                bots=self.bots,
                difficulty=self.difficulty,
            )
        if seed is not None:
            self.seed = seed

    def reset(self, seed: int | None = None) -> tuple[dict, dict, dict]:
        if seed is not None:
            self.seed = seed
        if not self._use_ts:
            return self._stub_reset()
        self._ensure()
        assert self._pool is not None
        try:
            self._pool.send("reset", slot=self._slot, seed=self.seed)
            obs, mask, info, timing = self._pool.recv_slot_result(cmd="reset")
            self._restart_count = 0
        except (RpcTimeout, RpcDied, RpcError) as e:
            print(f"[openfront_ai] RPC failure on reset: {e}", file=sys.stderr, flush=True)
            self._restart_and_recover(seed=self.seed)
            assert self._pool is not None
            self._pool.send("reset", slot=self._slot, seed=self.seed)
            obs, mask, info, timing = self._pool.recv_slot_result(cmd="reset")
            self._restart_count = 0
        self._last_mask = mask
        self.last_timing = timing
        return obs, mask, info

    def _stub_reset(self) -> tuple[dict, dict, dict]:
        self._stub_tick = 0
        self._stub_tiles = 50
        self._stub_troops = 25000.0
        self._stub_gold = 0
        self._last_mask = _parse_mask(None)
        self._last_mask["actionType"][2] = True
        info = {
            "tick": 0,
            "done": False,
            "reward": 0.0,
            "winner": None,
            "agentTiles": self._stub_tiles,
            "agentTroops": self._stub_troops,
            "agentGold": self._stub_gold,
            "agentCities": 0,
            "nationTiles": 50,
            "troopCap": 100000.0,
            "troopRatio": self._stub_troops / 100000.0,
            "nationTroops": 25000.0,
            "opponentsAlive": 72,
            "enemyTilesTotal": 3600,
            "enemyTroopsTotal": 25000.0 * 72,
            "strongestEnemyTiles": 50,
            "strongestEnemyTroops": 25000.0,
            "placement": 1,
            "agentBoats": 0,
            "opponentIds": [f"nation-{i}" for i in range(72)],
        }
        return self._stub_obs(), self._last_mask, info

    def _stub_obs(self) -> dict:
        rng = np.random.default_rng(self.seed + self._stub_tick)
        return {
            "global": rng.random((GLOBAL_C, GLOBAL_H, GLOBAL_W), dtype=np.float32)
            * 0.01,
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
        if not self._use_ts:
            return self._stub_step(atype)
        self._ensure()
        assert self._pool is not None
        try:
            self._pool.send("step", slot=self._slot, action=_action_dict(action))
            obs, mask, info, timing = self._pool.recv_slot_result(cmd="step")
            self._restart_count = 0
        except (RpcTimeout, RpcDied, RpcError) as e:
            print(f"[openfront_ai] RPC failure on step: {e}", file=sys.stderr, flush=True)
            self._restart_and_recover(seed=self.seed)
            raise RpcError(f"abandoned in-flight step after restart: {e}") from e
        self._last_mask = mask
        self.last_timing = timing
        return StepResult(
            obs,
            float(info.get("reward", 0.0)),
            bool(info.get("done", False)),
            info,
            self._last_mask,
            timing=self.last_timing,
        )

    def _stub_step(self, atype: int) -> StepResult:
        self._stub_tick += self.stride
        if atype == 2:
            self._stub_tiles += 3
            self._stub_troops *= 0.98
        self._stub_gold += 500
        self._stub_troops += 50
        done = self._stub_tick > 2000 or self._stub_tiles > 8000
        reward = (5.0 if atype == 2 else 0.0) + 0.05
        if done:
            reward += 10.0 if self._stub_tiles > 4000 else -10.0
        return StepResult(
            self._stub_obs(),
            reward,
            done,
            {
                "tick": self._stub_tick,
                "stub": True,
                "agentTiles": self._stub_tiles,
                "troopRatio": self._stub_troops / 100000.0,
                "troopCap": 100000.0,
            },
            self._last_mask,
        )

    def legal_action_types(self) -> np.ndarray:
        return self._last_mask["actionType"].copy()

    def legal_mask(self) -> dict:
        return {k: v.copy() for k, v in self._last_mask.items()}

    def set_difficulty(self, difficulty: str) -> None:
        self.set_scale(difficulty=difficulty)

    def set_scale(
        self,
        *,
        map_name: str | None = None,
        nations: int | None = None,
        bots: int | None = None,
        difficulty: str | None = None,
    ) -> None:
        """Update map / opponent counts / difficulty via pool configure (no respawn)."""
        changed = False
        if map_name is not None and map_name != self.map_name:
            self.map_name = map_name
            changed = True
        if nations is not None and nations != self.nations:
            self.nations = int(nations)
            changed = True
        if bots is not None and bots != self.bots:
            self.bots = int(bots)
            changed = True
        if difficulty is not None and difficulty != self.difficulty:
            self.difficulty = difficulty
            changed = True
        if not changed:
            return
        if self._pool is not None:
            self._pool.configure(
                slot=self._slot if not self._owns_pool else None,
                map_name=self.map_name,
                nations=self.nations,
                bots=self.bots,
                difficulty=self.difficulty,
            )

    def close(self) -> None:
        if self._pool is not None and self._owns_pool:
            self._pool.close()
            self._pool = None


class VecEnv:
    """Parallel envs on one TsRpcPoolClient (N worker threads)."""

    def __init__(
        self,
        n: int,
        *,
        map_name: str = "world",
        seed: int = 0,
        stride: int = DEFAULT_STRIDE,
        use_ts: bool = True,
        difficulty: str = "Impossible",
        nations: int = MAX_NATIONS,
        bots: int = 0,
        max_ticks: int = 30_000,
        reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
        step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
        max_restarts: int = MAX_RPC_RESTARTS,
    ):
        if n < 1:
            raise ValueError("VecEnv requires n >= 1")
        self.n = n
        self.map_name = map_name
        self.seed = seed
        self.stride = stride
        self.difficulty = difficulty
        self.nations = nations
        self.bots = bots
        self.max_ticks = max_ticks
        self.reset_timeout_s = reset_timeout_s
        self.step_timeout_s = step_timeout_s
        self.max_restarts = max_restarts
        self._use_ts = use_ts
        self._pool: Optional[TsRpcPoolClient] = None
        self._restart_count = 0
        # Lightweight per-slot facades (shared pool) for API compatibility.
        if use_ts:
            self._pool = TsRpcPoolClient(
                slots=n,
                map_name=map_name,
                stride=stride,
                difficulty=difficulty,
                nations=nations,
                bots=bots,
                max_ticks=max_ticks,
                reset_timeout_s=reset_timeout_s,
                step_timeout_s=step_timeout_s,
            )
            self.envs = [
                OpenFrontEnv(
                    seed=seed + i,
                    difficulty=difficulty,
                    use_ts=True,
                    map_name=map_name,
                    stride=stride,
                    nations=nations,
                    bots=bots,
                    max_ticks=max_ticks,
                    reset_timeout_s=reset_timeout_s,
                    step_timeout_s=step_timeout_s,
                    max_restarts=max_restarts,
                    pool=self._pool,
                    slot=i,
                )
                for i in range(n)
            ]
        else:
            self.envs = [
                OpenFrontEnv(
                    seed=seed + i,
                    difficulty=difficulty,
                    use_ts=False,
                    map_name=map_name,
                    stride=stride,
                    nations=nations,
                    bots=bots,
                    max_ticks=max_ticks,
                )
                for i in range(n)
            ]
        self._obs: list[dict] = []
        self._masks: list[dict] = []
        self._infos: list[dict] = []

    def _ensure_pool(self) -> TsRpcPoolClient:
        assert self._pool is not None
        self._pool.start()
        return self._pool

    def _restart_pool(self) -> None:
        self._restart_count += 1
        if self._restart_count > self.max_restarts:
            raise RpcError(
                f"RPC restart budget exhausted ({self.max_restarts}); giving up"
            )
        print(
            f"[VecEnv] RPC pool restart {self._restart_count}/{self.max_restarts}",
            file=sys.stderr,
            flush=True,
        )
        if self._pool is not None:
            self._pool.close()
        self._pool = TsRpcPoolClient(
            slots=self.n,
            map_name=self.map_name,
            stride=self.stride,
            difficulty=self.difficulty,
            nations=self.nations,
            bots=self.bots,
            max_ticks=self.max_ticks,
            reset_timeout_s=self.reset_timeout_s,
            step_timeout_s=self.step_timeout_s,
        )
        self._pool.start()
        for i, env in enumerate(self.envs):
            env._pool = self._pool
            env._owns_pool = False
            env._slot = i
            env.map_name = self.map_name
            env.nations = self.nations
            env.bots = self.bots
            env.difficulty = self.difficulty

    def reset(self, seeds: list[int] | int) -> tuple[list[dict], list[dict], list[dict]]:
        if isinstance(seeds, int):
            seed_list = [seeds + i for i in range(self.n)]
        else:
            seed_list = list(seeds)
            if len(seed_list) != self.n:
                raise ValueError("seeds length must match num envs")

        if not self._use_ts:
            self._obs, self._masks, self._infos = [], [], []
            for env, seed in zip(self.envs, seed_list):
                o, m, info = env.reset(seed)
                self._obs.append(o)
                self._masks.append(m)
                self._infos.append(info)
            return self._obs, self._masks, self._infos

        for env, seed in zip(self.envs, seed_list):
            env.seed = seed

        pool = self._ensure_pool()
        try:
            pool.send("reset_batch", seeds=seed_list)
            decoded = pool.recv_batch_results(self.n, cmd="reset_batch")
            self._restart_count = 0
        except (RpcTimeout, RpcDied, RpcError) as e:
            print(f"[VecEnv] reset_batch failed: {e}", file=sys.stderr, flush=True)
            self._restart_pool()
            pool = self._ensure_pool()
            pool.send("reset_batch", seeds=seed_list)
            decoded = pool.recv_batch_results(self.n, cmd="reset_batch")
            self._restart_count = 0

        self._obs, self._masks, self._infos = [], [], []
        for env, (obs, mask, info, timing) in zip(self.envs, decoded):
            env._last_mask = mask
            env.last_timing = timing
            self._obs.append(obs)
            self._masks.append(mask)
            self._infos.append(info)
        return self._obs, self._masks, self._infos

    def step(
        self, actions: list[Tuple[int, int, int, int, int, int]]
    ) -> list[StepResult]:
        if len(actions) != self.n:
            raise ValueError("actions length must match num envs")

        if not self._use_ts:
            return [env.step(a) for env, a in zip(self.envs, actions)]

        pool = self._ensure_pool()
        try:
            pool.send(
                "step_batch",
                actions=[_action_dict(a) for a in actions],
            )
            decoded = pool.recv_batch_results(self.n, cmd="step_batch")
            self._restart_count = 0
        except (RpcTimeout, RpcDied, RpcError) as e:
            print(f"[VecEnv] step_batch failed: {e}", file=sys.stderr, flush=True)
            results = [
                StepResult(
                    {
                        "global": np.zeros(
                            (GLOBAL_C, GLOBAL_H, GLOBAL_W), dtype=np.float32
                        ),
                        "local": np.zeros(
                            (LOCAL_C, LOCAL_H, LOCAL_W), dtype=np.float32
                        ),
                        "vector": np.zeros(VECTOR_DIM, dtype=np.float32),
                    },
                    0.0,
                    True,
                    {"rpc_error": True, "error": str(e), "done": True, "reward": 0.0},
                    env._last_mask,
                )
                for env in self.envs
            ]
            try:
                self._restart_pool()
            except Exception:
                pass
            self._obs = [r.obs for r in results]
            self._masks = [r.mask for r in results]
            self._infos = [r.info for r in results]
            return results

        results: list[StepResult] = []
        for env, (obs, mask, info, timing) in zip(self.envs, decoded):
            env._last_mask = mask
            env.last_timing = timing
            results.append(
                StepResult(
                    obs,
                    float(info.get("reward", 0.0)),
                    bool(info.get("done", False)),
                    info,
                    env._last_mask,
                    timing=timing,
                )
            )
        self._obs = [r.obs for r in results]
        self._masks = [r.mask for r in results]
        self._infos = [r.info for r in results]
        return results

    def set_difficulty(self, difficulty: str) -> None:
        self.set_scale(difficulty=difficulty)

    def set_scale(
        self,
        *,
        map_name: str | None = None,
        nations: int | None = None,
        bots: int | None = None,
        difficulty: str | None = None,
    ) -> None:
        changed = False
        if map_name is not None and map_name != self.map_name:
            self.map_name = map_name
            changed = True
        if nations is not None and nations != self.nations:
            self.nations = int(nations)
            changed = True
        if bots is not None and bots != self.bots:
            self.bots = int(bots)
            changed = True
        if difficulty is not None and difficulty != self.difficulty:
            self.difficulty = difficulty
            changed = True
        for env in self.envs:
            if map_name is not None:
                env.map_name = self.map_name
            if nations is not None:
                env.nations = self.nations
            if bots is not None:
                env.bots = self.bots
            if difficulty is not None:
                env.difficulty = self.difficulty
        if not changed:
            return
        if self._pool is not None:
            self._pool.start()
            self._pool.configure(
                map_name=self.map_name,
                nations=self.nations,
                bots=self.bots,
                difficulty=self.difficulty,
            )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        for env in self.envs:
            env._pool = None


def make_vec_env(
    n: int,
    map_name: str = "world",
    seed: int = 0,
    stride: int = DEFAULT_STRIDE,
    use_ts: bool = True,
    difficulty: str = "Impossible",
    nations: int = MAX_NATIONS,
    bots: int = 0,
    max_ticks: int = 30_000,
) -> list[OpenFrontEnv]:
    """Backward-compat: list of envs. Prefer make_parallel_env for training."""
    if use_ts and n > 1:
        # Shared pool under the hood via VecEnv facades.
        vec = make_parallel_env(
            n,
            map_name=map_name,
            seed=seed,
            stride=stride,
            use_ts=True,
            difficulty=difficulty,
            nations=nations,
            bots=bots,
            max_ticks=max_ticks,
        )
        return vec.envs
    return [
        OpenFrontEnv(
            map_name=map_name,
            seed=seed + i,
            use_ts=use_ts,
            stride=stride,
            difficulty=difficulty,
            nations=nations,
            bots=bots,
            max_ticks=max_ticks,
        )
        for i in range(n)
    ]


def make_parallel_env(
    n: int,
    map_name: str = "world",
    seed: int = 0,
    stride: int = DEFAULT_STRIDE,
    use_ts: bool = True,
    difficulty: str = "Impossible",
    nations: int = MAX_NATIONS,
    bots: int = 0,
    max_ticks: int = 30_000,
) -> VecEnv:
    return VecEnv(
        n,
        map_name=map_name,
        seed=seed,
        stride=stride,
        use_ts=use_ts,
        difficulty=difficulty,
        nations=nations,
        bots=bots,
        max_ticks=max_ticks,
    )
