"""Unit tests for env wire format, VecEnv pipelining helpers, and RPC errors."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openfront_ai.env import (
    GLOBAL_C,
    GLOBAL_H,
    GLOBAL_W,
    LOCAL_C,
    LOCAL_H,
    LOCAL_W,
    VECTOR_DIM,
    OpenFrontEnv,
    RpcDied,
    RpcTimeout,
    TsRpcClient,
    VecEnv,
    _reshape_obs,
)


def test_reshape_obs_base64_roundtrip():
    g = np.linspace(0, 1, GLOBAL_C * GLOBAL_H * GLOBAL_W, dtype=np.float32)
    l = np.linspace(0, 1, LOCAL_C * LOCAL_H * LOCAL_W, dtype=np.float32)
    v = np.linspace(0, 1, VECTOR_DIM, dtype=np.float32)
    raw = {
        "encoding": "f32b64",
        "global": base64.b64encode(g.tobytes()).decode("ascii"),
        "local": base64.b64encode(l.tobytes()).decode("ascii"),
        "vector": base64.b64encode(v.tobytes()).decode("ascii"),
    }
    obs = _reshape_obs(raw)
    assert obs["global"].shape == (GLOBAL_C, GLOBAL_H, GLOBAL_W)
    assert obs["local"].shape == (LOCAL_C, LOCAL_H, LOCAL_W)
    assert obs["vector"].shape == (VECTOR_DIM,)
    np.testing.assert_allclose(obs["global"].ravel(), g, rtol=0, atol=0)
    np.testing.assert_allclose(obs["local"].ravel(), l, rtol=0, atol=0)
    np.testing.assert_allclose(obs["vector"], v, rtol=0, atol=0)


def test_reshape_obs_legacy_arrays():
    raw = {
        "global": [0.1] * (GLOBAL_C * GLOBAL_H * GLOBAL_W),
        "local": [0.2] * (LOCAL_C * LOCAL_H * LOCAL_W),
        "vector": [0.3] * VECTOR_DIM,
    }
    obs = _reshape_obs(raw)
    assert obs["global"].shape == (GLOBAL_C, GLOBAL_H, GLOBAL_W)
    assert float(obs["global"][0, 0, 0]) == pytest.approx(0.1)


def test_stub_env_opt_in_only():
    env = OpenFrontEnv(use_ts=False, seed=7)
    obs, mask, info = env.reset(7)
    assert obs["global"].shape[0] == GLOBAL_C
    assert mask["actionType"][0] or mask["actionType"][2]
    step = env.step((2, 0, 0, 0, 0, 0))
    assert isinstance(step.reward, float)
    env.close()


def test_ts_rpc_send_recv_pending_guard():
    client = TsRpcClient(map_name="plains")
    client.proc = MagicMock()
    client.proc.stdin = MagicMock()
    client._pending = False
    # Bypass start()
    client.start = lambda: None  # type: ignore
    client.send("ping")
    assert client._pending is True
    with pytest.raises(Exception):
        client.send("ping")


def test_vec_env_requires_matching_actions():
    vec = VecEnv(2, use_ts=False, seed=0)
    vec.reset([1, 2])
    with pytest.raises(ValueError):
        vec.step([(0, 0, 0, 0, 0, 0)])
    results = vec.step([(0, 0, 0, 0, 0, 0), (2, 0, 1, 1, 10, 0)])
    assert len(results) == 2
    vec.close()


def test_fd3_obs_mask_roundtrip():
    from openfront_ai.env import (
        FRAME_BYTES,
        NUM_ACTION_TYPES,
        NUM_BUILD,
        NUM_CELL,
        NUM_FRAC,
        NUM_TARGET,
        decode_obs_mask_frame,
        pack_obs_mask_frame,
    )

    rng = np.random.default_rng(0)
    obs = {
        "global": rng.random((GLOBAL_C, GLOBAL_H, GLOBAL_W), dtype=np.float32),
        "local": rng.random((LOCAL_C, LOCAL_H, LOCAL_W), dtype=np.float32),
        "vector": rng.random(VECTOR_DIM, dtype=np.float32),
    }
    mask = {
        "actionType": rng.integers(0, 2, NUM_ACTION_TYPES).astype(bool),
        "targetPlayer": rng.integers(0, 2, NUM_TARGET).astype(bool),
        "cell": rng.integers(0, 2, NUM_CELL).astype(bool),
        "troopFrac": rng.integers(0, 2, NUM_FRAC).astype(bool),
        "buildType": rng.integers(0, 2, NUM_BUILD).astype(bool),
    }
    packed = pack_obs_mask_frame(obs, mask)
    assert len(packed) == FRAME_BYTES
    obs2, mask2 = decode_obs_mask_frame(packed)
    np.testing.assert_allclose(obs2["global"], obs["global"], rtol=0, atol=0)
    np.testing.assert_allclose(obs2["local"], obs["local"], rtol=0, atol=0)
    np.testing.assert_allclose(obs2["vector"], obs["vector"], rtol=0, atol=0)
    for k in mask:
        np.testing.assert_array_equal(mask2[k], mask[k])


def test_pool_client_send_recv_slot_ordering():
    """TsRpcPoolClient tracks pending count so pipelined slot sends stay ordered."""
    from openfront_ai.env import TsRpcPoolClient

    client = TsRpcPoolClient(slots=2, map_name="plains")
    client.proc = MagicMock()
    client.proc.stdin = MagicMock()
    client.start = lambda: None  # type: ignore
    client._binary = False
    ids = []
    ids.append(client.send("step", slot=0, action={"actionType": 0}))
    ids.append(client.send("step", slot=1, action={"actionType": 2}))
    assert ids == [1, 2]
    assert client._pending == 2
    # Simulate JSON responses without obs fd — inject f32b64 results.
    g = base64.b64encode(
        np.zeros(GLOBAL_C * GLOBAL_H * GLOBAL_W, dtype=np.float32).tobytes()
    ).decode("ascii")
    l = base64.b64encode(
        np.zeros(LOCAL_C * LOCAL_H * LOCAL_W, dtype=np.float32).tobytes()
    ).decode("ascii")
    v = base64.b64encode(np.zeros(VECTOR_DIM, dtype=np.float32).tobytes()).decode(
        "ascii"
    )
    fake_results = [
        {
            "ok": True,
            "result": {
                "encoding": "f32b64",
                "obs": {"encoding": "f32b64", "global": g, "local": l, "vector": v},
                "mask": {
                    "actionType": [True] + [False] * 9,
                    "targetPlayer": [True] * 73,
                    "cell": [True] * (32 * 16),
                    "troopFrac": [True] * 100,
                    "buildType": [False] * 10,
                },
                "info": {"reward": 0.0, "done": False, "slot": 0},
                "timing": {},
            },
        },
        {
            "ok": True,
            "result": {
                "encoding": "f32b64",
                "obs": {"encoding": "f32b64", "global": g, "local": l, "vector": v},
                "mask": {
                    "actionType": [True] + [False] * 9,
                    "targetPlayer": [True] * 73,
                    "cell": [True] * (32 * 16),
                    "troopFrac": [True] * 100,
                    "buildType": [False] * 10,
                },
                "info": {"reward": 1.0, "done": False, "slot": 1},
                "timing": {},
            },
        },
    ]
    client._readline_json = lambda timeout_s=1.0: fake_results.pop(0)  # type: ignore
    o0, m0, i0, _ = client.recv_slot_result(cmd="step")
    o1, m1, i1, _ = client.recv_slot_result(cmd="step")
    assert i0["slot"] == 0
    assert i1["slot"] == 1
    assert o0["global"].shape == (GLOBAL_C, GLOBAL_H, GLOBAL_W)
    assert client._pending == 0
    assert m0["actionType"][0]
    assert m1["actionType"][0]


def test_rpc_error_hierarchy():
    assert issubclass(RpcTimeout, Exception)
    assert issubclass(RpcDied, Exception)
    e = RpcTimeout("boom")
    assert "boom" in str(e)
