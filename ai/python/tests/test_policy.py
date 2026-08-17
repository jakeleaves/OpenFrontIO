"""Unit tests for factorized policy masking and joint log-prob."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openfront_ai.env import (
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
)
from openfront_ai.policy import MacroMicroPolicy, _mix_uniform_legal


def _obs(b: int = 2):
    return {
        "global": torch.zeros(b, GLOBAL_C, GLOBAL_H, GLOBAL_W),
        "local": torch.zeros(b, LOCAL_C, LOCAL_H, LOCAL_W),
        "vector": torch.zeros(b, VECTOR_DIM),
    }


def test_masked_action_type_never_samples_illegal():
    model = MacroMicroPolicy()
    model.eval()
    mask = torch.zeros(2, NUM_ACTION_TYPES, dtype=torch.bool)
    mask[:, 0] = True  # NOOP only
    masks = {
        "actionType": mask,
        "targetPlayer": torch.ones(2, NUM_TARGET, dtype=torch.bool),
        "cell": torch.ones(2, NUM_CELL, dtype=torch.bool),
        "troopFrac": torch.ones(2, NUM_FRAC, dtype=torch.bool),
        "buildType": torch.zeros(2, NUM_BUILD, dtype=torch.bool),
    }
    with torch.no_grad():
        for _ in range(20):
            (at, *_), _, _, _ = model.act(_obs(), masks=masks)
            assert torch.all(at == 0)


def test_joint_log_prob_finite():
    model = MacroMicroPolicy()
    obs = _obs(4)
    masks = {
        "actionType": torch.ones(4, NUM_ACTION_TYPES, dtype=torch.bool),
        "targetPlayer": torch.ones(4, NUM_TARGET, dtype=torch.bool),
        "cell": torch.ones(4, NUM_CELL, dtype=torch.bool),
        "troopFrac": torch.ones(4, NUM_FRAC, dtype=torch.bool),
        "buildType": torch.ones(4, NUM_BUILD, dtype=torch.bool),
    }
    actions = torch.zeros(4, 6, dtype=torch.long)
    actions[:, 0] = 2
    logp, value, ent, _, _ = model.evaluate_actions(obs, actions, masks=masks)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(ent)


def test_noop_log_prob_ignores_parameter_heads():
    """Irrelevant random heads must not add PPO ratio noise for NOOP."""
    batch = 3
    out = {
        "action_type_logits": torch.randn(batch, NUM_ACTION_TYPES),
        "target_logits": torch.randn(batch, NUM_TARGET),
        "cell_logits": torch.randn(batch, NUM_CELL),
        "troop_logits": torch.randn(batch, NUM_FRAC),
        "build_logits": torch.randn(batch, NUM_BUILD),
    }
    at = torch.zeros(batch, dtype=torch.long)
    zeros = torch.zeros(batch, dtype=torch.long)
    first = MacroMicroPolicy.joint_log_prob(out, at, zeros, zeros, zeros, zeros)
    for key in ("target_logits", "cell_logits", "troop_logits", "build_logits"):
        out[key] = torch.randn_like(out[key]) * 100
    second = MacroMicroPolicy.joint_log_prob(out, at, zeros, zeros, zeros, zeros)
    assert torch.allclose(first, second)


def test_action_exploration_floor_only_covers_legal_actions():
    logits = torch.tensor([[20.0, -20.0, -20.0]])
    mask = torch.tensor([[True, True, False]])
    mixed = _mix_uniform_legal(logits, mask, epsilon=0.05)
    probs = mixed.softmax(dim=-1)
    assert probs[0, 1] >= 0.024
    assert probs[0, 2] == 0


def test_partial_load_skips_resized_target_head():
    """World FFA enlarges NUM_TARGET; legacy 1v1 heads must soft-load."""
    from openfront_ai.policy import load_policy_state_dict

    model = MacroMicroPolicy()
    state = model.state_dict()
    # Simulate a 1v1 checkpoint with a 2-way target head.
    legacy = {k: v.clone() for k, v in state.items()}
    for k, v in list(legacy.items()):
        if "target" in k and v.ndim >= 1 and v.shape[-1] == NUM_TARGET:
            legacy[k] = v[..., :2].contiguous()
    missing, skipped = load_policy_state_dict(model, legacy, log=False)
    assert any("target" in s for s in skipped)
    assert model.target_player.out_features == NUM_TARGET


def test_num_target_is_world_ffa():
    assert NUM_TARGET == 73

    from openfront_ai.demos import stack_transitions

    model = MacroMicroPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    rows = []
    for _ in range(16):
        rows.append(
            {
                "obs": {
                    "global": np.zeros((GLOBAL_C, GLOBAL_H, GLOBAL_W), np.float32),
                    "local": np.zeros((LOCAL_C, LOCAL_H, LOCAL_W), np.float32),
                    "vector": np.zeros(VECTOR_DIM, np.float32),
                },
                "mask": {
                    "actionType": np.ones(NUM_ACTION_TYPES, dtype=bool),
                    "targetPlayer": np.ones(NUM_TARGET, dtype=bool),
                    "cell": np.ones(NUM_CELL, dtype=bool),
                    "troopFrac": np.ones(NUM_FRAC, dtype=bool),
                    "buildType": np.ones(NUM_BUILD, dtype=bool),
                },
                "action": (2, 0, 16, 8, 1, 0),
                "reward": 1.0,
                "done": False,
            }
        )
    batch = stack_transitions(rows)
    import torch.nn.functional as F
    from openfront_ai.env import COARSE_W

    first = None
    last = None
    for _ in range(30):
        obs = {
            "global": torch.from_numpy(batch["global"]),
            "local": torch.from_numpy(batch["local"]),
            "vector": torch.from_numpy(batch["vector"]),
        }
        masks = {
            "actionType": torch.from_numpy(batch["actionType_mask"]),
            "targetPlayer": torch.from_numpy(batch["targetPlayer_mask"]),
            "cell": torch.from_numpy(batch["cell_mask"]),
            "troopFrac": torch.from_numpy(batch["troopFrac_mask"]),
            "buildType": torch.from_numpy(batch["buildType_mask"]),
        }
        actions = torch.from_numpy(batch["action"])
        out, _ = model.forward(obs, masks=masks)
        cell = actions[:, 2] + actions[:, 3] * COARSE_W
        loss = (
            F.cross_entropy(out["action_type_logits"], actions[:, 0])
            + F.cross_entropy(out["target_logits"], actions[:, 1])
            + F.cross_entropy(out["cell_logits"], cell)
            + F.cross_entropy(out["troop_logits"], actions[:, 4])
            + F.cross_entropy(out["build_logits"], actions[:, 5])
        )
        if first is None:
            first = float(loss.item())
        last = float(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert last < first
