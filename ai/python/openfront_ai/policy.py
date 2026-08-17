"""Hierarchical macro/micro policy (Impala-style CNN + MLP + factorized heads)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import (
    COARSE_H,
    COARSE_W,
    GLOBAL_C,
    LOCAL_C,
    NUM_ACTION_TYPES,
    NUM_BUILD,
    NUM_FRAC,
    NUM_TARGET,
    VECTOR_DIM,
)

NUM_MACRO_GOALS = 5  # expand, crush, eco, nuke, defend
NUM_CELL = COARSE_H * COARSE_W
# Kept for call-site compatibility after dropping the LSTM (always None).
Hx = Tuple[torch.Tensor, torch.Tensor]


def load_policy_state_dict(
    model: nn.Module, state: dict, *, log: bool = True
) -> tuple[list[str], list[str]]:
    """Load weights, skipping size-mismatched keys (e.g. expanded troopFrac head)."""
    current = model.state_dict()
    filtered: dict = {}
    skipped: list[str] = []
    for k, v in state.items():
        if k not in current:
            skipped.append(k)
            continue
        if current[k].shape != v.shape:
            skipped.append(f"{k} {tuple(v.shape)}->{tuple(current[k].shape)}")
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if log and (skipped or missing):
        print(
            f"Partial policy load: skipped={len(skipped)} missing={len(missing)}",
            flush=True,
        )
        for s in skipped[:8]:
            print(f"  skip {s}", flush=True)
    return list(missing), skipped


class ImpalaCNN(nn.Module):
    def __init__(self, in_c: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask.bool(), -1e9)


def _mix_uniform_legal(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Keep a small probability floor over legal top-level actions.

    Returning log-probabilities keeps sampling and PPO evaluation exactly
    consistent while preventing irreversible NOOP collapse.
    """
    masked = _mask_logits(logits, mask)
    legal = (
        mask.bool()
        if mask is not None
        else torch.ones_like(masked, dtype=torch.bool)
    )
    policy_probs = F.softmax(masked, dim=-1)
    uniform = legal.float() / legal.sum(dim=-1, keepdim=True).clamp_min(1)
    probs = (1.0 - epsilon) * policy_probs + epsilon * uniform
    return torch.where(legal, probs.clamp_min(1e-12).log(), masked)


class MacroMicroPolicy(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.hidden = hidden
        self.global_enc = ImpalaCNN(GLOBAL_C, 256)
        self.local_enc = ImpalaCNN(LOCAL_C, 128)
        self.vec_mlp = nn.Sequential(
            nn.Linear(VECTOR_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(256 + 128 + 128, hidden),
            nn.ReLU(inplace=True),
        )
        # Feed-forward only: rollouts previously carried LSTM state but PPO
        # updates shuffled minibatches with hx=None, so the recurrent path
        # was never trained honestly.
        self.macro_head = nn.Linear(hidden, NUM_MACRO_GOALS)
        self.action_type = nn.Linear(hidden + NUM_MACRO_GOALS, NUM_ACTION_TYPES)
        self.target_player = nn.Linear(hidden, NUM_TARGET)
        self.cell = nn.Linear(hidden, NUM_CELL)
        self.troop_frac = nn.Linear(hidden, NUM_FRAC)
        self.build_type = nn.Linear(hidden, NUM_BUILD)
        self.value = nn.Linear(hidden, 1)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        g = self.global_enc(obs["global"])
        l = self.local_enc(obs["local"])
        v = self.vec_mlp(obs["vector"])
        return self.fuse(torch.cat([g, l, v], dim=-1))

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hx: Hx | None = None,
        masks: Dict[str, torch.Tensor] | None = None,
        action_mask: torch.Tensor | None = None,
    ):
        """
        obs tensors: global (B,C,H,W), local (B,C,H,W), vector (B,D)
        masks: optional dict with actionType/targetPlayer/cell/troopFrac/buildType bool tensors
        action_mask: legacy (B, NUM_ACTION_TYPES) — used if masks is None
        hx is ignored (API compatibility); always returns hx_out=None.
        """
        del hx  # feed-forward policy; callers may still pass hx=None
        h = self.encode(obs)

        macro_logits = self.macro_head(h)
        macro_oh = F.softmax(macro_logits, dim=-1)
        h_micro = torch.cat([h, macro_oh], dim=-1)

        at_mask = None
        tgt_mask = None
        cell_mask = None
        troop_mask = None
        build_mask = None
        if masks is not None:
            at_mask = masks.get("actionType")
            tgt_mask = masks.get("targetPlayer")
            cell_mask = masks.get("cell")
            troop_mask = masks.get("troopFrac")
            build_mask = masks.get("buildType")
        elif action_mask is not None:
            at_mask = action_mask

        return {
            "macro_logits": macro_logits,
            "action_type_logits": _mix_uniform_legal(
                self.action_type(h_micro), at_mask
            ),
            "target_logits": _mask_logits(self.target_player(h), tgt_mask),
            "cell_logits": _mask_logits(self.cell(h), cell_mask),
            "troop_logits": _mask_logits(self.troop_frac(h), troop_mask),
            "build_logits": _mask_logits(self.build_type(h), build_mask),
            "value": self.value(h).squeeze(-1),
        }, None

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        hx: Hx | None = None,
        masks: Dict[str, torch.Tensor] | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        out, _ = self.forward(obs, hx=hx, masks=masks, action_mask=action_mask)

        def sample(logits):
            if deterministic:
                return logits.argmax(dim=-1)
            return torch.distributions.Categorical(logits=logits).sample()

        at = sample(out["action_type_logits"])
        tgt = sample(out["target_logits"])
        cell = sample(out["cell_logits"])
        cx = cell % COARSE_W
        cy = cell // COARSE_W
        frac = sample(out["troop_logits"])
        build = sample(out["build_logits"])
        logp = self.joint_log_prob(out, at, tgt, cell, frac, build)
        return (at, tgt, cx, cy, frac, build), out["value"], logp, None

    @staticmethod
    def joint_log_prob(
        out: Dict[str, torch.Tensor],
        action_type: torch.Tensor,
        target: torch.Tensor,
        cell: torch.Tensor,
        troop: torch.Tensor,
        build: torch.Tensor,
    ) -> torch.Tensor:
        """Log-prob of the sampled action and only its causal parameters.

        Irrelevant heads must not enter PPO's importance ratio. For example,
        NOOP has no target/cell/troop/build parameters; including their random
        samples made the ratio noisy and let PPO optimize unrelated heads.
        """
        lp = torch.distributions.Categorical(logits=out["action_type_logits"]).log_prob(
            action_type
        )
        target_lp = torch.distributions.Categorical(
            logits=out["target_logits"]
        ).log_prob(target)
        cell_lp = torch.distributions.Categorical(logits=out["cell_logits"]).log_prob(cell)
        troop_lp = torch.distributions.Categorical(
            logits=out["troop_logits"]
        ).log_prob(troop)
        build_lp = torch.distributions.Categorical(
            logits=out["build_logits"]
        ).log_prob(build)

        uses_target = action_type == 2  # ATTACK
        uses_cell = (
            (action_type == 1)  # SPAWN
            | (action_type == 3)  # BOAT
            | (action_type == 6)  # BUILD
            | (action_type == 8)  # MOVE_WARSHIP
        )
        uses_troop = (action_type == 2) | (action_type == 3)
        uses_build = action_type == 6
        lp = lp + torch.where(uses_target, target_lp, torch.zeros_like(target_lp))
        lp = lp + torch.where(uses_cell, cell_lp, torch.zeros_like(cell_lp))
        lp = lp + torch.where(uses_troop, troop_lp, torch.zeros_like(troop_lp))
        lp = lp + torch.where(uses_build, build_lp, torch.zeros_like(build_lp))
        return lp

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        hx: Hx | None = None,
        masks: Dict[str, torch.Tensor] | None = None,
    ):
        """
        actions: (B, 6) = actionType, target, cellX, cellY, troopFrac, buildType
        """
        out, _ = self.forward(obs, hx=hx, masks=masks)
        at = actions[:, 0]
        tgt = actions[:, 1]
        cell = actions[:, 2] + actions[:, 3] * COARSE_W
        frac = actions[:, 4]
        build = actions[:, 5]
        logp = self.joint_log_prob(out, at, tgt, cell, frac, build)
        at_ent = torch.distributions.Categorical(
            logits=out["action_type_logits"]
        ).entropy()
        target_ent = torch.distributions.Categorical(
            logits=out["target_logits"]
        ).entropy()
        cell_ent = torch.distributions.Categorical(
            logits=out["cell_logits"]
        ).entropy()
        troop_ent = torch.distributions.Categorical(
            logits=out["troop_logits"]
        ).entropy()
        build_ent = torch.distributions.Categorical(
            logits=out["build_logits"]
        ).entropy()
        macro_ent = torch.distributions.Categorical(
            logits=out["macro_logits"]
        ).entropy()
        uses_target = at == 2
        uses_cell = (at == 1) | (at == 3) | (at == 6) | (at == 8)
        uses_troop = (at == 2) | (at == 3)
        uses_build = at == 6
        ent = (
            at_ent
            + target_ent * uses_target
            + cell_ent * uses_cell
            + troop_ent * uses_troop
            + build_ent * uses_build
            + 0.25 * macro_ent
        )
        return logp, out["value"], ent.mean(), out, None


def export_onnx(model: MacroMicroPolicy, path: str, opset: int = 17) -> None:
    model.eval()
    B = 1
    dummy = {
        "global": torch.zeros(B, GLOBAL_C, 64, 128),
        "local": torch.zeros(B, LOCAL_C, 64, 64),
        "vector": torch.zeros(B, VECTOR_DIM),
    }

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, g, l, v):
            out, _ = self.m.forward({"global": g, "local": l, "vector": v})
            return (
                out["action_type_logits"],
                out["target_logits"],
                out["cell_logits"],
                out["troop_logits"],
                out["build_logits"],
                out["value"],
            )

    wrap = Wrapper(model)
    torch.onnx.export(
        wrap,
        (dummy["global"], dummy["local"], dummy["vector"]),
        path,
        input_names=["global", "local", "vector"],
        output_names=[
            "action_type_logits",
            "target_logits",
            "cell_logits",
            "troop_logits",
            "build_logits",
            "value",
        ],
        dynamo=False,
        opset_version=opset,
    )
