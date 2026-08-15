"""Hierarchical macro/micro policy (Impala-style CNN + MLP + factorized heads)."""

from __future__ import annotations

from typing import Dict, Tuple

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


class MacroMicroPolicy(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
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
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.macro_head = nn.Linear(hidden, NUM_MACRO_GOALS)
        self.action_type = nn.Linear(hidden + NUM_MACRO_GOALS, NUM_ACTION_TYPES)
        self.target_player = nn.Linear(hidden, NUM_TARGET)
        self.cell = nn.Linear(hidden, COARSE_H * COARSE_W)
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
        hx: Tuple[torch.Tensor, torch.Tensor] | None = None,
        action_mask: torch.Tensor | None = None,
    ):
        """
        obs tensors: global (B,C,H,W), local (B,C,H,W), vector (B,D)
        action_mask: (B, NUM_ACTION_TYPES) bool — True = legal
        """
        h = self.encode(obs)
        # Single-step LSTM
        h_seq = h.unsqueeze(1)
        if hx is None:
            out, hx = self.lstm(h_seq)
        else:
            out, hx = self.lstm(h_seq, hx)
        h = out.squeeze(1)

        macro_logits = self.macro_head(h)
        macro_oh = F.softmax(macro_logits, dim=-1)
        h_micro = torch.cat([h, macro_oh], dim=-1)

        at_logits = self.action_type(h_micro)
        if action_mask is not None:
            at_logits = at_logits.masked_fill(~action_mask, -1e9)

        return {
            "macro_logits": macro_logits,
            "action_type_logits": at_logits,
            "target_logits": self.target_player(h),
            "cell_logits": self.cell(h),
            "troop_logits": self.troop_frac(h),
            "build_logits": self.build_type(h),
            "value": self.value(h).squeeze(-1),
        }, hx

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        out, _ = self.forward(obs, action_mask=action_mask)

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
        return (at, tgt, cx, cy, frac, build), out["value"]


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
