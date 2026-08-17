"""Unit tests for PPO clipped value loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import clipped_value_loss


def test_clipped_value_loss_matches_mse_when_within_clip():
    value = torch.tensor([1.0, 2.0, 3.0])
    old = torch.tensor([1.05, 1.9, 3.1])
    returns = torch.tensor([1.2, 2.2, 2.8])
    got = clipped_value_loss(value, old, returns, clip=0.2)
    expect = F.mse_loss(value, returns)
    assert torch.allclose(got, expect)


def test_clipped_value_loss_blocks_huge_jumps_toward_returns():
    value = torch.tensor([100.0])
    old = torch.tensor([1.0])
    returns = torch.tensor([100.0])
    got = clipped_value_loss(value, old, returns, clip=0.2)
    unclipped = F.mse_loss(value, returns)
    v_clipped = torch.tensor([1.2])
    expect = F.mse_loss(v_clipped, returns)
    assert got > unclipped
    assert torch.allclose(got, expect)
