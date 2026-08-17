#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX for the Rust/Node inference client."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_ROOT))

from openfront_ai.policy import MacroMicroPolicy, export_onnx, load_policy_state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = MacroMicroPolicy()
    state = torch.load(args.ckpt, map_location="cpu")
    blob = state["model"] if isinstance(state, dict) and "model" in state else state
    load_policy_state_dict(model, blob)
    export_onnx(model, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
