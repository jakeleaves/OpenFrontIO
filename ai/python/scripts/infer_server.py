#!/usr/bin/env python3
"""HTTP inference sidecar for viz.ts — loads policy.pt (feed-forward).

  python ai/python/scripts/infer_server.py --ckpt ai/fixtures/checkpoints/policy.pt --port 9101
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = Path(__file__).resolve().parents[2]
ROOT = AI_ROOT.parent
sys.path.insert(0, str(PY_ROOT))

from openfront_ai.env import (  # noqa: E402
    _parse_mask,
    _reshape_obs,
)
from openfront_ai.policy import MacroMicroPolicy, load_policy_state_dict  # noqa: E402

# Minimum seconds between policy.pt hot-reloads (PPO may save every step ~39MB).
RELOAD_DEBOUNCE_S = 45.0

STATE: dict[str, Any] = {
    "model": None,
    "device": torch.device("cpu"),
    "hx": None,
    "deterministic": True,
    "ckpt": None,
    "ckpt_mtime": None,
    "reloads": 0,
    "ppo_step": None,
    "last_reload_at": 0.0,
    "skipped_reloads": 0,
    "debounce_skip_logged": False,
}


def load_model(ckpt: Path, device: torch.device) -> MacroMicroPolicy:
    model = MacroMicroPolicy().to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    load_policy_state_dict(model, state)
    model.eval()
    STATE["ppo_step"] = int(blob["ppo_step"]) if isinstance(blob, dict) and "ppo_step" in blob else None
    return model


def maybe_reload() -> bool:
    """Hot-reload policy.pt when mtime advances, debounced to avoid PPO save thrash."""
    ckpt: Path | None = STATE["ckpt"]
    if ckpt is None or not ckpt.exists():
        return False
    mtime = ckpt.stat().st_mtime
    if STATE["ckpt_mtime"] is not None and mtime <= STATE["ckpt_mtime"]:
        return False
    now = time.monotonic()
    last = float(STATE["last_reload_at"] or 0.0)
    elapsed = now - last if last > 0 else RELOAD_DEBOUNCE_S
    if last > 0 and elapsed < RELOAD_DEBOUNCE_S:
        STATE["skipped_reloads"] = int(STATE["skipped_reloads"]) + 1
        if not STATE.get("debounce_skip_logged"):
            STATE["debounce_skip_logged"] = True
            print(
                f"Reload skipped (debounce {elapsed:.1f}s < {RELOAD_DEBOUNCE_S:.0f}s; "
                f"pending mtime change; skipped=#{STATE['skipped_reloads']})",
                flush=True,
            )
        return False
    try:
        STATE["model"] = load_model(ckpt, STATE["device"])
        STATE["ckpt_mtime"] = mtime
        STATE["hx"] = None
        STATE["last_reload_at"] = now
        STATE["reloads"] = int(STATE["reloads"]) + 1
        STATE["debounce_skip_logged"] = False
        print(
            f"Reloaded {ckpt} (#{STATE['reloads']}; debounce ok after {elapsed:.1f}s)",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"Reload failed: {e}", flush=True)
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            maybe_reload()
            self._json(
                200,
                {
                    "ok": True,
                    "has_model": STATE["model"] is not None,
                    "reloads": STATE["reloads"],
                    "skipped_reloads": STATE["skipped_reloads"],
                    "ckpt_mtime": STATE["ckpt_mtime"],
                    "ppo_step": STATE["ppo_step"],
                    "reload_debounce_s": RELOAD_DEBOUNCE_S,
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            msg = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "bad json"})
            return

        if self.path.startswith("/reset"):
            STATE["hx"] = None
            self._json(200, {"ok": True})
            return

        if self.path.startswith("/act"):
            try:
                maybe_reload()
                model: MacroMicroPolicy = STATE["model"]
                if model is None:
                    self._json(500, {"ok": False, "error": "no model"})
                    return
                # Viz sends encoding=f32b64; older clients may send float arrays.
                obs = _reshape_obs(msg["obs"])
                mask = _parse_mask(msg.get("mask"))
                device = STATE["device"]
                o = {
                    "global": torch.from_numpy(obs["global"]).unsqueeze(0).to(device),
                    "local": torch.from_numpy(obs["local"]).unsqueeze(0).to(device),
                    "vector": torch.from_numpy(obs["vector"]).unsqueeze(0).to(device),
                }
                masks = {
                    "actionType": torch.from_numpy(mask["actionType"])
                    .unsqueeze(0)
                    .to(device),
                    "targetPlayer": torch.from_numpy(mask["targetPlayer"])
                    .unsqueeze(0)
                    .to(device),
                    "cell": torch.from_numpy(mask["cell"]).unsqueeze(0).to(device),
                    "troopFrac": torch.from_numpy(mask["troopFrac"])
                    .unsqueeze(0)
                    .to(device),
                    "buildType": torch.from_numpy(mask["buildType"])
                    .unsqueeze(0)
                    .to(device),
                }
                with torch.no_grad():
                    (at, tgt, cx, cy, frac, build), value, logp, hx = model.act(
                        o,
                        hx=STATE["hx"],
                        masks=masks,
                        deterministic=bool(
                            msg.get("deterministic", STATE["deterministic"])
                        ),
                    )
                STATE["hx"] = hx
                action = {
                    "actionType": int(at.item()),
                    "targetPlayer": int(tgt.item()),
                    "cellX": int(cx.item()),
                    "cellY": int(cy.item()),
                    "troopFrac": int(frac.item()),
                    "buildType": int(build.item()),
                }
                self._json(
                    200,
                    {
                        "ok": True,
                        "action": action,
                        "value": float(value.item()),
                        "logp": float(logp.item()),
                        "reloads": STATE["reloads"],
                    },
                )
            except Exception as e:
                print(f"/act error: {e}", flush=True)
                self._json(500, {"ok": False, "error": str(e)})
            return

        self._json(404, {"ok": False, "error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=str(AI_ROOT / "fixtures" / "checkpoints" / "policy.pt"),
    )
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    print(f"Loading {ckpt} on {device}", flush=True)
    STATE["device"] = device
    STATE["ckpt"] = ckpt
    STATE["model"] = load_model(ckpt, device)
    STATE["ckpt_mtime"] = ckpt.stat().st_mtime if ckpt.exists() else None
    STATE["deterministic"] = not args.stochastic
    STATE["hx"] = None
    STATE["reloads"] = 0
    STATE["skipped_reloads"] = 0
    STATE["last_reload_at"] = time.monotonic()
    server = HTTPServer((args.host, args.port), Handler)
    print(
        f"Infer server → http://{args.host}:{args.port} "
        f"(hot-reload on, debounce={RELOAD_DEBOUNCE_S:.0f}s)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
