#!/usr/bin/env python3
"""Behavior-clone expand demos, then APPO vs Impossible Nation (TS env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PY_ROOT = Path(__file__).resolve().parents[1]  # ai/python
AI_ROOT = Path(__file__).resolve().parents[2]  # ai/
sys.path.insert(0, str(PY_ROOT))

from openfront_ai.env import OpenFrontEnv
from openfront_ai.policy import MacroMicroPolicy, export_onnx


def collect_batch(env: OpenFrontEnv, n_steps: int, scripted: bool = True):
    obs = env.reset()
    batch = []
    for _ in range(n_steps):
        mask = env.legal_action_types()
        if scripted:
            atype = 2 if len(mask) > 2 and mask[2] else 0
            action = (atype, 0, 16, 8, 2, 0)
        else:
            action = (0, 0, 0, 0, 0, 0)
        result = env.step(action)
        batch.append(
            {"obs": obs, "action": action, "reward": result.reward, "done": result.done}
        )
        obs = result.obs
        if result.done:
            obs = env.reset()
    return batch


def bc_loss(model, batch, device):
    g = torch.stack([torch.from_numpy(b["obs"]["global"]) for b in batch]).to(device)
    l = torch.stack([torch.from_numpy(b["obs"]["local"]) for b in batch]).to(device)
    v = torch.stack([torch.from_numpy(b["obs"]["vector"]) for b in batch]).to(device)
    actions = torch.tensor([b["action"] for b in batch], device=device)
    out, _ = model.forward({"global": g, "local": l, "vector": v})
    return (
        F.cross_entropy(out["action_type_logits"], actions[:, 0])
        + F.cross_entropy(out["target_logits"], actions[:, 1])
        + F.cross_entropy(out["cell_logits"], actions[:, 2] + actions[:, 3] * 32)
        + F.cross_entropy(out["troop_logits"], actions[:, 4])
        + F.cross_entropy(out["build_logits"], actions[:, 5])
    )


def appo_update(model, opt, batch, device, vf_coef=0.5, ent_coef=0.01):
    g = torch.stack([torch.from_numpy(b["obs"]["global"]) for b in batch]).to(device)
    l = torch.stack([torch.from_numpy(b["obs"]["local"]) for b in batch]).to(device)
    v = torch.stack([torch.from_numpy(b["obs"]["vector"]) for b in batch]).to(device)
    actions = torch.tensor([b["action"] for b in batch], device=device)
    rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device)

    returns = torch.zeros_like(rewards)
    G = 0.0
    for i in reversed(range(len(batch))):
        if batch[i]["done"]:
            G = 0.0
        G = float(rewards[i]) + 0.99 * G
        returns[i] = G

    out, _ = model.forward({"global": g, "local": l, "vector": v})
    dist = torch.distributions.Categorical(logits=out["action_type_logits"])
    logp = dist.log_prob(actions[:, 0])
    ent = dist.entropy().mean()
    adv = returns - out["value"].detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    policy_loss = -(logp * adv).mean()
    value_loss = F.mse_loss(out["value"], returns)
    loss = policy_loss + vf_coef * value_loss - ent_coef * ent
    opt.zero_grad()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.item()),
        "policy": float(policy_loss.item()),
        "value": float(value_loss.item()),
        "ent": float(ent.item()),
        "grad": float(grad),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="plains")
    ap.add_argument("--bc-steps", type=int, default=4)
    ap.add_argument("--appo-steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-ts", action="store_true", help="force stub env")
    ap.add_argument(
        "--ckpt",
        default=str(AI_ROOT / "fixtures" / "checkpoints" / "policy.pt"),
    )
    ap.add_argument(
        "--onnx",
        default=str(AI_ROOT / "fixtures" / "checkpoints" / "policy.onnx"),
    )
    ap.add_argument("--league-size", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = MacroMicroPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    env = OpenFrontEnv(map_name=args.map, seed=1, use_ts=not args.no_ts, stride=10)

    print(f"TS env: {not args.no_ts}")
    print("Phase 1: behavior cloning (scripted expand ≈ Nation bootstrap)")
    for step in range(args.bc_steps):
        batch = collect_batch(env, args.batch_size, scripted=True)
        loss = bc_loss(model, batch, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  bc {step}: loss={loss.item():.4f}")

    print("Phase 2: APPO vs Impossible Nation / stub")
    historic = []
    for step in range(args.appo_steps):
        obs = env.reset()
        onpol = []
        for _ in range(args.batch_size):
            with torch.no_grad():
                o = {
                    "global": torch.from_numpy(obs["global"]).unsqueeze(0).to(device),
                    "local": torch.from_numpy(obs["local"]).unsqueeze(0).to(device),
                    "vector": torch.from_numpy(obs["vector"]).unsqueeze(0).to(device),
                }
                mask = torch.from_numpy(env.legal_action_types()).unsqueeze(0).to(device)
                (at, tgt, cx, cy, frac, build), _ = model.act(o, action_mask=mask)
                action = tuple(int(x.item()) for x in (at, tgt, cx, cy, frac, build))
            result = env.step(action)
            onpol.append(
                {"obs": obs, "action": action, "reward": result.reward, "done": result.done}
            )
            obs = result.obs if not result.done else env.reset()
        stats = appo_update(model, opt, onpol, device)
        print(f"  appo {step}: {stats}")
        if step % 2 == 0:
            historic.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if len(historic) > args.league_size:
                historic.pop(0)

    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "league": len(historic)}, ckpt)
    print(f"Saved {ckpt}")

    try:
        export_onnx(model.cpu(), args.onnx)
        print(f"Exported ONNX {args.onnx}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")

    meta = {
        "map": args.map,
        "bc_steps": args.bc_steps,
        "appo_steps": args.appo_steps,
        "league_checkpoints": len(historic),
        "env": "ts" if not args.no_ts else "stub",
    }
    (ckpt.parent / "train_meta.json").write_text(json.dumps(meta, indent=2))
    env.close()


if __name__ == "__main__":
    main()
