#!/usr/bin/env python3
"""4-seed long evaluation against the current policy (real win metric)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_ROOT))

from openfront_ai.curriculum import classify_eval_outcome  # noqa: E402
from openfront_ai.env import OpenFrontEnv  # noqa: E402
from openfront_ai.policy import MacroMicroPolicy, load_policy_state_dict  # noqa: E402
from scripts.train import masks_to_torch, obs_to_torch  # noqa: E402


def evaluate_with_progress(
    env: OpenFrontEnv,
    model: MacroMicroPolicy,
    device: torch.device,
    seeds: list[int],
    max_steps: int,
    checkpoint_path: Path | None = None,
) -> dict:
    """Same as train.evaluate, but prints per-seed progress."""
    # Reuse evaluate by calling it one seed at a time so we can log.
    seed_results = []
    tiles = []
    nation_tiles = []
    enemy_tiles = []
    opponents_alive = []
    placements = []
    boats = []
    ratios = []
    returns = []
    ticks = []
    cities = []
    wins = losses = episodes_finished = 0
    rng_state = torch.random.get_rng_state()
    for seed in seeds:
        print(f"seed {seed}: starting (max_steps={max_steps})", flush=True)
        torch.manual_seed(100_000 + seed)
        obs, mask, _ = env.reset(seed)
        hx = None
        total_r = 0.0
        last_info = {}
        done = False
        for step_i in range(max_steps):
            with torch.no_grad():
                (at, tgt, cx, cy, frac, build), _, _, hx = model.act(
                    obs_to_torch(obs, device),
                    hx=hx,
                    masks=masks_to_torch(mask, device),
                    deterministic=False,
                )
                action = tuple(int(x.item()) for x in (at, tgt, cx, cy, frac, build))
            step = env.step(action)
            total_r += step.reward
            last_info = step.info
            obs, mask = step.obs, step.mask
            if step_i % 50 == 0 or step.done:
                print(
                    f"  seed {seed} step={step_i} tick={last_info.get('tick', 0)} "
                    f"tiles={last_info.get('agentTiles', 0)} "
                    f"place={last_info.get('placement', 1)} "
                    f"cities={last_info.get('agentCities', 0)} "
                    f"boats={last_info.get('agentBoats', 0)} done={step.done}",
                    flush=True,
                )
            if step.done:
                done = True
                episodes_finished += 1
                break
        tiles.append(float(last_info.get("agentTiles", 0)))
        nation_tiles.append(float(last_info.get("nationTiles", 0)))
        enemy_tiles.append(
            float(last_info.get("enemyTilesTotal", last_info.get("nationTiles", 0)))
        )
        opponents_alive.append(float(last_info.get("opponentsAlive", 0)))
        placements.append(float(last_info.get("placement", 1)))
        boats.append(float(last_info.get("agentBoats", 0)))
        ratios.append(float(last_info.get("troopRatio", 0)))
        returns.append(total_r)
        ticks.append(float(last_info.get("tick", 0)))
        cities.append(float(last_info.get("agentCities", 0)))
        winner = last_info.get("winner")
        placement = int(last_info.get("placement", 1))
        agent_tiles = int(last_info.get("agentTiles", 0))
        strongest_enemy = int(
            last_info.get(
                "strongestEnemyTiles",
                last_info.get("nationTiles", 0),
            )
        )
        outcome = classify_eval_outcome(
            done=done,
            winner=winner,
            placement=placement,
            agent_tiles=agent_tiles,
            strongest_enemy_tiles=strongest_enemy,
        )
        if outcome == "win":
            wins += 1
        else:
            losses += 1
        row = {
            "seed": seed,
            "winner": winner,
            "done": done,
            "outcome": outcome,
            "tick": int(last_info.get("tick", 0)),
            "agent_tiles": int(last_info.get("agentTiles", 0)),
            "nation_tiles": int(last_info.get("nationTiles", 0)),
            "enemy_tiles_total": int(last_info.get("enemyTilesTotal", 0)),
            "opponents_alive": int(last_info.get("opponentsAlive", 0)),
            "placement": placement,
            "agent_cities": int(last_info.get("agentCities", 0)),
            "agent_boats": int(last_info.get("agentBoats", 0)),
        }
        seed_results.append(row)
        print(f"seed {seed}: finished {row}", flush=True)
        # Checkpoint after each seed so a mid-run kill still leaves useful results.
        partial = {
            "wins": wins,
            "losses": losses,
            "episodes_finished": episodes_finished,
            "evaluated": len(seed_results),
            "win_rate": wins / len(seed_results) if seed_results else 0.0,
            "mean_ticks": float(__import__("numpy").mean(ticks)),
            "mean_tiles": float(__import__("numpy").mean(tiles)),
            "mean_enemy_tiles": float(__import__("numpy").mean(enemy_tiles)),
            "mean_opponents_alive": float(__import__("numpy").mean(opponents_alive)),
            "mean_placement": float(__import__("numpy").mean(placements)),
            "mean_boats": float(__import__("numpy").mean(boats)),
            "mean_cities": float(__import__("numpy").mean(cities)),
            "mean_troop_ratio": float(__import__("numpy").mean(ratios)),
            "mean_return": float(__import__("numpy").mean(returns)),
            "seed_results": list(seed_results),
            "partial": True,
            "seeds_completed": [r["seed"] for r in seed_results],
        }
        if checkpoint_path is not None:
            checkpoint_path.write_text(json.dumps(partial, indent=2) + "\n")
            print(f"  checkpointed {checkpoint_path}", flush=True)
    torch.random.set_rng_state(rng_state)
    import numpy as np

    return {
        "wins": wins,
        "losses": losses,
        "episodes_finished": episodes_finished,
        "evaluated": len(seeds),
        "win_rate": wins / len(seeds) if seeds else 0.0,
        "mean_ticks": float(np.mean(ticks)),
        "mean_tiles": float(np.mean(tiles)),
        "mean_nation_tiles": float(np.mean(nation_tiles)),
        "mean_enemy_tiles": float(np.mean(enemy_tiles)),
        "mean_opponents_alive": float(np.mean(opponents_alive)),
        "mean_placement": float(np.mean(placements)),
        "mean_boats": float(np.mean(boats)),
        "mean_troop_ratio": float(np.mean(ratios)),
        "mean_return": float(np.mean(returns)),
        "mean_cities": float(np.mean(cities)) if cities else 0.0,
        "policy_mode": "stochastic",
        "seed_results": seed_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ai/fixtures/checkpoints/policy.pt")
    ap.add_argument("--out", default="ai/fixtures/checkpoints/long_eval_latest.json")
    ap.add_argument("--map", default="world")
    ap.add_argument("--difficulty", default="Easy")
    ap.add_argument("--nations", type=int, default=12)
    ap.add_argument("--bots", type=int, default=12)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-ticks", type=int, default=4000)
    ap.add_argument("--seeds", default="1,2,3,4")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    # One macro-step = stride ticks; cover max-ticks.
    max_steps = max(1, (args.max_ticks + args.stride - 1) // args.stride)

    device = torch.device("cpu")
    model = MacroMicroPolicy().to(device)
    ckpt = Path(args.ckpt)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    load_policy_state_dict(model, state)
    model.eval()
    ppo_step = int(blob.get("ppo_step", -1)) if isinstance(blob, dict) else -1
    print(f"Loaded {ckpt} ppo_step={ppo_step}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = OpenFrontEnv(
        map_name=args.map,
        stride=args.stride,
        difficulty=args.difficulty,
        nations=args.nations,
        bots=args.bots,
        max_ticks=args.max_ticks,
    )
    try:
        result = evaluate_with_progress(
            env, model, device, seeds, max_steps, checkpoint_path=out
        )
    finally:
        env.close()

    result.update(
        {
            "map": args.map,
            "difficulty": args.difficulty,
            "nations": args.nations,
            "bots": args.bots,
            "stride": args.stride,
            "max_ticks": args.max_ticks,
            "seeds": seeds,
            "checkpoint": str(ckpt),
            "ppo_step": ppo_step,
            "note": "Authoritative win metric; short --eval-steps proxies are not.",
        }
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
