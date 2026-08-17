#!/usr/bin/env python3
"""Benchmark GameEnv throughput and timing breakdown.

  cd ai && python python/scripts/bench_env.py --steps 20 --map world --nations 12 --bots 12
  cd ai && python python/scripts/bench_env.py --steps 40 --map onion --nations 3 --bots 2 --num-envs 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_ROOT))

from openfront_ai.env import (  # noqa: E402
    DEFAULT_STRIDE,
    OpenFrontEnv,
    make_parallel_env,
)


def _bench_single(args: argparse.Namespace) -> dict:
    env = OpenFrontEnv(
        map_name=args.map,
        seed=args.seed,
        use_ts=True,
        stride=args.stride,
        difficulty=args.difficulty,
        nations=args.nations,
        bots=args.bots,
        max_ticks=args.max_ticks,
    )

    t0 = time.perf_counter()
    obs, mask, info = env.reset(args.seed)
    reset_s = time.perf_counter() - t0
    reset_timing = dict(env.last_timing)

    t1 = time.perf_counter()
    env.reset(args.seed + 1)
    reset2_s = time.perf_counter() - t1

    sim_ms = obs_ms = mask_ms = ser_ms = 0.0
    step_wall = 0.0
    action = (0, 0, 0, 0, 0, 0)
    for i in range(args.steps):
        t = time.perf_counter()
        result = env.step(action)
        step_wall += time.perf_counter() - t
        timing = result.timing or env.last_timing
        sim_ms += float(timing.get("simMs", 0))
        obs_ms += float(timing.get("obsMs", 0))
        mask_ms += float(timing.get("maskMs", 0))
        ser_ms += float(timing.get("serializeMs", 0))
        if result.done:
            env.reset(args.seed + 100 + i)

    env.close()
    n = max(1, args.steps)
    return {
        "num_envs": 1,
        "map": args.map,
        "nations": args.nations,
        "bots": args.bots,
        "stride": args.stride,
        "steps": args.steps,
        "reset_s": reset_s,
        "reset_cached_s": reset2_s,
        "reset_timing": reset_timing,
        "step_wall_s": step_wall,
        "steps_per_sec": n / max(1e-9, step_wall),
        "aggregate_steps_per_sec": n / max(1e-9, step_wall),
        "mean_sim_ms": sim_ms / n,
        "mean_obs_ms": obs_ms / n,
        "mean_mask_ms": mask_ms / n,
        "mean_serialize_ms": ser_ms / n,
        "mean_step_wall_ms": 1000.0 * step_wall / n,
        "obs_nbytes": int(
            obs["global"].nbytes + obs["local"].nbytes + obs["vector"].nbytes
        ),
        "mask_legal_actions": int(np.sum(mask["actionType"])),
        "info_tick": info.get("tick"),
    }


def _bench_vec(args: argparse.Namespace) -> dict:
    n_env = args.num_envs
    vec = make_parallel_env(
        n_env,
        map_name=args.map,
        seed=args.seed,
        stride=args.stride,
        use_ts=True,
        difficulty=args.difficulty,
        nations=args.nations,
        bots=args.bots,
        max_ticks=args.max_ticks,
    )

    t0 = time.perf_counter()
    obs_list, mask_list, info_list = vec.reset(args.seed)
    reset_s = time.perf_counter() - t0
    reset_timing = dict(vec.envs[0].last_timing)

    t1 = time.perf_counter()
    vec.reset(args.seed + n_env)
    reset2_s = time.perf_counter() - t1

    sim_ms = obs_ms = mask_ms = ser_ms = 0.0
    step_wall = 0.0
    actions = [(0, 0, 0, 0, 0, 0)] * n_env
    for i in range(args.steps):
        t = time.perf_counter()
        results = vec.step(actions)
        step_wall += time.perf_counter() - t
        for result in results:
            timing = result.timing or {}
            sim_ms += float(timing.get("simMs", 0))
            obs_ms += float(timing.get("obsMs", 0))
            mask_ms += float(timing.get("maskMs", 0))
            ser_ms += float(timing.get("serializeMs", 0))
        if any(r.done for r in results):
            vec.reset(args.seed + 1000 + i * n_env)

    vec.close()
    wall_steps = max(1, args.steps)
    aggregate = wall_steps * n_env
    obs = obs_list[0]
    mask = mask_list[0]
    return {
        "num_envs": n_env,
        "map": args.map,
        "nations": args.nations,
        "bots": args.bots,
        "stride": args.stride,
        "steps": args.steps,
        "reset_s": reset_s,
        "reset_cached_s": reset2_s,
        "reset_timing": reset_timing,
        "step_wall_s": step_wall,
        "steps_per_sec": wall_steps / max(1e-9, step_wall),
        "aggregate_steps_per_sec": aggregate / max(1e-9, step_wall),
        "mean_sim_ms": sim_ms / aggregate,
        "mean_obs_ms": obs_ms / aggregate,
        "mean_mask_ms": mask_ms / aggregate,
        "mean_serialize_ms": ser_ms / aggregate,
        "mean_step_wall_ms": 1000.0 * step_wall / wall_steps,
        "obs_nbytes": int(
            obs["global"].nbytes + obs["local"].nbytes + obs["vector"].nbytes
        ),
        "mask_legal_actions": int(np.sum(mask["actionType"])),
        "info_tick": info_list[0].get("tick"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark TS GameEnv step throughput")
    ap.add_argument("--map", default="world")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--nations", type=int, default=12)
    ap.add_argument("--bots", type=int, default=12)
    ap.add_argument("--difficulty", default="Easy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ticks", type=int, default=8_000)
    ap.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel pool slots (1 = single OpenFrontEnv)",
    )
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()

    report = _bench_vec(args) if args.num_envs > 1 else _bench_single(args)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== OpenFront env bench ===")
        print(
            f"map={args.map} nations={args.nations} bots={args.bots} "
            f"num_envs={report['num_envs']} stride={args.stride} steps={args.steps}"
        )
        print(f"reset (cold):   {report['reset_s']*1000:.1f} ms")
        print(f"reset (cached): {report['reset_cached_s']*1000:.1f} ms")
        print(f"wall steps/sec: {report['steps_per_sec']:.2f}")
        print(f"agg steps/sec:  {report['aggregate_steps_per_sec']:.2f}")
        print(
            f"mean step wall: {report['mean_step_wall_ms']:.1f} ms "
            f"(sim={report['mean_sim_ms']:.1f} "
            f"obs={report['mean_obs_ms']:.1f} "
            f"mask={report['mean_mask_ms']:.1f} "
            f"ser={report['mean_serialize_ms']:.1f})"
        )
        print(f"obs nbytes:     {report['obs_nbytes']}")


if __name__ == "__main__":
    main()
