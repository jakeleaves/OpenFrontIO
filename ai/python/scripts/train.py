#!/usr/bin/env python3
"""Behavior-clone expert demos, then factorized masked PPO vs Impossible Nation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PY_ROOT = Path(__file__).resolve().parents[1]  # ai/python
AI_ROOT = Path(__file__).resolve().parents[2]  # ai/
sys.path.insert(0, str(PY_ROOT))

from openfront_ai.curriculum import (
    CURRICULUM_DEMOTE_FLOORS,
    CURRICULUM_DEMOTE_STREAK,
    CURRICULUM_PROMOTE_STREAK,
    CURRICULUM_SCALES,
    CURRICULUM_THRESHOLDS,
    DIFFICULTIES,
    EASY_MASTERY_MAX_PLACEMENT,
    EASY_MASTERY_MIN_WIN_RATE,
    classify_eval_outcome,
    easy_mastery_reached,
    mix_demo_sources,
    promote_streak_required,
    scale_for_difficulty,
    update_curriculum,
)
from openfront_ai.demos import load_jsonl_demos, stack_transitions
from openfront_ai.env import (
    COARSE_W,
    DEFAULT_STRIDE,
    OpenFrontEnv,
    RpcError,
    VecEnv,
    make_parallel_env,
)
from openfront_ai.policy import MacroMicroPolicy, export_onnx, load_policy_state_dict


@dataclass
class StepTimer:
    """Accumulate wall-clock splits for one PPO iteration."""

    env_reset_s: float = 0.0
    env_step_s: float = 0.0
    policy_act_s: float = 0.0
    ppo_update_s: float = 0.0
    eval_s: float = 0.0
    env_steps: int = 0
    ts_sim_ms: float = 0.0
    ts_obs_ms: float = 0.0
    ts_mask_ms: float = 0.0
    ts_serialize_ms: float = 0.0

    def as_dict(self) -> dict:
        wall = (
            self.env_reset_s
            + self.env_step_s
            + self.policy_act_s
            + self.ppo_update_s
            + self.eval_s
        )
        return {
            "env_reset_s": round(self.env_reset_s, 4),
            "env_step_s": round(self.env_step_s, 4),
            "policy_act_s": round(self.policy_act_s, 4),
            "ppo_update_s": round(self.ppo_update_s, 4),
            "eval_s": round(self.eval_s, 4),
            "wall_s": round(wall, 4),
            "steps_per_sec": (
                round(self.env_steps / max(1e-9, self.env_step_s + self.env_reset_s), 3)
                if self.env_steps
                else 0.0
            ),
            "env_steps": self.env_steps,
            "ts_mean_sim_ms": round(self.ts_sim_ms / max(1, self.env_steps), 3),
            "ts_mean_obs_ms": round(self.ts_obs_ms / max(1, self.env_steps), 3),
            "ts_mean_mask_ms": round(self.ts_mask_ms / max(1, self.env_steps), 3),
            "ts_mean_serialize_ms": round(
                self.ts_serialize_ms / max(1, self.env_steps), 3
            ),
        }

    def note_ts_timing(self, timing: dict | None) -> None:
        if not timing:
            return
        self.ts_sim_ms += float(timing.get("simMs", 0))
        self.ts_obs_ms += float(timing.get("obsMs", 0))
        self.ts_mask_ms += float(timing.get("maskMs", 0))
        self.ts_serialize_ms += float(timing.get("serializeMs", 0))


def log_rollout_progress(
    *,
    completed: int,
    total: int,
    timer: StepTimer,
    latest_timing: dict | None = None,
    slots: int = 1,
) -> None:
    """Make an expensive environment step distinguishable from a stalled run."""
    timing = latest_timing or {}
    print(
        "  rollout "
        f"{completed}/{total} slots={slots} "
        f"sps={timer.as_dict()['steps_per_sec']:.2f} "
        f"last_ms=sim:{float(timing.get('simMs', 0)):.0f} "
        f"obs:{float(timing.get('obsMs', 0)):.0f} "
        f"mask:{float(timing.get('maskMs', 0)):.0f}",
        flush=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_to_torch(obs: dict, device: torch.device) -> dict:
    return {
        "global": torch.from_numpy(obs["global"]).unsqueeze(0).to(device),
        "local": torch.from_numpy(obs["local"]).unsqueeze(0).to(device),
        "vector": torch.from_numpy(obs["vector"]).unsqueeze(0).to(device),
    }


def batch_obs_to_torch(obses: list[dict], device: torch.device) -> dict:
    return {
        "global": torch.from_numpy(
            np.stack([o["global"] for o in obses], axis=0)
        ).to(device),
        "local": torch.from_numpy(
            np.stack([o["local"] for o in obses], axis=0)
        ).to(device),
        "vector": torch.from_numpy(
            np.stack([o["vector"] for o in obses], axis=0)
        ).to(device),
    }


def masks_to_torch(mask: dict, device: torch.device) -> dict:
    return {
        "actionType": torch.from_numpy(mask["actionType"]).unsqueeze(0).to(device),
        "targetPlayer": torch.from_numpy(mask["targetPlayer"]).unsqueeze(0).to(device),
        "cell": torch.from_numpy(mask["cell"]).unsqueeze(0).to(device),
        "troopFrac": torch.from_numpy(mask["troopFrac"]).unsqueeze(0).to(device),
        "buildType": torch.from_numpy(mask["buildType"]).unsqueeze(0).to(device),
    }


def batch_masks_list_to_torch(masks: list[dict], device: torch.device) -> dict:
    return {
        "actionType": torch.from_numpy(
            np.stack([m["actionType"] for m in masks], axis=0)
        ).to(device),
        "targetPlayer": torch.from_numpy(
            np.stack([m["targetPlayer"] for m in masks], axis=0)
        ).to(device),
        "cell": torch.from_numpy(np.stack([m["cell"] for m in masks], axis=0)).to(
            device
        ),
        "troopFrac": torch.from_numpy(
            np.stack([m["troopFrac"] for m in masks], axis=0)
        ).to(device),
        "buildType": torch.from_numpy(
            np.stack([m["buildType"] for m in masks], axis=0)
        ).to(device),
    }


def batch_masks_to_torch(batch: dict, device: torch.device) -> dict:
    return {
        "actionType": torch.from_numpy(batch["actionType_mask"]).to(device),
        "targetPlayer": torch.from_numpy(batch["targetPlayer_mask"]).to(device),
        "cell": torch.from_numpy(batch["cell_mask"]).to(device),
        "troopFrac": torch.from_numpy(batch["troopFrac_mask"]).to(device),
        "buildType": torch.from_numpy(batch["buildType_mask"]).to(device),
    }


def rebalance_demos(demos: list[dict], noop_frac: float = 0.15) -> list[dict]:
    """Downsample NOOP so BC does not collapse to idle."""
    if not demos:
        return demos
    noop = [d for d in demos if d["action"][0] == 0]
    other = [d for d in demos if d["action"][0] != 0]
    if not other:
        return demos
    max_noop = int(len(other) * noop_frac / max(1e-6, 1.0 - noop_frac))
    keep_noop = random.sample(noop, min(len(noop), max_noop)) if noop else []
    out = other + keep_noop
    random.shuffle(out)
    return out


def bc_loss(model: MacroMicroPolicy, batch: dict, device: torch.device) -> torch.Tensor:
    obs = {
        "global": torch.from_numpy(batch["global"]).to(device),
        "local": torch.from_numpy(batch["local"]).to(device),
        "vector": torch.from_numpy(batch["vector"]).to(device),
    }
    masks = batch_masks_to_torch(batch, device)
    actions = torch.from_numpy(batch["action"]).to(device)
    out, _ = model.forward(obs, masks=masks)
    at = actions[:, 0]
    cell = actions[:, 2] + actions[:, 3] * COARSE_W

    # Per-head losses only where the head is causal for that action type.
    # Avoid CE on fully-masked logits (e.g. BUILD type when action is ATTACK).
    is_attack = at == 2
    is_build = at == 6
    is_boat = at == 3
    uses_cell = is_build | is_boat | (at == 1) | (at == 8)  # spawn/boat/build/warship
    uses_troop = is_attack | is_boat
    uses_target = is_attack

    def masked_mean(per_sample: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        if keep.any():
            return per_sample[keep].mean()
        return per_sample.new_zeros(())

    loss = F.cross_entropy(out["action_type_logits"], at)
    loss = loss + masked_mean(
        F.cross_entropy(out["target_logits"], actions[:, 1], reduction="none"),
        uses_target,
    )
    loss = loss + masked_mean(
        F.cross_entropy(out["cell_logits"], cell, reduction="none"),
        uses_cell & ~is_attack,
    )
    loss = loss + masked_mean(
        F.cross_entropy(out["troop_logits"], actions[:, 4], reduction="none"),
        uses_troop,
    )
    loss = loss + masked_mean(
        F.cross_entropy(out["build_logits"], actions[:, 5], reduction="none"),
        is_build,
    )
    if "macro" in batch:
        macro = torch.from_numpy(batch["macro"]).to(device)
        has_macro = macro >= 0
        if has_macro.any():
            loss = loss + F.cross_entropy(
                out["macro_logits"][has_macro], macro[has_macro]
            )
    return loss


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_nonterminal = 0.0 if dones[t] else 1.0
        next_value = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def clipped_value_loss(
    value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip: float = 0.2,
) -> torch.Tensor:
    """PPO clipped value loss. Equals MSE when |v - v_old| <= clip."""
    v_clipped = old_value + (value - old_value).clamp(-clip, clip)
    vf1 = (value - returns).pow(2)
    vf2 = (v_clipped - returns).pow(2)
    return torch.max(vf1, vf2).mean()


def collect_rollout(
    env: OpenFrontEnv,
    model: MacroMicroPolicy,
    device: torch.device,
    horizon: int,
    seed: int,
    episode_rollouts: bool = False,
    max_episode_steps: int = 1500,
    timer: StepTimer | None = None,
) -> tuple[list[dict], float, dict]:
    t0 = time.perf_counter()
    try:
        obs, mask, info = env.reset(seed)
    except RpcError as e:
        print(f"collect_rollout reset failed: {e}", flush=True)
        return [], 0.0, {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "city_builds": 0,
            "city_build_intents": 0,
            "max_cities": 0,
            "rpc_error": True,
        }
    if timer is not None:
        timer.env_reset_s += time.perf_counter() - t0
        timer.note_ts_timing(env.last_timing)
    hx = None
    traj: list[dict] = []
    limit = max_episode_steps if episode_rollouts else horizon
    finished = wins = losses = 0
    city_builds = 0
    max_cities = int(info.get("agentCities", 0) or 0)
    prev_cities = max_cities
    done = False
    progress_every = max(1, limit // 4)
    for rollout_step in range(limit):
        with torch.no_grad():
            t_act = time.perf_counter()
            o = obs_to_torch(obs, device)
            m = masks_to_torch(mask, device)
            (at, tgt, cx, cy, frac, build), value, logp, hx = model.act(
                o, hx=hx, masks=m
            )
            action = tuple(int(x.item()) for x in (at, tgt, cx, cy, frac, build))
            value_f = float(value.item())
            logp_f = float(logp.item())
            if timer is not None:
                timer.policy_act_s += time.perf_counter() - t_act
        try:
            t_step = time.perf_counter()
            step = env.step(action)
            if timer is not None:
                timer.env_step_s += time.perf_counter() - t_step
                timer.env_steps += 1
                timer.note_ts_timing(step.timing)
        except RpcError as e:
            print(f"collect_rollout step failed (truncating): {e}", flush=True)
            break
        if (rollout_step + 1) % progress_every == 0:
            log_rollout_progress(
                completed=rollout_step + 1,
                total=limit,
                timer=timer or StepTimer(),
                latest_timing=step.timing,
            )
        cities_now = int(step.info.get("agentCities", 0) or 0)
        if cities_now > prev_cities:
            city_builds += cities_now - prev_cities
        prev_cities = cities_now
        max_cities = max(max_cities, cities_now)
        traj.append(
            {
                "obs": obs,
                "mask": mask,
                "action": action,
                "reward": step.reward,
                "done": step.done,
                "value": value_f,
                "logp": logp_f,
                "info": step.info,
            }
        )
        obs, mask = step.obs, step.mask
        if step.done or step.info.get("rpc_error"):
            finished += 1
            if step.info.get("winner") == "agent":
                wins += 1
            else:
                losses += 1
            if episode_rollouts or step.info.get("rpc_error"):
                done = True
                break
            try:
                t_reset = time.perf_counter()
                obs, mask, info = env.reset(seed + len(traj))
                if timer is not None:
                    timer.env_reset_s += time.perf_counter() - t_reset
                    timer.note_ts_timing(env.last_timing)
            except RpcError as e:
                print(f"collect_rollout mid-reset failed: {e}", flush=True)
                done = True
                break
            hx = None
            prev_cities = int(info.get("agentCities", 0) or 0)
            max_cities = max(max_cities, prev_cities)
    last_value = 0.0
    if traj and not done and not traj[-1]["done"]:
        with torch.no_grad():
            out, _ = model.forward(
                obs_to_torch(obs, device),
                hx=hx,
                masks=masks_to_torch(mask, device),
            )
            last_value = float(out["value"].item())
    city_build_intents = sum(
        1 for t in traj if t["action"][0] == 6 and t["action"][5] == 0
    )
    return traj, last_value, {
        "episodes": finished,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / finished if finished else 0.0,
        "city_builds": city_builds,
        "city_build_intents": city_build_intents,
        "max_cities": max_cities,
    }


def collect_vec_rollout(
    vec: VecEnv,
    model: MacroMicroPolicy,
    device: torch.device,
    horizon: int,
    seed: int,
    timer: StepTimer | None = None,
) -> tuple[list[dict], list[float], list[int], dict]:
    """Parallel rollout: batch policy forward, pipeline env steps."""
    n = vec.n
    t0 = time.perf_counter()
    obses, masks, infos = vec.reset([seed + i for i in range(n)])
    if timer is not None:
        timer.env_reset_s += time.perf_counter() - t0
        for env in vec.envs:
            timer.note_ts_timing(env.last_timing)

    trajs: list[list[dict]] = [[] for _ in range(n)]
    hx = None
    finished = wins = losses = 0
    city_builds = 0
    max_cities = max(int(info.get("agentCities", 0) or 0) for info in infos)
    prev_cities = [int(info.get("agentCities", 0) or 0) for info in infos]
    active = [True] * n

    progress_every = max(1, horizon // 4)
    for rollout_step in range(horizon):
        if not any(active):
            break
        with torch.no_grad():
            t_act = time.perf_counter()
            o = batch_obs_to_torch(obses, device)
            m = batch_masks_list_to_torch(masks, device)
            (at, tgt, cx, cy, frac, build), value, logp, hx = model.act(
                o, hx=hx, masks=m
            )
            if timer is not None:
                timer.policy_act_s += time.perf_counter() - t_act
            actions = [
                (
                    int(at[i].item()),
                    int(tgt[i].item()),
                    int(cx[i].item()),
                    int(cy[i].item()),
                    int(frac[i].item()),
                    int(build[i].item()),
                )
                for i in range(n)
            ]
            values = [float(value[i].item()) for i in range(n)]
            logps = [float(logp[i].item()) for i in range(n)]

        t_step = time.perf_counter()
        steps = vec.step(actions)
        if timer is not None:
            timer.env_step_s += time.perf_counter() - t_step
            timer.env_steps += n
            for env in vec.envs:
                timer.note_ts_timing(env.last_timing)
        if (rollout_step + 1) % progress_every == 0:
            log_rollout_progress(
                completed=rollout_step + 1,
                total=horizon,
                timer=timer or StepTimer(),
                latest_timing=steps[0].timing if steps else None,
                slots=n,
            )

        need_reset_idx: list[int] = []
        for i, step in enumerate(steps):
            if not active[i]:
                continue
            cities_now = int(step.info.get("agentCities", 0) or 0)
            if cities_now > prev_cities[i]:
                city_builds += cities_now - prev_cities[i]
            prev_cities[i] = cities_now
            max_cities = max(max_cities, cities_now)
            trajs[i].append(
                {
                    "obs": obses[i],
                    "mask": masks[i],
                    "action": actions[i],
                    "reward": step.reward,
                    "done": step.done,
                    "value": values[i],
                    "logp": logps[i],
                    "info": step.info,
                }
            )
            obses[i] = step.obs
            masks[i] = step.mask
            if step.done or step.info.get("rpc_error"):
                finished += 1
                if step.info.get("winner") == "agent":
                    wins += 1
                else:
                    losses += 1
                if step.info.get("rpc_error"):
                    active[i] = False
                else:
                    need_reset_idx.append(i)

        if need_reset_idx:
            # Reset finished envs individually (keep others mid-episode).
            for i in need_reset_idx:
                t_reset = time.perf_counter()
                try:
                    o, m, info = vec.envs[i].reset(seed + 10_000 + len(trajs[i]) + i)
                    if timer is not None:
                        timer.env_reset_s += time.perf_counter() - t_reset
                        timer.note_ts_timing(vec.envs[i].last_timing)
                    obses[i], masks[i] = o, m
                    prev_cities[i] = int(info.get("agentCities", 0) or 0)
                    # Feed-forward policy; hx is unused.
                except RpcError:
                    active[i] = False

    # Bootstrap values per env trajectory.
    last_values: list[float] = []
    flat: list[dict] = []
    segment_lengths: list[int] = []
    with torch.no_grad():
        for i in range(n):
            if not trajs[i]:
                last_values.append(0.0)
                continue
            segment_lengths.append(len(trajs[i]))
            flat.extend(trajs[i])
            if trajs[i][-1]["done"]:
                last_values.append(0.0)
            else:
                out, _ = model.forward(
                    obs_to_torch(obses[i], device),
                    masks=masks_to_torch(masks[i], device),
                )
                last_values.append(float(out["value"].item()))

    city_build_intents = sum(
        1 for t in flat if t["action"][0] == 6 and t["action"][5] == 0
    )
    return flat, last_values, segment_lengths, {
        "episodes": finished,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / finished if finished else 0.0,
        "city_builds": city_builds,
        "city_build_intents": city_build_intents,
        "max_cities": max_cities,
    }


def ppo_update(
    model: MacroMicroPolicy,
    opt: torch.optim.Optimizer,
    traj: list[dict],
    device: torch.device,
    last_value: float | list[float],
    segment_lengths: list[int] | None = None,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.03,
    ent_coef_floor: float = 0.01,
    epochs: int = 2,
    minibatch: int = 32,
    target_kl: float = 0.03,
) -> dict:
    # Preserve terminal ±lambdaW. Clipping it to ±20 made hundreds of NOOP
    # growth rewards outweigh losing, despite the intended terminal dominance.
    ent_coef = max(ent_coef, ent_coef_floor)
    rewards = [
        float(t["reward"])
        if t["done"]
        else float(np.clip(t["reward"], -20.0, 20.0))
        for t in traj
    ]
    values = [t["value"] for t in traj]
    dones = [t["done"] for t in traj]
    if segment_lengths is None:
        adv, returns = compute_gae(rewards, values, dones, last_value=float(last_value))
    else:
        if not isinstance(last_value, list) or len(last_value) != len(segment_lengths):
            raise ValueError("segment_lengths must match one bootstrap value per rollout")
        adv_parts = []
        return_parts = []
        offset = 0
        for length, bootstrap in zip(segment_lengths, last_value):
            end = offset + length
            part_adv, part_returns = compute_gae(
                rewards[offset:end], values[offset:end], dones[offset:end], bootstrap
            )
            adv_parts.append(part_adv)
            return_parts.append(part_returns)
            offset = end
        adv = np.concatenate(adv_parts) if adv_parts else np.zeros(0, dtype=np.float32)
        returns = (
            np.concatenate(return_parts) if return_parts else np.zeros(0, dtype=np.float32)
        )
    if len(traj) < 2:
        print("skipped_empty_rollout: traj length < 2", flush=True)
        return {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "ent": 0.0,
            "approx_kl": 0.0,
            "grad_norm": 0.0,
            "updates": 0,
            "early_stop": False,
            "skipped_empty_rollout": True,
        }
    adv_t = torch.from_numpy(adv).to(device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    # Keep returns in reward units (do not z-score). Easy/Medium batch mix
    # otherwise moves the value target scale every curriculum flip.
    ret_t = torch.from_numpy(returns).to(device)
    old_logp = torch.tensor([t["logp"] for t in traj], dtype=torch.float32, device=device)
    value_arr = np.asarray(values, dtype=np.float32)
    old_values = torch.from_numpy(value_arr).to(device)
    return_var = float(np.var(returns))
    explained_variance = (
        1.0 - float(np.var(returns - value_arr)) / return_var
        if return_var > 1e-8
        else 0.0
    )

    stacked = stack_transitions(traj)
    n = len(traj)
    idxs = list(range(n))
    stats = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "ent": 0.0,
        "approx_kl": 0.0,
        "grad_norm": 0.0,
        "updates": 0,
        "early_stop": False,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "explained_variance": explained_variance,
    }

    for _ in range(epochs):
        random.shuffle(idxs)
        early = False
        for start in range(0, n, minibatch):
            mb = idxs[start : start + minibatch]
            if len(mb) < 2:
                continue
            obs = {
                "global": torch.from_numpy(stacked["global"][mb]).to(device),
                "local": torch.from_numpy(stacked["local"][mb]).to(device),
                "vector": torch.from_numpy(stacked["vector"][mb]).to(device),
            }
            masks = {
                "actionType": torch.from_numpy(stacked["actionType_mask"][mb]).to(device),
                "targetPlayer": torch.from_numpy(stacked["targetPlayer_mask"][mb]).to(
                    device
                ),
                "cell": torch.from_numpy(stacked["cell_mask"][mb]).to(device),
                "troopFrac": torch.from_numpy(stacked["troopFrac_mask"][mb]).to(device),
                "buildType": torch.from_numpy(stacked["buildType_mask"][mb]).to(device),
            }
            actions = torch.from_numpy(stacked["action"][mb]).to(device)
            logp, value, ent, _, _ = model.evaluate_actions(obs, actions, masks=masks)
            ratio = torch.exp(logp - old_logp[mb])
            approx_kl = float(torch.mean(old_logp[mb] - logp).abs().item())
            if approx_kl > target_kl:
                stats["early_stop"] = True
                early = True
                break
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = clipped_value_loss(
                value, old_values[mb], ret_t[mb], clip=clip
            )
            loss = policy_loss + vf_coef * value_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            stats["loss"] += float(loss.item())
            stats["policy"] += float(policy_loss.item())
            stats["value"] += float(value_loss.item())
            stats["ent"] += float(ent.item())
            stats["approx_kl"] += approx_kl
            stats["grad_norm"] += grad
            stats["updates"] += 1
        if early:
            break

    if stats["updates"]:
        for k in ("loss", "policy", "value", "ent", "approx_kl", "grad_norm"):
            stats[k] /= stats["updates"]
    return stats


def evaluate(
    env: OpenFrontEnv,
    model: MacroMicroPolicy,
    device: torch.device,
    seeds: list[int],
    steps: int = 0,
    max_steps: int = 2000,
    deterministic: bool = False,
) -> dict:
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
    seed_results = []
    wins = losses = 0
    episodes_finished = 0
    # Stochastic eval should be reproducible without perturbing PPO sampling.
    rng_state = torch.random.get_rng_state()
    for seed in seeds:
        torch.manual_seed(100_000 + seed)
        obs, mask, _ = env.reset(seed)
        hx = None
        total_r = 0.0
        last_info = {}
        done = False
        for _ in range(max_steps if steps == 0 else min(steps, max_steps)):
            with torch.no_grad():
                (at, tgt, cx, cy, frac, build), _, _, hx = model.act(
                    obs_to_torch(obs, device),
                    hx=hx,
                    masks=masks_to_torch(mask, device),
                    deterministic=deterministic,
                )
                action = tuple(int(x.item()) for x in (at, tgt, cx, cy, frac, build))
            step = env.step(action)
            total_r += step.reward
            last_info = step.info
            obs, mask = step.obs, step.mask
            if step.done:
                done = True
                episodes_finished += 1
                break
        tiles.append(float(last_info.get("agentTiles", 0)))
        nation_tiles.append(float(last_info.get("nationTiles", 0)))
        enemy_tiles.append(float(last_info.get("enemyTilesTotal", last_info.get("nationTiles", 0))))
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
        seed_results.append(
            {
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
        )
    torch.random.set_rng_state(rng_state)
    n = len(seeds)
    return {
        "wins": wins,
        "losses": losses,
        "episodes_finished": episodes_finished,
        "evaluated": n,
        "win_rate": wins / n if n else 0.0,
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
        "policy_mode": "deterministic" if deterministic else "stochastic",
        "seed_results": seed_results,
    }




def apply_env_scale(
    train_vec: VecEnv,
    eval_env: OpenFrontEnv,
    difficulty_index: int,
    *,
    use_scale_table: bool,
) -> dict:
    """Apply difficulty (+ optional map/nations/bots scale) to train and eval envs."""
    difficulty = DIFFICULTIES[difficulty_index]
    if use_scale_table:
        scale = scale_for_difficulty(difficulty_index)
        train_vec.set_scale(
            map_name=scale.map_name,
            nations=scale.nations,
            bots=scale.bots,
            difficulty=difficulty,
        )
        eval_env.set_scale(
            map_name=scale.map_name,
            nations=scale.nations,
            bots=scale.bots,
            difficulty=difficulty,
        )
        return {
            "difficulty": difficulty,
            "map": scale.map_name,
            "nations": scale.nations,
            "bots": scale.bots,
        }
    train_vec.set_difficulty(difficulty)
    eval_env.set_difficulty(difficulty)
    return {"difficulty": difficulty}


def save_checkpoint(
    model: MacroMicroPolicy,
    ckpt: Path,
    *,
    stride: int,
    map_name: str,
    step: int,
    league: int,
    opt: torch.optim.Optimizer | None = None,
    best_ckpt: Path | None = None,
    is_best: bool = False,
    rolling_every: int = 0,
) -> None:
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "league": league,
        "stride": stride,
        "map": map_name,
        "ppo_step": step,
    }
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(ckpt)
    print(f"Saved {ckpt} (ppo_step={step})", flush=True)
    if is_best and best_ckpt is not None:
        best_tmp = best_ckpt.with_suffix(best_ckpt.suffix + ".tmp")
        torch.save(payload, best_tmp)
        best_tmp.replace(best_ckpt)
        print(f"Saved best {best_ckpt} (ppo_step={step})", flush=True)
    if rolling_every > 0 and step % rolling_every == 0:
        roll = ckpt.parent / f"policy_step{step}.pt"
        torch.save(payload, roll)
        print(f"Saved rolling {roll}", flush=True)


def maybe_poll_long_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def spawn_long_eval(
    *,
    ckpt: Path,
    map_name: str,
    stride: int,
    nations: int,
    bots: int,
    difficulty: str,
    max_ticks: int,
    out: Path,
) -> subprocess.Popen | None:
    script = PY_ROOT / "scripts" / "long_eval.py"
    if not script.exists():
        return None
    cmd = [
        sys.executable,
        str(script),
        "--ckpt",
        str(ckpt),
        "--map",
        map_name,
        "--stride",
        str(stride),
        "--nations",
        str(nations),
        "--bots",
        str(bots),
        "--difficulty",
        difficulty,
        "--max-ticks",
        str(max_ticks),
        "--out",
        str(out),
    ]
    log_path = AI_ROOT / "fixtures" / "logs" / "long_eval_async.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a")
    print(f"Spawning async long_eval → {out}", flush=True)
    return subprocess.Popen(
        cmd,
        cwd=str(AI_ROOT.parent),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="onion")
    ap.add_argument(
        "--demos",
        default=str(AI_ROOT / "fixtures" / "demos" / "onion_ffa.jsonl"),
        help="Expert demo JSONL for BC (default: onion_ffa.jsonl)",
    )
    ap.add_argument("--bc-epochs", type=int, default=4)
    ap.add_argument(
        "--resume-bc-epochs",
        type=int,
        default=1,
        help="BC anchor epochs after loading a checkpoint (default 1)",
    )
    ap.add_argument("--bc-batch", type=int, default=32)
    ap.add_argument(
        "--ppo-steps",
        type=int,
        default=8,
        help="PPO iterations; ignored when --forever is set",
    )
    ap.add_argument(
        "--forever",
        action="store_true",
        help="Keep running PPO indefinitely, saving checkpoints as it goes",
    )
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--ent-coef",
        type=float,
        default=0.03,
        help="Causal action entropy bonus used by PPO",
    )
    ap.add_argument(
        "--ent-coef-floor",
        type=float,
        default=0.01,
        help="Minimum entropy coefficient (prevents collapse)",
    )
    ap.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="Approx-KL early stop threshold for PPO epochs",
    )
    ap.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument(
        "--nations",
        type=int,
        default=3,
        help="Manifest nations in headless GameEnv (Easy-scale default: 3)",
    )
    ap.add_argument(
        "--bots",
        type=int,
        default=2,
        help="Bot tribes in headless GameEnv (Easy-scale default: 2)",
    )
    ap.add_argument(
        "--max-ticks",
        type=int,
        default=8_000,
        help="Episode truncation in core ticks (default: 8000)",
    )
    ap.add_argument("--episode-rollouts", action="store_true")
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=1600,
        help="Macro-steps per eval seed (default 1600 ≈ 8000 ticks at stride 5)",
    )
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument(
        "--difficulty",
        choices=("Easy", "Medium", "Hard", "Impossible"),
        default="Easy",
    )
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument(
        "--resume-curriculum",
        action="store_true",
        help="Restore difficulty/promote/demote streaks from train_meta.json on --resume",
    )
    ap.add_argument(
        "--nation-demos",
        default=None,
        help="Optional Nation demonstration JSONL mixed into BC batches",
    )
    ap.add_argument("--bc-nation-alpha", type=float, default=0.0)
    ap.add_argument(
        "--city-demos",
        default=str(AI_ROOT / "fixtures" / "demos" / "world_city.jsonl"),
        help="City-economy expert demos mixed into BC (empty string to disable)",
    )
    ap.add_argument(
        "--bc-city-alpha",
        type=float,
        default=0.0,
        help="Fraction of each BC batch drawn from --city-demos",
    )
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
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument(
        "--save-every",
        type=int,
        default=4,
        help="Write policy.pt every N PPO iterations (default 4)",
    )
    ap.add_argument(
        "--rolling-every",
        type=int,
        default=50,
        help="Write policy_step{N}.pt every N PPO iterations (0=off)",
    )
    ap.add_argument(
        "--long-eval-every",
        type=int,
        default=0,
        help="Spawn async long_eval.py every N PPO steps (0=off)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Load --ckpt before training (skips BC if weights load)",
    )
    ap.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights but discard stale optimizer momentum",
    )
    ap.add_argument(
        "--no-onnx",
        action="store_true",
        help="Skip final ONNX export (useful for long forever runs)",
    )
    ap.add_argument(
        "--metrics",
        default=None,
        help="Append per-step training metrics as JSONL (default: checkpoint directory)",
    )
    args = ap.parse_args()
    if not 0.0 <= args.bc_nation_alpha <= 1.0:
        ap.error("--bc-nation-alpha must be between 0 and 1")
    if not 0.0 <= args.bc_city_alpha <= 1.0:
        ap.error("--bc-city-alpha must be between 0 and 1")
    if args.bc_nation_alpha + args.bc_city_alpha > 1.0:
        ap.error("--bc-nation-alpha + --bc-city-alpha must be <= 1")
    if args.nations < 1 or args.bots < 0:
        ap.error("--nations must be >= 1 and --bots must be >= 0")
    if args.nations + args.bots > 72:
        ap.error("--nations + --bots must be <= 72 policy target slots")
    if args.max_ticks < 1:
        ap.error("--max-ticks must be >= 1")

    seed_everything(args.seed)
    device = torch.device(args.device)
    model = MacroMicroPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt = Path(args.ckpt)
    best_ckpt = ckpt.parent / "policy_best.pt"
    long_eval_path = ckpt.parent / "long_eval_latest.json"
    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else ckpt.parent / "training_metrics.jsonl"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    start_step = 0
    resumed = False
    meta_path = ckpt.parent / "train_meta.json"
    restored_curriculum: dict | None = None
    best_eval_win: dict[str, float] = {}
    long_eval_proc: subprocess.Popen | None = None

    if args.resume and ckpt.exists():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        skipped = load_policy_state_dict(model, state)[1]
        if skipped:
            args.reset_optimizer = True
        if (
            isinstance(blob, dict)
            and "optimizer" in blob
            and not args.reset_optimizer
        ):
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as e:
                print(f"Optimizer state not restored: {e}", flush=True)
        elif args.reset_optimizer:
            print("Optimizer state reset", flush=True)
        if isinstance(blob, dict):
            start_step = int(blob.get("ppo_step", 0)) + 1
        resumed = True
        print(f"Resumed {ckpt} from ppo_step={start_step - 1}", flush=True)
        if args.resume_curriculum and meta_path.exists():
            try:
                restored_curriculum = json.loads(meta_path.read_text())
            except Exception as e:
                print(f"Curriculum meta not restored: {e}", flush=True)

    demos_path = Path(args.demos)
    if demos_path.exists():
        demos = rebalance_demos(load_jsonl_demos(demos_path))
        print(
            f"Loaded {len(demos)} expert transitions from {demos_path} (NOOP-rebalanced)",
            flush=True,
        )
    else:
        demos = []
        print(
            f"No demos at {demos_path}; BC will use expert_step via env if TS available",
            flush=True,
        )

    city_demos: list[dict] = []
    if args.city_demos:
        city_demos_path = Path(args.city_demos)
        if city_demos_path.exists():
            city_demos = rebalance_demos(load_jsonl_demos(city_demos_path))
            print(
                f"Loaded {len(city_demos)} city-economy transitions from {city_demos_path} (NOOP-rebalanced)",
                flush=True,
            )
        else:
            print(f"No city demos at {city_demos_path}; skipping city BC mix", flush=True)

    nation_demos: list[dict] = []
    if args.nation_demos:
        nation_demos_path = Path(args.nation_demos)
        if nation_demos_path.exists():
            nation_demos = rebalance_demos(load_jsonl_demos(nation_demos_path))
            print(
                f"Loaded {len(nation_demos)} Nation transitions from {nation_demos_path} (NOOP-rebalanced)",
                flush=True,
            )
        else:
            print(f"No Nation demos at {nation_demos_path}; skipping mixed BC", flush=True)

    print(
        f"TS env: {not args.no_ts} stride={args.stride} "
        f"difficulty={args.difficulty} nations={args.nations} "
        f"bots={args.bots} max_ticks={args.max_ticks} seed={args.seed} "
        f"num_envs={args.num_envs}",
        flush=True,
    )
    bc_epochs = args.resume_bc_epochs if resumed else args.bc_epochs
    print("Phase 1: behavior cloning", flush=True)
    if (demos or city_demos or nation_demos) and bc_epochs > 0:
        for epoch in range(bc_epochs):
            source_demos = demos or city_demos or nation_demos
            random.shuffle(source_demos)
            losses = []
            for i in range(0, max(len(source_demos), args.bc_batch), args.bc_batch):
                if demos or city_demos or nation_demos:
                    chunk = mix_demo_sources(
                        demos,
                        city_demos,
                        nation_demos,
                        args.bc_batch,
                        args.bc_city_alpha if city_demos else 0.0,
                        args.bc_nation_alpha if nation_demos else 0.0,
                    )
                else:
                    chunk = source_demos[i : i + args.bc_batch]
                if len(chunk) < 2:
                    continue
                batch = stack_transitions(chunk)
                loss = bc_loss(model, batch, device)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            print(f"  bc {epoch}: loss={np.mean(losses):.4f}", flush=True)
    else:
        reason = "resumed from checkpoint" if resumed else "no demos"
        print(f"  skipped ({reason})", flush=True)

    print(
        f"Phase 2: factorized masked PPO vs {args.nations} nations + "
        f"{args.bots} bots ({args.difficulty})"
        + (" (forever)" if args.forever else ""),
        flush=True,
    )
    train_vec = make_parallel_env(
        args.num_envs,
        map_name=args.map,
        seed=1,
        stride=args.stride,
        use_ts=not args.no_ts,
        difficulty=args.difficulty,
        nations=args.nations,
        bots=args.bots,
        max_ticks=args.max_ticks,
    )
    eval_env = OpenFrontEnv(
        map_name=args.map,
        seed=42,
        use_ts=not args.no_ts,
        stride=args.stride,
        difficulty=args.difficulty,
        nations=args.nations,
        bots=args.bots,
        max_ticks=args.max_ticks,
    )
    historic: list[dict] = []
    last_eval: dict | None = None
    step = start_step
    difficulty_index = DIFFICULTIES.index(args.difficulty)
    promote_streak = 0
    demote_streak = 0
    if restored_curriculum and args.resume_curriculum:
        saved_diff = restored_curriculum.get("difficulty")
        if saved_diff in DIFFICULTIES:
            difficulty_index = DIFFICULTIES.index(saved_diff)
        promote_streak = int(restored_curriculum.get("promote_streak", 0))
        demote_streak = int(restored_curriculum.get("demote_streak", 0))
        print(
            f"Restored curriculum difficulty={DIFFICULTIES[difficulty_index]} "
            f"promote={promote_streak} demote={demote_streak}",
            flush=True,
        )
    # Curriculum owns map/nations/bots via the scale table; CLI overrides when off.
    active_scale = apply_env_scale(
        train_vec,
        eval_env,
        difficulty_index,
        use_scale_table=args.curriculum,
    )
    if not args.curriculum:
        active_scale = {
            "difficulty": args.difficulty,
            "map": args.map,
            "nations": args.nations,
            "bots": args.bots,
        }
    print(
        f"Curriculum start difficulty={DIFFICULTIES[difficulty_index]} "
        f"scale={active_scale} "
        f"thresholds={CURRICULUM_THRESHOLDS} "
        f"promote_streaks={CURRICULUM_PROMOTE_STREAK} "
        f"demote_floors={CURRICULUM_DEMOTE_FLOORS} "
        f"easy_mastery=wr>={EASY_MASTERY_MIN_WIN_RATE}/place<={EASY_MASTERY_MAX_PLACEMENT} "
        f"scales={list(CURRICULUM_SCALES)}",
        flush=True,
    )

    try:
        while True:
            if not args.forever and step - start_step >= args.ppo_steps:
                break
            train_difficulty = DIFFICULTIES[difficulty_index]
            timer = StepTimer()
            if args.num_envs > 1 and not args.episode_rollouts:
                trajs, last_values, segment_lengths, rollout = collect_vec_rollout(
                    train_vec,
                    model,
                    device,
                    args.horizon,
                    seed=1000 + step * 100,
                    timer=timer,
                )
                rollout_stats = {**rollout}
            else:
                trajs = []
                last_values = []
                segment_lengths = []
                rollout_stats = {
                    "episodes": 0,
                    "wins": 0,
                    "losses": 0,
                    "city_builds": 0,
                    "city_build_intents": 0,
                    "max_cities": 0,
                }
                for ei, env in enumerate(train_vec.envs):
                    traj, last_value, rollout = collect_rollout(
                        env,
                        model,
                        device,
                        args.horizon,
                        seed=1000 + step * 100 + ei,
                        episode_rollouts=args.episode_rollouts,
                        timer=timer,
                    )
                    trajs.extend(traj)
                    last_values.append(last_value)
                    segment_lengths.append(len(traj))
                    for key in (
                        "episodes",
                        "wins",
                        "losses",
                        "city_builds",
                        "city_build_intents",
                    ):
                        rollout_stats[key] += rollout.get(key, 0)
                    rollout_stats["max_cities"] = max(
                        rollout_stats["max_cities"], rollout.get("max_cities", 0)
                    )

            t_ppo = time.perf_counter()
            stats = ppo_update(
                model,
                opt,
                trajs,
                device,
                last_values if len(last_values) > 1 else (last_values[0] if last_values else 0.0),
                segment_lengths if len(last_values) > 1 else None,
                ent_coef=args.ent_coef,
                ent_coef_floor=args.ent_coef_floor,
                target_kl=args.target_kl,
            )
            timer.ppo_update_s += time.perf_counter() - t_ppo
            rollout_stats["win_rate"] = (
                rollout_stats["wins"] / rollout_stats["episodes"]
                if rollout_stats["episodes"]
                else 0.0
            )
            msg = (
                f"  ppo {step} ({train_difficulty}): "
                f"loss={stats['loss']:.4f} policy={stats['policy']:.4f} "
                f"value={stats['value']:.4f} ent={stats['ent']:.4f} "
                f"kl={stats.get('approx_kl', 0):.4f} "
                f"ret={stats.get('return_mean', 0):.1f}±"
                f"{stats.get('return_std', 0):.1f} "
                f"ev={stats.get('explained_variance', 0):.2f} "
                f"sps={timer.as_dict()['steps_per_sec']:.2f} "
                f"cities={rollout_stats['max_cities']}"
                f"/{rollout_stats['city_build_intents']}"
            )
            curriculum_event = None
            is_best = False
            if step % args.eval_every == 0:
                seed0 = 42 + ((step // max(1, args.eval_every)) * args.eval_seeds)
                eval_seeds = list(range(seed0, seed0 + args.eval_seeds))
                t_eval = time.perf_counter()
                last_eval = evaluate(
                    eval_env,
                    model,
                    device,
                    seeds=eval_seeds,
                    steps=args.eval_steps,
                    deterministic=False,
                )
                timer.eval_s += time.perf_counter() - t_eval
                last_eval["difficulty"] = train_difficulty
                eval_mode = last_eval.get("policy_mode", "stochastic")
                msg += (
                    f" | eval[{train_difficulty}/{eval_mode}] "
                    f"win_rate={last_eval['win_rate']:.2f} "
                    f"tiles={last_eval['mean_tiles']:.0f}/"
                    f"{last_eval.get('mean_enemy_tiles', last_eval.get('mean_nation_tiles', 0)):.0f} "
                    f"alive={last_eval.get('mean_opponents_alive', 0):.0f} "
                    f"place={last_eval.get('mean_placement', 1):.1f} "
                    f"cities={last_eval.get('mean_cities', 0):.1f} "
                    f"boats={last_eval.get('mean_boats', 0):.1f} "
                    f"ratio={last_eval['mean_troop_ratio']:.2f} "
                    f"return={last_eval['mean_return']:.1f}"
                )
                wr = float(last_eval["win_rate"])
                prev_best = best_eval_win.get(train_difficulty, -1.0)
                if wr > prev_best:
                    best_eval_win[train_difficulty] = wr
                    is_best = True
                long_result = maybe_poll_long_eval(long_eval_path)
                curriculum_wr = wr
                curriculum_placement = float(last_eval.get("mean_placement", 1.0))
                # Prefer finished-episode count when available; otherwise any
                # scored seeds (including truncated placement wins) drive curriculum.
                curriculum_evaluated = int(
                    last_eval.get("evaluated", last_eval["episodes_finished"])
                )
                used_long_eval = False
                if (
                    long_result
                    and long_result.get("difficulty") == train_difficulty
                    and long_result.get("win_rate") is not None
                ):
                    curriculum_wr = float(long_result["win_rate"])
                    curriculum_placement = float(
                        long_result.get("mean_placement", curriculum_placement)
                    )
                    curriculum_evaluated = int(
                        long_result.get(
                            "evaluated",
                            long_result.get(
                                "episodes_finished",
                                len(long_result.get("seeds", [])) or 1,
                            ),
                        )
                    )
                    used_long_eval = True
                    msg += f" | long_eval wr={curriculum_wr:.2f}"
                if args.curriculum:
                    if curriculum_evaluated > 0:
                        apply_curriculum = True
                        if difficulty_index == 0 and not easy_mastery_reached(
                            curriculum_wr, curriculum_placement
                        ):
                            apply_curriculum = False
                            msg += (
                                " | curriculum=frozen"
                                f" (Easy mastery wr>={EASY_MASTERY_MIN_WIN_RATE}"
                                f" place<={EASY_MASTERY_MAX_PLACEMENT})"
                            )
                        elif not used_long_eval:
                            t_det = time.perf_counter()
                            det_eval = evaluate(
                                eval_env,
                                model,
                                device,
                                seeds=eval_seeds,
                                steps=args.eval_steps,
                                deterministic=True,
                            )
                            timer.eval_s += time.perf_counter() - t_det
                            curriculum_wr = float(det_eval["win_rate"])
                            curriculum_placement = float(
                                det_eval.get("mean_placement", 1.0)
                            )
                            curriculum_evaluated = int(
                                det_eval.get(
                                    "evaluated", det_eval["episodes_finished"]
                                )
                            )
                            msg += (
                                f" | det wr={curriculum_wr:.2f}"
                                f" place={curriculum_placement:.1f}"
                            )
                            if (
                                difficulty_index == 0
                                and not easy_mastery_reached(
                                    curriculum_wr, curriculum_placement
                                )
                            ):
                                apply_curriculum = False
                                msg += " | curriculum=frozen (deterministic Easy gate)"
                        if apply_curriculum:
                            prev_index = difficulty_index
                            (
                                difficulty_index,
                                promote_streak,
                                demote_streak,
                                curriculum_event,
                            ) = update_curriculum(
                                difficulty_index=difficulty_index,
                                win_rate=curriculum_wr,
                                promote_streak=promote_streak,
                                demote_streak=demote_streak,
                            )
                            if curriculum_event == "promoted":
                                active_scale = apply_env_scale(
                                    train_vec,
                                    eval_env,
                                    difficulty_index,
                                    use_scale_table=True,
                                )
                                msg += (
                                    f" | promoted={active_scale['difficulty']}"
                                    f" map={active_scale['map']}"
                                    f" n+b={active_scale['nations']}+{active_scale['bots']}"
                                )
                            elif curriculum_event == "demoted":
                                active_scale = apply_env_scale(
                                    train_vec,
                                    eval_env,
                                    difficulty_index,
                                    use_scale_table=True,
                                )
                                # Keep writing policy_best.pt for manual rollback, but
                                # do not auto-restore on demote — that wiped Medium
                                # learning with an Easy specialist.
                                msg += (
                                    f" | demoted={active_scale['difficulty']}"
                                    f"(from {DIFFICULTIES[prev_index]})"
                                    f" map={active_scale['map']}"
                                )
                            elif promote_streak:
                                need = promote_streak_required(difficulty_index)
                                msg += f" | streak={promote_streak}/{need}"
                            elif demote_streak:
                                msg += (
                                    f" | demote_streak={demote_streak}/"
                                    f"{CURRICULUM_DEMOTE_STREAK}"
                                )
                    else:
                        msg += " | curriculum=held (no completed proxy episodes)"
                historic.append(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                )
                if len(historic) > args.league_size:
                    historic.pop(0)
            print(msg, flush=True)
            timing = timer.as_dict()
            metrics = {
                "ppo_step": step,
                "wall_clock": datetime.now(timezone.utc).isoformat(),
                "loss": stats["loss"],
                "policy_loss": stats["policy"],
                "value_loss": stats["value"],
                "entropy": stats["ent"],
                "approx_kl": stats.get("approx_kl", 0.0),
                "grad_norm": stats.get("grad_norm", 0.0),
                "return_mean": stats.get("return_mean", 0.0),
                "return_std": stats.get("return_std", 0.0),
                "explained_variance": stats.get("explained_variance", 0.0),
                "difficulty": DIFFICULTIES[difficulty_index],
                "train_difficulty": train_difficulty,
                "scale": active_scale,
                "curriculum": args.curriculum,
                "promote_streak": promote_streak,
                "demote_streak": demote_streak,
                "curriculum_event": curriculum_event,
                "rollout": {**rollout_stats, "difficulty": train_difficulty},
                "eval": last_eval if step % args.eval_every == 0 else None,
                "timing": timing,
            }
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(json.dumps(metrics) + "\n")

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    model,
                    ckpt,
                    stride=args.stride,
                    map_name=args.map,
                    step=step,
                    league=len(historic),
                    opt=opt,
                    best_ckpt=best_ckpt,
                    is_best=is_best,
                    rolling_every=args.rolling_every,
                )
                meta = {
                    "map": args.map,
                    "bc_epochs": bc_epochs,
                    "ppo_step": step,
                    "ppo_steps": None if args.forever else args.ppo_steps,
                    "forever": args.forever,
                    "horizon": args.horizon,
                    "stride": args.stride,
                    "nations": args.nations,
                    "bots": args.bots,
                    "max_ticks": args.max_ticks,
                    "num_envs": args.num_envs,
                    "seed": args.seed,
                    "difficulty": DIFFICULTIES[difficulty_index],
                    "curriculum": args.curriculum,
                    "promote_streak": promote_streak,
                    "demote_streak": demote_streak,
                    "league_checkpoints": len(historic),
                    "demos": str(demos_path) if demos else None,
                    "city_demos": str(args.city_demos) if city_demos else None,
                    "env": "ts" if not args.no_ts else "stub",
                    "last_eval": last_eval,
                    "best_eval_win": best_eval_win,
                    "metrics": str(metrics_path),
                    "timing": timing,
                }
                meta_path.write_text(json.dumps(meta, indent=2))

            if (
                args.long_eval_every > 0
                and step > 0
                and step % args.long_eval_every == 0
            ):
                if long_eval_proc is None or long_eval_proc.poll() is not None:
                    long_eval_proc = spawn_long_eval(
                        ckpt=ckpt,
                        map_name=args.map,
                        stride=args.stride,
                        nations=args.nations,
                        bots=args.bots,
                        difficulty=train_difficulty,
                        max_ticks=args.max_ticks,
                        out=long_eval_path,
                    )
            step += 1
    except KeyboardInterrupt:
        print("Interrupted — saving final checkpoint", flush=True)
        save_checkpoint(
            model,
            ckpt,
            stride=args.stride,
            map_name=args.map,
            step=max(start_step, step - 1),
            league=len(historic),
            opt=opt,
            best_ckpt=best_ckpt,
            is_best=False,
        )

    if not args.forever:
        save_checkpoint(
            model,
            ckpt,
            stride=args.stride,
            map_name=args.map,
            step=max(start_step, step - 1),
            league=len(historic),
            opt=opt,
            best_ckpt=best_ckpt,
            is_best=False,
        )
        if not args.no_onnx:
            try:
                export_onnx(model.cpu(), args.onnx)
                print(f"Exported ONNX {args.onnx}", flush=True)
            except Exception as e:
                print(f"ONNX export skipped: {e}", flush=True)

    train_vec.close()
    eval_env.close()


if __name__ == "__main__":
    main()
