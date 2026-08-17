"""Curriculum promote/demote helpers for PPO training."""

from __future__ import annotations

import random
from typing import Any, NamedTuple

DIFFICULTIES = ("Easy", "Medium", "Hard", "Impossible")
# Require clear mastery before promoting. Hard needs a longer streak than Easy/Medium.
CURRICULUM_THRESHOLDS = (0.8, 0.7, 0.65)  # Easy, Medium, Hard
CURRICULUM_PROMOTE_STREAK = (2, 2, 3)  # Hard must clear 3 consecutive evals
# Demote after sustained failure below these floors (Easy..Impossible).
CURRICULUM_DEMOTE_FLOORS = (0.35, 0.30, 0.25, 0.15)
CURRICULUM_DEMOTE_STREAK = 2
# Stay on Easy until a 6-seed eval is actually competitive. Fake 2-seed
# truncated 100% scores previously yo-yoed the policy into Medium.
EASY_MASTERY_MIN_WIN_RATE = 0.5
EASY_MASTERY_MAX_PLACEMENT = 1.3


class ScaleTier(NamedTuple):
    map_name: str
    nations: int
    bots: int


# Map / opponent count scale paired with difficulty tiers.
# Easy/Medium/Hard stay on onion with a soft opponent ramp; Impossible goes world.
CURRICULUM_SCALES: tuple[ScaleTier, ...] = (
    ScaleTier("onion", 3, 2),  # Easy
    ScaleTier("onion", 5, 4),  # Medium
    ScaleTier("onion", 8, 8),  # Hard — master crowded onion before world
    ScaleTier("world", 12, 12),  # Impossible
)


def scale_for_difficulty(difficulty_index: int) -> ScaleTier:
    if difficulty_index < 0:
        return CURRICULUM_SCALES[0]
    if difficulty_index >= len(CURRICULUM_SCALES):
        return CURRICULUM_SCALES[-1]
    return CURRICULUM_SCALES[difficulty_index]


def promote_streak_required(difficulty_index: int) -> int:
    if difficulty_index < 0 or difficulty_index >= len(CURRICULUM_PROMOTE_STREAK):
        return CURRICULUM_PROMOTE_STREAK[-1]
    return CURRICULUM_PROMOTE_STREAK[difficulty_index]


def easy_mastery_reached(win_rate: float, mean_placement: float) -> bool:
    """True when Easy eval is strong enough to unfreeze curriculum."""
    return (
        float(win_rate) >= EASY_MASTERY_MIN_WIN_RATE
        and float(mean_placement) <= EASY_MASTERY_MAX_PLACEMENT
    )


def demote_floor(difficulty_index: int) -> float:
    if difficulty_index < 0 or difficulty_index >= len(CURRICULUM_DEMOTE_FLOORS):
        return CURRICULUM_DEMOTE_FLOORS[-1]
    return CURRICULUM_DEMOTE_FLOORS[difficulty_index]


def classify_eval_outcome(
    *,
    done: bool,
    winner: str | None,
    placement: int,
    agent_tiles: int = 0,
    strongest_enemy_tiles: int = 0,
) -> str:
    """Classify one eval seed as ``win`` or ``loss``.

    Finished episodes use the true winner. Truncated episodes (eval step
    budget exhausted before ``done``) count as a win only when the agent is
    placement 1 *and* holds at least as many tiles as the strongest enemy —
    place-1 with a tiny foothold is not mastery.
    """
    if done:
        return "win" if winner == "agent" else "loss"
    if int(placement) <= 1 and int(agent_tiles) >= int(strongest_enemy_tiles):
        return "win"
    return "loss"


def update_curriculum(
    *,
    difficulty_index: int,
    win_rate: float,
    promote_streak: int,
    demote_streak: int,
) -> tuple[int, int, int, str | None]:
    """Return (new_index, promote_streak, demote_streak, event).

    event is 'promoted', 'demoted', or None. Never jumps more than one tier.
    """
    event: str | None = None
    # Demotion first so a collapsing Hard policy steps to Medium, not Easy.
    if difficulty_index > 0 and win_rate < demote_floor(difficulty_index):
        demote_streak += 1
        promote_streak = 0
        if demote_streak >= CURRICULUM_DEMOTE_STREAK:
            difficulty_index -= 1
            demote_streak = 0
            promote_streak = 0
            event = "demoted"
        return difficulty_index, promote_streak, demote_streak, event

    demote_streak = 0
    if (
        difficulty_index < len(CURRICULUM_THRESHOLDS)
        and win_rate >= CURRICULUM_THRESHOLDS[difficulty_index]
    ):
        promote_streak += 1
        if promote_streak >= promote_streak_required(difficulty_index):
            difficulty_index += 1
            promote_streak = 0
            demote_streak = 0
            event = "promoted"
    else:
        promote_streak = 0
    return difficulty_index, promote_streak, demote_streak, event


def mix_demo_sources(
    expert: list[dict[str, Any]],
    city: list[dict[str, Any]],
    nation: list[dict[str, Any]],
    batch_size: int,
    city_alpha: float,
    nation_alpha: float,
) -> list[dict[str, Any]]:
    """Sample a mixed BC batch. city_alpha/nation_alpha are fractions of batch."""
    city_alpha = max(0.0, min(1.0, city_alpha))
    nation_alpha = max(0.0, min(1.0 - city_alpha, nation_alpha))
    city_n = round(batch_size * city_alpha) if city else 0
    nation_n = round(batch_size * nation_alpha) if nation else 0
    expert_n = max(0, batch_size - city_n - nation_n)
    chunk: list[dict[str, Any]] = []
    if expert and expert_n:
        chunk.extend(random.choices(expert, k=expert_n))
    if city and city_n:
        chunk.extend(random.choices(city, k=city_n))
    if nation and nation_n:
        chunk.extend(random.choices(nation, k=nation_n))
    if not chunk:
        source = expert or city or nation
        chunk = (
            random.choices(source, k=min(batch_size, len(source))) if source else []
        )
    random.shuffle(chunk)
    return chunk
