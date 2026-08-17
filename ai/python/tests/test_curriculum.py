"""Unit tests for curriculum promote/demote helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openfront_ai.curriculum import (
    CURRICULUM_DEMOTE_STREAK,
    CURRICULUM_PROMOTE_STREAK,
    CURRICULUM_SCALES,
    classify_eval_outcome,
    easy_mastery_reached,
    mix_demo_sources,
    promote_streak_required,
    scale_for_difficulty,
    update_curriculum,
)


def test_classify_eval_outcome_finished_and_truncated():
    assert (
        classify_eval_outcome(done=True, winner="agent", placement=1) == "win"
    )
    assert (
        classify_eval_outcome(done=True, winner="nation-0", placement=2) == "loss"
    )
    # Truncated place-1 with tile lead → win.
    assert (
        classify_eval_outcome(
            done=False,
            winner=None,
            placement=1,
            agent_tiles=80_000,
            strongest_enemy_tiles=50_000,
        )
        == "win"
    )
    # Truncated place-1 without tile lead → loss (tiny foothold).
    assert (
        classify_eval_outcome(
            done=False,
            winner=None,
            placement=1,
            agent_tiles=21_000,
            strongest_enemy_tiles=70_000,
        )
        == "loss"
    )
    assert (
        classify_eval_outcome(done=False, winner=None, placement=3) == "loss"
    )


def test_hard_requires_longer_promote_streak():
    assert promote_streak_required(0) == 2
    assert promote_streak_required(1) == 2
    assert promote_streak_required(2) == 3
    assert CURRICULUM_PROMOTE_STREAK[2] == 3


def test_promote_after_required_streak():
    idx, promo, demo, event = 0, 0, 0, None
    for _ in range(2):
        idx, promo, demo, event = update_curriculum(
            difficulty_index=idx,
            win_rate=0.9,
            promote_streak=promo,
            demote_streak=demo,
        )
    assert event == "promoted"
    assert idx == 1
    assert promo == 0
    assert demo == 0


def test_hard_needs_three_consecutive_evals():
    idx, promo, demo, event = 2, 0, 0, None
    for _ in range(2):
        idx, promo, demo, event = update_curriculum(
            difficulty_index=idx,
            win_rate=0.7,
            promote_streak=promo,
            demote_streak=demo,
        )
    assert event is None
    assert idx == 2
    assert promo == 2
    idx, promo, demo, event = update_curriculum(
        difficulty_index=idx,
        win_rate=0.7,
        promote_streak=promo,
        demote_streak=demo,
    )
    assert event == "promoted"
    assert idx == 3


def test_demote_one_tier_after_sustained_failure():
    idx, promo, demo, event = 2, 1, 0, None
    idx, promo, demo, event = update_curriculum(
        difficulty_index=idx,
        win_rate=0.0,
        promote_streak=promo,
        demote_streak=demo,
    )
    assert event is None
    assert idx == 2
    assert promo == 0
    assert demo == 1
    idx, promo, demo, event = update_curriculum(
        difficulty_index=idx,
        win_rate=0.1,
        promote_streak=promo,
        demote_streak=demo,
    )
    assert event == "demoted"
    assert idx == 1
    assert promo == 0
    assert demo == 0
    assert CURRICULUM_DEMOTE_STREAK == 2


def test_never_demotes_below_easy():
    idx, promo, demo, event = update_curriculum(
        difficulty_index=0,
        win_rate=0.0,
        promote_streak=0,
        demote_streak=5,
    )
    assert event is None
    assert idx == 0


def test_mix_demo_sources_respects_alphas():
    expert = [{"action": (2, 0, 0, 0, 0, 0)} for _ in range(10)]
    city = [{"action": (6, 0, 0, 0, 0, 0)} for _ in range(10)]
    nation = [{"action": (2, 1, 0, 0, 0, 0)} for _ in range(10)]
    chunk = mix_demo_sources(
        expert, city, nation, batch_size=20, city_alpha=0.5, nation_alpha=0.25
    )
    assert len(chunk) == 20
    city_n = sum(1 for d in chunk if d["action"][0] == 6)
    assert 4 <= city_n <= 16


def test_curriculum_scale_table():
    assert CURRICULUM_SCALES[0] == ("onion", 3, 2)
    assert CURRICULUM_SCALES[1] == ("onion", 5, 4)
    assert CURRICULUM_SCALES[2] == ("onion", 8, 8)
    assert CURRICULUM_SCALES[3] == ("world", 12, 12)
    assert scale_for_difficulty(2) == ("onion", 8, 8)
    assert scale_for_difficulty(3) == ("world", 12, 12)
    assert scale_for_difficulty(-1) == CURRICULUM_SCALES[0]
    assert scale_for_difficulty(99) == CURRICULUM_SCALES[-1]


def test_promote_advances_scale_tier():
    idx, promo, demo, event = 0, 0, 0, None
    for _ in range(2):
        idx, promo, demo, event = update_curriculum(
            difficulty_index=idx,
            win_rate=0.9,
            promote_streak=promo,
            demote_streak=demo,
        )
    assert event == "promoted"
    assert scale_for_difficulty(idx).map_name == "onion"
    assert scale_for_difficulty(idx).nations == 5
    assert scale_for_difficulty(idx).bots == 4


def test_easy_mastery_gate():
    assert easy_mastery_reached(0.5, 1.3)
    assert easy_mastery_reached(1.0, 1.0)
    assert not easy_mastery_reached(0.49, 1.0)
    assert not easy_mastery_reached(0.8, 1.31)
    assert not easy_mastery_reached(0.17, 1.8)

