"""Unit tests for shaped reward city component."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openfront_ai.rewards import shaped_reward_torch


def test_city_gain_is_rewarded_once():
    base = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_cities=0.0,
    )
    gained = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_cities=1.0,
        lambda_city=12.0,
    )
    assert abs((gained - base) - 12.0) < 1e-6


def test_city_loss_does_not_farm_negative_as_positive():
    r = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_cities=-1.0,
        lambda_city=12.0,
    )
    assert r == 0.0


def test_city_reward_below_terminal_win():
    city = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_cities=1.0,
        lambda_city=12.0,
        lambda_w=300.0,
    )
    win = shaped_reward_torch(
        win=True,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_cities=0.0,
        lambda_w=300.0,
    )
    assert city < win


def test_boat_launch_is_small_positive():
    base = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
    )
    launched = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        boat_launched=True,
        lambda_boat_launch=1.0,
    )
    assert abs((launched - base) - 1.0) < 1e-6
    assert launched < 300.0


def test_placement_improvement_is_rewarded():
    base = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
    )
    improved = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_placement=1.0,
        lambda_placement=20.0,
    )
    assert abs((improved - base) - 20.0) < 1e-6


def test_tile_lead_gain_matches_land_scale():
    base = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
    )
    lead = shaped_reward_torch(
        win=None,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        d_tile_lead=0.1,
        lambda_tile_lead=20.0,
    )
    assert abs((lead - base) - 2.0) < 1e-6
    win = shaped_reward_torch(
        win=True,
        d_land=0.0,
        d_troop_diff_norm=0.0,
        d_gold_norm=0.0,
        boat_sunk=False,
        lambda_w=300.0,
    )
    assert lead < win

