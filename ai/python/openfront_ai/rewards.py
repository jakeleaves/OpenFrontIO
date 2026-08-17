"""Reward helpers mirroring ai/ts/env shapedReward."""


def shaped_reward_torch(
    win: bool | None,
    d_land: float,
    d_troop_diff_norm: float,
    d_gold_norm: float,
    boat_sunk: bool,
    lambda_w: float = 300.0,
    lambda_n: float = 20.0,
    lambda_nation_tiles: float = 0.1,
    lambda_struct: float = 2.0,
    lambda_city: float = 12.0,
    lambda_t: float = 0.5,
    lambda_g: float = 0.05,
    lambda_s: float = 0.5,
    d_nation_tiles: float = 0.0,
    d_struct: float = 0.0,
    d_cities: float = 0.0,
    troop_ratio: float | None = None,
    lambda_reserve: float = 8.0,
    reserve_ratio: float = 0.35,
    growth_efficiency: float | None = None,
    previous_growth_efficiency: float | None = None,
    lambda_growth: float = 1.0,
    boat_launched: bool = False,
    lambda_boat_launch: float = 1.0,
    lambda_placement: float = 20.0,
    d_placement: float = 0.0,
    lambda_tile_lead: float = 20.0,
    d_tile_lead: float = 0.0,
) -> float:
    # Only reward city-count increases (never decreases) to avoid farming losses.
    city_gain = max(0.0, d_cities)
    r = (
        lambda_n * d_land
        + lambda_nation_tiles * d_nation_tiles
        + lambda_struct * d_struct
        + lambda_city * city_gain
        + lambda_t * d_troop_diff_norm
        + lambda_g * d_gold_norm
        + lambda_placement * d_placement
        + lambda_tile_lead * d_tile_lead
    )
    if boat_sunk:
        r -= lambda_s
    if boat_launched:
        r += lambda_boat_launch
    if troop_ratio is not None and troop_ratio < reserve_ratio:
        r -= lambda_reserve * (reserve_ratio - troop_ratio)
    if growth_efficiency is not None and previous_growth_efficiency is not None:
        r += lambda_growth * (growth_efficiency - previous_growth_efficiency)
    if win is True:
        r += lambda_w
    elif win is False:
        r -= lambda_w
    return r


# FFA stub (not used in v1 Nations 1v1)
def ffa_utility(placement: float, is_first: bool, p_tip: float, kappa: float = 0.5) -> float:
    return placement - kappa * (1.0 if is_first else 0.0) * (1.0 - p_tip)
