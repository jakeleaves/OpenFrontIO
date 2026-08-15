"""Reward helpers mirroring ai/sim action::shaped_reward."""


def shaped_reward_torch(
    win: bool | None,
    d_land: float,
    d_troop_diff_norm: float,
    d_gold_norm: float,
    boat_sunk: bool,
    lambda_w: float = 10.0,
    lambda_n: float = 1.0,
    lambda_t: float = 0.3,
    lambda_g: float = 0.05,
    lambda_s: float = 0.5,
) -> float:
    r = lambda_n * d_land + lambda_t * d_troop_diff_norm + lambda_g * d_gold_norm
    if boat_sunk:
        r -= lambda_s
    if win is True:
        r += lambda_w
    elif win is False:
        r -= lambda_w
    return r


# FFA stub (not used in v1 Nations 1v1)
def ffa_utility(placement: float, is_first: bool, p_tip: float, kappa: float = 0.5) -> float:
    return placement - kappa * (1.0 if is_first else 0.0) * (1.0 - p_tip)
