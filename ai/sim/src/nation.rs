//! Nation AI — port of NationExecution + AiAttackBehavior (simplified Impossible).

use crate::config::UnitType;
use crate::game::{Game, Intent, PlayerType};
use crate::js_math::simple_hash;
use crate::rng::PseudoRandom;

pub fn tick_nation(g: &mut Game, pi: usize) {
    if g.players[pi].tiles.is_empty() {
        return;
    }
    let seed = simple_hash(&g.players[pi].id)
        .wrapping_add(simple_hash(&g.game_id))
        .wrapping_add(g.ticks as i32);
    let mut rng = PseudoRandom::new(seed);

    // Build city / defense if rich
    maybe_build(g, pi, &mut rng);

    // Attack logic
    maybe_attack(g, pi, &mut rng);
}

fn maybe_build(g: &mut Game, pi: usize, rng: &mut PseudoRandom) {
    let gold = g.players[pi].gold;
    let cities = g.players[pi]
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::City)
        .count();
    if gold > 200_000 && cities < 5 {
        if let Some(&tile) = g.players[pi]
            .tiles
            .iter()
            .nth(rng.next_int(0.0, g.players[pi].tiles.len().max(1) as f64) as usize)
        {
            g.try_build(pi, UnitType::City, tile);
        }
    }
    let ports = g.players[pi]
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::Port)
        .count();
    if gold > 200_000 && ports < 3 {
        // Find coastal tile
        let coastal: Vec<_> = g.players[pi]
            .tiles
            .iter()
            .copied()
            .filter(|&t| {
                let mut w = false;
                g.map.for_each_neighbor(t, |nb| {
                    if g.map.is_water(nb) {
                        w = true;
                    }
                });
                w
            })
            .collect();
        if let Some(&tile) = coastal.first() {
            g.try_build(pi, UnitType::Port, tile);
        }
    }
    if gold > 1_000_000 {
        if let Some(&tile) = g.players[pi].tiles.iter().next() {
            g.try_build(pi, UnitType::DefensePost, tile);
        }
    }
}

fn maybe_attack(g: &mut Game, pi: usize, rng: &mut PseudoRandom) {
    let troops = g.players[pi].troops;
    let trigger = 0.5 + rng.next_float(0.0, 0.1);
    let max = {
        let pt = g.players[pi].player_type.into();
        let city: f64 = g.players[pi]
            .units
            .iter()
            .filter(|u| u.unit_type == UnitType::City)
            .map(|u| u.level as f64)
            .sum();
        g.config
            .max_troops(pt, g.players[pi].num_tiles_owned(), city)
    };
    if troops / max < trigger {
        return;
    }
    let reserve = 0.3 + rng.next_float(0.0, 0.1);
    let attack_troops = troops * (1.0 - reserve);

    // Prefer weakest neighbor or terra nullius
    let mut best_target: Option<u16> = Some(0); // TN
    let mut best_score = f64::MAX;

    // Scan borders for enemy neighbors
    let borders: Vec<_> = g.players[pi].border_tiles.iter().copied().collect();
    let mut nbuf = [0u32; 4];
    for t in borders {
        let n = g.map.neighbors4(t, &mut nbuf);
        for i in 0..n {
            let owner = g.map.owner_id(nbuf[i]);
            if owner == 0 {
                best_target = Some(0);
                best_score = 0.0;
                break;
            }
            if owner == g.players[pi].small_id {
                continue;
            }
            if let Some(oi) = g.player_by_small(owner) {
                if g.players[pi].allies.contains(&g.players[oi].id) {
                    continue;
                }
                let score = g.players[oi].troops;
                if score < best_score {
                    best_score = score;
                    best_target = Some(owner);
                }
            }
        }
        if best_score == 0.0 {
            break;
        }
    }

    // Impossible: also consider boat attacks occasionally
    if rng.chance(8) {
        // Random distant land tile
        for _ in 0..50 {
            let x = rng.next_int(0.0, g.map.width() as f64) as u32;
            let y = rng.next_int(0.0, g.map.height() as f64) as u32;
            let r = g.map.ref_xy(x, y);
            if g.map.is_land(r) && !g.map.is_impassable(r) {
                let owner = g.map.owner_id(r);
                if owner != g.players[pi].small_id {
                    g.start_boat(pi, r, attack_troops * 0.2);
                    return;
                }
            }
        }
    }

    if let Some(tgt) = best_target {
        g.start_attack(pi, tgt, attack_troops, None);
    }
}

/// Record Nation decisions as intents for behavior cloning.
pub fn nation_intent_trace(g: &Game, pi: usize) -> Vec<Intent> {
    // Placeholder — training harness records intents applied by tick_nation via hooks.
    let _ = (g, pi);
    vec![Intent::Noop]
}

pub fn is_nation(g: &Game, pi: usize) -> bool {
    g.players[pi].player_type == PlayerType::Nation
}
