//! Observation tensors for RL — global mini-map + local crop + vector.

use crate::config::UnitType;
use crate::game::{Game, PlayerType};

pub const GLOBAL_H: usize = 64;
pub const GLOBAL_W: usize = 128;
pub const LOCAL_H: usize = 64;
pub const LOCAL_W: usize = 64;
pub const GLOBAL_C: usize = 12;
pub const LOCAL_C: usize = 8;
pub const VECTOR_DIM: usize = 64;

#[derive(Clone, Debug)]
pub struct Observation {
    /// (C, H, W) row-major: c * H * W + y * W + x
    pub global: Vec<f32>,
    pub local: Vec<f32>,
    pub vector: Vec<f32>,
}

impl Observation {
    pub fn zeros() -> Self {
        Self {
            global: vec![0.0; GLOBAL_C * GLOBAL_H * GLOBAL_W],
            local: vec![0.0; LOCAL_C * LOCAL_H * LOCAL_W],
            vector: vec![0.0; VECTOR_DIM],
        }
    }
}

pub fn encode(game: &Game, ego_idx: usize) -> Observation {
    let mut obs = Observation::zeros();
    let ego_sid = game.players[ego_idx].small_id;
    let mw = game.map.width() as f32;
    let mh = game.map.height() as f32;

    // Global tensor from full map downsampled
    for gy in 0..GLOBAL_H {
        for gx in 0..GLOBAL_W {
            let mx = ((gx as f32 + 0.5) / GLOBAL_W as f32 * mw) as u32;
            let my = ((gy as f32 + 0.5) / GLOBAL_H as f32 * mh) as u32;
            let mx = mx.min(game.map.width() - 1);
            let my = my.min(game.map.height() - 1);
            let r = game.map.ref_xy(mx, my);
            let idx = |c: usize| c * GLOBAL_H * GLOBAL_W + gy * GLOBAL_W + gx;

            if game.map.is_water(r) {
                obs.global[idx(4)] = 1.0; // water
                continue;
            }
            let owner = game.map.owner_id(r);
            if owner == 0 {
                obs.global[idx(3)] = 1.0; // TN
            } else if owner == ego_sid {
                obs.global[idx(0)] = 1.0; // self
            } else if game.players[ego_idx]
                .allies
                .iter()
                .any(|id| game.player_by_id(id).map(|i| game.players[i].small_id) == Some(owner))
            {
                obs.global[idx(2)] = 1.0; // ally
            } else {
                obs.global[idx(1)] = 1.0; // enemy
            }
            obs.global[idx(5)] = game.map.magnitude(r) as f32 / 31.0;
            if game.map.has_fallout(r) {
                obs.global[idx(6)] = 1.0;
            }
            if game.map.has_defense_bonus(r) {
                obs.global[idx(7)] = 1.0;
            }
        }
    }

    // Structure channels on global
    for (pi, p) in game.players.iter().enumerate() {
        let chan = if pi == ego_idx { 8 } else { 9 };
        for u in &p.units {
            let gx = ((game.map.x(u.tile) as f32 / mw) * GLOBAL_W as f32) as usize;
            let gy = ((game.map.y(u.tile) as f32 / mh) * GLOBAL_H as f32) as usize;
            let gx = gx.min(GLOBAL_W - 1);
            let gy = gy.min(GLOBAL_H - 1);
            let c = match u.unit_type {
                UnitType::Warship | UnitType::TransportShip => 10,
                UnitType::City | UnitType::Port | UnitType::Factory => chan,
                UnitType::SamLauncher | UnitType::MissileSilo => 11,
                _ => chan,
            };
            obs.global[c * GLOBAL_H * GLOBAL_W + gy * GLOBAL_W + gx] = 1.0;
        }
    }

    // Local crop centered on border centroid
    let (cx, cy) = border_center(game, ego_idx);
    let half = (LOCAL_W as i32) / 2;
    for ly in 0..LOCAL_H {
        for lx in 0..LOCAL_W {
            let mx = cx + lx as i32 - half;
            let my = cy + ly as i32 - half;
            let idx = |c: usize| c * LOCAL_H * LOCAL_W + ly * LOCAL_W + lx;
            if mx < 0 || my < 0 || mx >= game.map.width() as i32 || my >= game.map.height() as i32 {
                obs.local[idx(4)] = 1.0;
                continue;
            }
            let r = game.map.ref_xy(mx as u32, my as u32);
            if game.map.is_water(r) {
                obs.local[idx(4)] = 1.0;
                continue;
            }
            let owner = game.map.owner_id(r);
            if owner == ego_sid {
                obs.local[idx(0)] = 1.0;
            } else if owner == 0 {
                obs.local[idx(3)] = 1.0;
            } else {
                obs.local[idx(1)] = 1.0;
            }
            obs.local[idx(5)] = game.map.magnitude(r) as f32 / 31.0;
            if game.map.has_fallout(r) {
                obs.local[idx(6)] = 1.0;
            }
            if game.players[ego_idx].border_tiles.contains(&r) {
                obs.local[idx(7)] = 1.0;
            }
        }
    }

    // Vector features
    let p = &game.players[ego_idx];
    let city: f64 = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::City)
        .map(|u| u.level as f64)
        .sum();
    let max_t = game
        .config
        .max_troops(p.player_type.into(), p.num_tiles_owned(), city);
    let land = game.map.num_land_tiles().max(1) as f32;
    let enemy = game
        .players
        .iter()
        .enumerate()
        .filter(|(i, pl)| *i != ego_idx && pl.player_type != PlayerType::Bot)
        .max_by(|a, b| a.1.troops.partial_cmp(&b.1.troops).unwrap())
        .map(|(_, pl)| pl);

    let v = &mut obs.vector;
    v[0] = (p.gold as f32).ln_1p();
    v[1] = game.config.gold_addition_rate(p.player_type.into()) as f32;
    v[2] = p.troops as f32 / max_t.max(1.0) as f32;
    v[3] = p.troops as f32 / 100_000.0;
    v[4] = p.num_tiles_owned() as f32 / land;
    v[5] = enemy.map(|e| e.troops as f32 / p.troops.max(1.0) as f32).unwrap_or(0.0);
    v[6] = enemy
        .map(|e| e.num_tiles_owned() as f32 / land)
        .unwrap_or(0.0);
    v[7] = p.units.iter().filter(|u| u.unit_type == UnitType::City).count() as f32;
    v[8] = p.units.iter().filter(|u| u.unit_type == UnitType::Port).count() as f32;
    v[9] = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::SamLauncher)
        .count() as f32;
    v[10] = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::MissileSilo)
        .count() as f32;
    v[11] = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::TransportShip)
        .count() as f32;
    v[12] = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::Warship)
        .count() as f32;
    v[13] = if game.in_spawn_phase { 1.0 } else { 0.0 };
    v[14] = game.ticks as f32 / 10_000.0;
    v[15] = if p.is_traitor { 1.0 } else { 0.0 };

    obs
}

fn border_center(game: &Game, ego: usize) -> (i32, i32) {
    let b = &game.players[ego].border_tiles;
    if b.is_empty() {
        if let Some(&t) = game.players[ego].tiles.iter().next() {
            return (game.map.x(t) as i32, game.map.y(t) as i32);
        }
        return (
            (game.map.width() / 2) as i32,
            (game.map.height() / 2) as i32,
        );
    }
    let (mut sx, mut sy) = (0i64, 0i64);
    for &t in b {
        sx += game.map.x(t) as i64;
        sy += game.map.y(t) as i64;
    }
    ((sx / b.len() as i64) as i32, (sy / b.len() as i64) as i32)
}
