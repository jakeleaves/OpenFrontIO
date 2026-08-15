//! Factorized action space + invalid-action masking + intent decode.

use crate::config::UnitType;
use crate::game::{Game, Intent};
use crate::obs::{GLOBAL_H, GLOBAL_W};
use serde::{Deserialize, Serialize};

pub const NUM_ACTION_TYPES: usize = 10;
pub const NUM_TARGET_PLAYERS: usize = 2; // TN, Enemy
pub const COARSE_W: usize = 32;
pub const COARSE_H: usize = 16;
pub const NUM_TROOP_FRACS: usize = 5;
pub const NUM_BUILD_TYPES: usize = 10;

pub const TROOP_FRACS: [f64; NUM_TROOP_FRACS] = [0.1, 0.2, 0.35, 0.5, 0.75];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ActionType {
    Noop = 0,
    Spawn = 1,
    Attack = 2,
    Boat = 3,
    CancelAttack = 4,
    CancelBoat = 5,
    Build = 6,
    Upgrade = 7,
    MoveWarship = 8,
    Delete = 9,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactorizedAction {
    pub action_type: u8,
    pub target_player: u8,
    pub cell_x: u8,
    pub cell_y: u8,
    pub troop_frac: u8,
    pub build_type: u8,
}

impl FactorizedAction {
    pub fn noop() -> Self {
        Self {
            action_type: ActionType::Noop as u8,
            target_player: 0,
            cell_x: 0,
            cell_y: 0,
            troop_frac: 0,
            build_type: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ActionMask {
    pub action_type: [bool; NUM_ACTION_TYPES],
    pub target_player: [bool; NUM_TARGET_PLAYERS],
    pub cell: Vec<bool>, // COARSE_H * COARSE_W
    pub troop_frac: [bool; NUM_TROOP_FRACS],
    pub build_type: [bool; NUM_BUILD_TYPES],
}

impl ActionMask {
    pub fn allow_all() -> Self {
        Self {
            action_type: [true; NUM_ACTION_TYPES],
            target_player: [true; NUM_TARGET_PLAYERS],
            cell: vec![true; COARSE_H * COARSE_W],
            troop_frac: [true; NUM_TROOP_FRACS],
            build_type: [true; NUM_BUILD_TYPES],
        }
    }
}

const BUILD_TYPES: [UnitType; NUM_BUILD_TYPES] = [
    UnitType::City,
    UnitType::Port,
    UnitType::DefensePost,
    UnitType::SamLauncher,
    UnitType::MissileSilo,
    UnitType::Factory,
    UnitType::Warship,
    UnitType::AtomBomb,
    UnitType::HydrogenBomb,
    UnitType::Mirv,
];

pub fn legal_mask(game: &Game, ego: usize) -> ActionMask {
    let mut m = ActionMask {
        action_type: [false; NUM_ACTION_TYPES],
        target_player: [true; NUM_TARGET_PLAYERS],
        cell: vec![false; COARSE_H * COARSE_W],
        troop_frac: [true; NUM_TROOP_FRACS],
        build_type: [false; NUM_BUILD_TYPES],
    };
    m.action_type[ActionType::Noop as usize] = true;

    let p = &game.players[ego];
    if game.in_spawn_phase && p.tiles.is_empty() {
        m.action_type[ActionType::Spawn as usize] = true;
        for y in 0..COARSE_H {
            for x in 0..COARSE_W {
                m.cell[y * COARSE_W + x] = true;
            }
        }
        return m;
    }
    if p.tiles.is_empty() {
        return m;
    }

    m.action_type[ActionType::Attack as usize] = true;
    let boats = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::TransportShip)
        .count();
    if boats < game.config.boat_max_number() as usize {
        m.action_type[ActionType::Boat as usize] = true;
    }
    m.action_type[ActionType::Build as usize] = true;
    if p.units.iter().any(|u| u.unit_type == UnitType::Warship) {
        m.action_type[ActionType::MoveWarship as usize] = true;
    }
    if !p.units.is_empty() {
        m.action_type[ActionType::Delete as usize] = true;
        m.action_type[ActionType::Upgrade as usize] = true;
    }
    m.action_type[ActionType::CancelAttack as usize] = true;
    m.action_type[ActionType::CancelBoat as usize] = boats > 0;

    for y in 0..COARSE_H {
        for x in 0..COARSE_W {
            m.cell[y * COARSE_W + x] = true;
        }
    }
    for (i, ut) in BUILD_TYPES.iter().enumerate() {
        let owned = p.units.iter().filter(|u| u.unit_type == *ut).count() as u32;
        let cost = game.config.unit_cost(*ut, owned);
        m.build_type[i] = p.gold >= cost;
    }
    let _ = (GLOBAL_H, GLOBAL_W);
    m
}

pub fn decode_intent(game: &Game, ego: usize, a: &FactorizedAction) -> Intent {
    let tile = coarse_to_tile(game, a.cell_x as usize, a.cell_y as usize);
    let frac = TROOP_FRACS[a.troop_frac as usize % NUM_TROOP_FRACS];
    let troops = game.players[ego].troops * frac;

    match a.action_type {
        x if x == ActionType::Noop as u8 => Intent::Noop,
        x if x == ActionType::Spawn as u8 => Intent::Spawn { tile },
        x if x == ActionType::Attack as u8 => {
            let target_id = if a.target_player == 0 {
                None
            } else {
                game.players
                    .iter()
                    .enumerate()
                    .find(|(i, p)| {
                        *i != ego
                            && p.player_type != crate::game::PlayerType::Bot
                            && !p.tiles.is_empty()
                    })
                    .map(|(_, p)| p.id.clone())
            };
            Intent::Attack {
                target_id,
                troops: Some(troops),
            }
        }
        x if x == ActionType::Boat as u8 => Intent::Boat { troops, dst: tile },
        x if x == ActionType::Build as u8 => {
            let ut = BUILD_TYPES[a.build_type as usize % NUM_BUILD_TYPES];
            Intent::BuildUnit {
                unit: ut.as_str().to_string(),
                tile,
                amount: None,
            }
        }
        x if x == ActionType::MoveWarship as u8 => {
            let ids: Vec<u32> = game.players[ego]
                .units
                .iter()
                .filter(|u| u.unit_type == UnitType::Warship)
                .map(|u| u.id)
                .collect();
            if ids.is_empty() {
                Intent::Noop
            } else {
                Intent::MoveWarship {
                    unit_ids: ids,
                    tile,
                }
            }
        }
        x if x == ActionType::Delete as u8 => {
            if let Some(u) = game.players[ego].units.first() {
                Intent::DeleteUnit { unit_id: u.id }
            } else {
                Intent::Noop
            }
        }
        x if x == ActionType::CancelBoat as u8 => {
            if let Some(u) = game.players[ego]
                .units
                .iter()
                .find(|u| u.unit_type == UnitType::TransportShip)
            {
                Intent::CancelBoat { unit_id: u.id }
            } else {
                Intent::Noop
            }
        }
        x if x == ActionType::Upgrade as u8 => {
            if let Some(u) = game.players[ego].units.first() {
                Intent::UpgradeStructure {
                    unit: u.unit_type.as_str().to_string(),
                    unit_id: u.id,
                    amount: Some(1),
                }
            } else {
                Intent::Noop
            }
        }
        _ => Intent::Noop,
    }
}

fn coarse_to_tile(game: &Game, cx: usize, cy: usize) -> u32 {
    let cx = cx.min(COARSE_W - 1);
    let cy = cy.min(COARSE_H - 1);
    let x = ((cx as f32 + 0.5) / COARSE_W as f32 * game.map.width() as f32) as u32;
    let y = ((cy as f32 + 0.5) / COARSE_H as f32 * game.map.height() as f32) as u32;
    game.map
        .ref_xy(x.min(game.map.width() - 1), y.min(game.map.height() - 1))
}

/// 1v1 reward shaping (plan §4).
pub fn shaped_reward(
    prev_tiles: u32,
    prev_troops_diff: f64,
    prev_gold: i64,
    game: &Game,
    ego: usize,
    boat_sunk: bool,
) -> f64 {
    let p = &game.players[ego];
    let enemy_troops = game
        .players
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ego)
        .map(|(_, pl)| pl.troops)
        .sum::<f64>();
    let city: f64 = p
        .units
        .iter()
        .filter(|u| u.unit_type == UnitType::City)
        .map(|u| u.level as f64)
        .sum();
    let max_t = game
        .config
        .max_troops(p.player_type.into(), p.num_tiles_owned(), city)
        .max(1.0);

    let mut r = 0.0;
    let lambda_n = 1.0;
    let lambda_t = 0.3;
    let lambda_g = 0.05;
    let lambda_s = 0.5;
    let lambda_w = 10.0;

    r += lambda_n * (p.num_tiles_owned() as f64 - prev_tiles as f64);
    r += lambda_t * ((p.troops - enemy_troops) - prev_troops_diff) / max_t;
    r += lambda_g * (p.gold - prev_gold) as f64 / (1.0 + p.gold as f64);
    if boat_sunk {
        r -= lambda_s;
    }
    if let Some(w) = game.winner {
        if w == p.small_id {
            r += lambda_w;
        } else {
            r -= lambda_w;
        }
    }
    r
}
