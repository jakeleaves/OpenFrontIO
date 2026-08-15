//! Land attack + boat executions.

use super::{Execution, Game, SmallId};
use crate::config::{AttackDefender, PlayerType as CPlayerType};
use crate::map::TileRef;
use crate::rng::PseudoRandom;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct Attack {
    pub id: String,
    pub troops: f64,
}

#[derive(Clone, Debug)]
pub struct AttackExecution {
    pub id: String,
    pub attacker_idx: usize,
    pub target_small: SmallId,
    pub troops: f64,
    pub source: Option<TileRef>,
    pub active: bool,
    pub to_conquer: BinaryHeap<HeapItem>,
    pub random: PseudoRandom,
    pub tiles_owed: f64,
}

#[derive(Clone, Debug)]
pub struct HeapItem {
    pub priority: i64,
    pub tile: TileRef,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.tile == other.tile
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| self.tile.cmp(&other.tile))
    }
}

impl AttackExecution {
    pub fn new(
        id: String,
        attacker_idx: usize,
        target_small: SmallId,
        troops: f64,
        source: Option<TileRef>,
    ) -> Self {
        Self {
            id,
            attacker_idx,
            target_small,
            troops,
            source,
            active: true,
            to_conquer: BinaryHeap::new(),
            random: PseudoRandom::new(123),
            tiles_owed: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoatExecution {
    pub player_idx: usize,
    pub unit_id: u32,
    pub dst: TileRef,
    pub path: Vec<TileRef>,
    pub path_i: usize,
    pub active: bool,
}

fn terrain_mag_prio(map: &crate::map::GameMap, t: TileRef) -> f64 {
    match map.terrain_type(t) {
        crate::map::TerrainType::Plains => 1.0,
        crate::map::TerrainType::Highland => 1.5,
        crate::map::TerrainType::Mountain => 2.0,
        _ => 1.0,
    }
}

fn enqueue_border_targets(g: &Game, atk: &mut AttackExecution) {
    atk.to_conquer.clear();
    let attacker_sid = g.players[atk.attacker_idx].small_id;
    let seeds: Vec<TileRef> = if let Some(src) = atk.source {
        vec![src]
    } else {
        g.players[atk.attacker_idx]
            .border_tiles
            .iter()
            .copied()
            .collect()
    };
    for tile in seeds {
        let mut nbuf = [0u32; 4];
        let n = g.map.neighbors4(tile, &mut nbuf);
        for i in 0..n {
            let nb = nbuf[i];
            if !g.map.is_land(nb) || g.map.is_impassable(nb) {
                continue;
            }
            let owner = g.map.owner_id(nb);
            let ok = if atk.target_small == 0 {
                owner == 0
            } else {
                owner == atk.target_small
            };
            if !ok || owner == attacker_sid {
                continue;
            }
            let mut owned_neighbors = 0i32;
            let mut nbuf2 = [0u32; 4];
            let n2 = g.map.neighbors4(nb, &mut nbuf2);
            for j in 0..n2 {
                if g.map.owner_id(nbuf2[j]) == attacker_sid {
                    owned_neighbors += 1;
                }
            }
            let rnd = atk.random.next_int(0.0, 7.0) as f64;
            let prio = ((rnd + 10.0)
                * (1.0 - owned_neighbors as f64 * 0.5 + terrain_mag_prio(&g.map, nb) / 2.0)
                + g.ticks as f64) as i64;
            atk.to_conquer.push(HeapItem {
                priority: prio,
                tile: nb,
            });
        }
    }
}

fn rebuild_queue(g: &mut Game, exec_idx: usize) {
    let mut atk = match std::mem::replace(&mut g.executions[exec_idx], Execution::WinCheck) {
        Execution::Attack(a) => a,
        other => {
            g.executions[exec_idx] = other;
            return;
        }
    };
    enqueue_border_targets(g, &mut atk);
    g.executions[exec_idx] = Execution::Attack(atk);
}

pub fn tick_attack(g: &mut Game, exec_idx: usize) {
    let (attacker_idx, target_small) = {
        let Execution::Attack(ref a) = g.executions[exec_idx] else {
            return;
        };
        if !a.active {
            return;
        }
        (a.attacker_idx, a.target_small)
    };

    {
        let empty = matches!(&g.executions[exec_idx], Execution::Attack(a) if a.to_conquer.is_empty());
        if empty {
            rebuild_queue(g, exec_idx);
        }
    }

    let mut tiles_budget = {
        let Execution::Attack(ref mut a) = g.executions[exec_idx] else {
            return;
        };
        let b = 1.0 + a.tiles_owed;
        a.tiles_owed = 0.0;
        b
    };

    while tiles_budget >= 1.0 {
        let tile = {
            let Execution::Attack(ref mut a) = g.executions[exec_idx] else {
                return;
            };
            let mut found = None;
            while let Some(item) = a.to_conquer.pop() {
                let owner = g.map.owner_id(item.tile);
                let ok = if a.target_small == 0 {
                    owner == 0
                } else {
                    owner == a.target_small
                };
                if ok && g.map.is_land(item.tile) && !g.map.is_impassable(item.tile) {
                    found = Some(item.tile);
                    break;
                }
            }
            found
        };

        let Some(tile) = tile else {
            rebuild_queue(g, exec_idx);
            let empty = matches!(&g.executions[exec_idx], Execution::Attack(a) if a.to_conquer.is_empty());
            if empty {
                if let Execution::Attack(a) = &mut g.executions[exec_idx] {
                    let ret = a.troops;
                    a.troops = 0.0;
                    a.active = false;
                    g.players[attacker_idx].troops += ret;
                }
                return;
            }
            continue;
        };

        let attacker_tiles = g.players[attacker_idx].num_tiles_owned();
        let attacker_type: CPlayerType = g.players[attacker_idx].player_type.into();
        let defense_post = if target_small != 0 {
            g.defense_post_near(target_small, tile)
        } else {
            false
        };
        let defender = if target_small == 0 {
            AttackDefender::TerraNullius
        } else if let Some(ti) = g.player_by_small(target_small) {
            AttackDefender::Player {
                troops: g.players[ti].troops,
                tiles: g.players[ti].num_tiles_owned().max(1),
                is_bot: g.players[ti].player_type == super::PlayerType::Bot,
                is_traitor: g.players[ti].is_traitor,
                is_disconnected_same_team: false,
            }
        } else {
            AttackDefender::TerraNullius
        };

        let troops_now = match &g.executions[exec_idx] {
            Execution::Attack(a) => a.troops,
            _ => 0.0,
        };
        if troops_now <= 0.0 {
            if let Execution::Attack(a) = &mut g.executions[exec_idx] {
                a.active = false;
            }
            return;
        }

        let res = g.config.attack_logic(
            &g.map,
            troops_now,
            attacker_type,
            attacker_tiles,
            defender,
            tile,
            defense_post,
        );

        let mut troops_left = 0.0;
        if let Execution::Attack(a) = &mut g.executions[exec_idx] {
            a.troops = (a.troops - res.attacker_troop_loss).max(0.0);
            troops_left = a.troops;
        }
        if target_small != 0 {
            if let Some(ti) = g.player_by_small(target_small) {
                g.players[ti].troops = (g.players[ti].troops - res.defender_troop_loss).max(0.0);
            }
        }

        g.conquer_tile(attacker_idx, tile);

        {
            let Execution::Attack(ref mut a) = g.executions[exec_idx] else {
                return;
            };
            let attacker_sid = g.players[attacker_idx].small_id;
            let mut nbuf = [0u32; 4];
            let n = g.map.neighbors4(tile, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if !g.map.is_land(nb) || g.map.is_impassable(nb) {
                    continue;
                }
                let owner = g.map.owner_id(nb);
                let ok = if target_small == 0 {
                    owner == 0
                } else {
                    owner == target_small
                };
                if !ok || owner == attacker_sid {
                    continue;
                }
                let rnd = a.random.next_int(0.0, 7.0) as f64;
                let prio = ((rnd + 10.0) * (1.0 + terrain_mag_prio(&g.map, nb) / 2.0)
                    + g.ticks as f64) as i64;
                a.to_conquer.push(HeapItem {
                    priority: prio,
                    tile: nb,
                });
            }
        }

        g.recompute_borders(attacker_idx);
        if target_small != 0 {
            if let Some(ti) = g.player_by_small(target_small) {
                g.recompute_borders(ti);
                if g.players[ti].tiles.is_empty() {
                    let loot = match g.players[ti].player_type {
                        super::PlayerType::Human => g.players[ti].gold / 2,
                        _ => g.players[ti].gold,
                    };
                    g.players[attacker_idx].gold += loot;
                    g.players[ti].gold = 0;
                }
            }
        }

        tiles_budget -= res.tiles_per_tick_used.max(0.01);
        if troops_left <= 0.0 {
            if let Execution::Attack(a) = &mut g.executions[exec_idx] {
                a.active = false;
            }
            return;
        }
    }

    if let Execution::Attack(a) = &mut g.executions[exec_idx] {
        a.tiles_owed = tiles_budget;
    }
}

pub fn tick_boat(g: &mut Game, exec_idx: usize) {
    let (pi, uid, dst, path_i, path_len) = {
        let Execution::Boat(ref b) = g.executions[exec_idx] else {
            return;
        };
        if !b.active {
            return;
        }
        (b.player_idx, b.unit_id, b.dst, b.path_i, b.path.len())
    };

    if path_i >= path_len {
        let troops = g.players[pi]
            .units
            .iter()
            .find(|u| u.id == uid)
            .and_then(|u| u.troops)
            .unwrap_or(0.0);
        g.players[pi].units.retain(|u| u.id != uid);
        let target_small = g.map.owner_id(dst);
        if target_small != g.players[pi].small_id {
            g.conquer_tile(pi, dst);
            g.recompute_borders(pi);
        }
        g.start_attack(pi, target_small, troops, Some(dst));
        if let Execution::Boat(b) = &mut g.executions[exec_idx] {
            b.active = false;
        }
        return;
    }

    let next_tile = {
        let Execution::Boat(ref b) = g.executions[exec_idx] else {
            return;
        };
        b.path[path_i]
    };
    if let Some(u) = g.players[pi].units.iter_mut().find(|u| u.id == uid) {
        u.tile = next_tile;
    }
    if let Execution::Boat(b) = &mut g.executions[exec_idx] {
        b.path_i += 1;
    }
}
