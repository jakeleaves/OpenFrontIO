//! Core game state and tick loop — port of GameImpl / PlayerImpl / executions.

mod attack;
mod player;
mod unit;

pub use attack::Attack;
pub use player::Player;
pub use unit::Unit;

use crate::config::{Config, PlayerType as CPlayerType, UnitType};
pub use crate::config::{Difficulty, GameMode};
use crate::js_math::simple_hash;
use crate::map::{GameMap, TileRef};
use crate::rng::PseudoRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub type PlayerId = String;
pub type SmallId = u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    Bot,
    Nation,
}

impl From<PlayerType> for CPlayerType {
    fn from(p: PlayerType) -> Self {
        match p {
            PlayerType::Human => CPlayerType::Human,
            PlayerType::Bot => CPlayerType::Bot,
            PlayerType::Nation => CPlayerType::Nation,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Intent {
    Spawn { tile: u32 },
    Attack {
        target_id: Option<String>,
        troops: Option<f64>,
    },
    Boat { troops: f64, dst: u32 },
    CancelAttack { attack_id: String },
    CancelBoat { unit_id: u32 },
    BuildUnit {
        unit: String,
        tile: u32,
        amount: Option<u32>,
    },
    UpgradeStructure {
        unit: String,
        unit_id: u32,
        amount: Option<u32>,
    },
    MoveWarship { unit_ids: Vec<u32>, tile: u32 },
    DeleteUnit { unit_id: u32 },
    AllianceRequest { recipient: String },
    AllianceReject { requestor: String },
    AllianceExtension { recipient: String },
    BreakAlliance { recipient: String },
    DonateGold {
        recipient: String,
        gold: Option<i64>,
    },
    DonateTroops {
        recipient: String,
        troops: Option<f64>,
    },
    Embargo {
        target_id: String,
        action: String,
    },
    Noop,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Turn {
    pub turn_number: u32,
    pub intents: Vec<StampedIntent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StampedIntent {
    pub client_id: String,
    #[serde(flatten)]
    pub intent: Intent,
}

#[derive(Clone, Debug)]
pub struct PlayerInfo {
    pub name: String,
    pub player_type: PlayerType,
    pub client_id: Option<String>,
    pub id: String,
}

pub struct GameBuilder {
    pub config: Config,
    pub map: GameMap,
    pub mini_map: GameMap,
    pub humans: Vec<PlayerInfo>,
    pub nations: Vec<PlayerInfo>,
    pub game_id: String,
    pub end_spawn_immediately: bool,
}

impl GameBuilder {
    pub fn from_map_dir(dir: &Path, compact: bool) -> anyhow::Result<Self> {
        let (map, mini_map, _man) = GameMap::load_pair(dir, compact)?;
        Ok(Self {
            config: Config::default(),
            map,
            mini_map,
            humans: vec![],
            nations: vec![],
            game_id: "test-game".into(),
            end_spawn_immediately: true,
        })
    }

    pub fn with_human(mut self, name: &str, id: &str) -> Self {
        self.humans.push(PlayerInfo {
            name: name.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
        });
        self
    }

    pub fn with_nation(mut self, name: &str, id: &str) -> Self {
        self.nations.push(PlayerInfo {
            name: name.into(),
            player_type: PlayerType::Nation,
            client_id: None,
            id: id.into(),
        });
        self
    }

    pub fn difficulty(mut self, d: Difficulty) -> Self {
        self.config.difficulty = d;
        self
    }

    pub fn build(self) -> Game {
        Game::new(self)
    }
}

pub(crate) enum Execution {
    PlayerEco { player_idx: usize },
    Attack(attack::AttackExecution),
    Nation { player_idx: usize, next_tick: u32 },
    WinCheck,
    Boat(attack::BoatExecution),
}

pub struct Game {
    pub config: Config,
    pub map: GameMap,
    pub mini_map: GameMap,
    pub players: Vec<Player>,
    pub ticks: u32,
    pub in_spawn_phase: bool,
    pub game_id: String,
    pub winner: Option<SmallId>,
    next_unit_id: u32,
    next_attack_id: u64,
    executions: Vec<Execution>,
    /// client_id -> player index
    client_to_player: HashMap<String, usize>,
    id_to_player: HashMap<String, usize>,
    water_components: Option<path_stub::WaterComponents>,
}

// Avoid circular import in path — thin stub used until path module fills in.
mod path_stub {
    use crate::map::{GameMap, TileRef};
    use std::collections::VecDeque;

    #[derive(Clone, Debug)]
    pub struct WaterComponents {
        pub component: Vec<i32>,
    }

    impl WaterComponents {
        pub fn build(map: &GameMap) -> Self {
            let n = map.len();
            let mut component = vec![-1i32; n];
            let mut cid = 0i32;
            let mut q = VecDeque::new();
            for r in 0..n as u32 {
                if !map.is_water(r) || component[r as usize] >= 0 {
                    continue;
                }
                component[r as usize] = cid;
                q.push_back(r);
                while let Some(cur) = q.pop_front() {
                    map.for_each_neighbor(cur, |nb| {
                        if map.is_water(nb) && component[nb as usize] < 0 {
                            component[nb as usize] = cid;
                            q.push_back(nb);
                        }
                    });
                }
                cid += 1;
            }
            Self { component }
        }

        pub fn same(&self, a: TileRef, b: TileRef) -> bool {
            let ca = self.component[a as usize];
            let cb = self.component[b as usize];
            ca >= 0 && ca == cb
        }
    }
}

impl Game {
    pub fn new(b: GameBuilder) -> Self {
        let mut players = Vec::new();
        let mut client_to_player = HashMap::new();
        let mut id_to_player = HashMap::new();
        let mut small: SmallId = 1; // 0 = terra nullius

        for info in b.humans.into_iter().chain(b.nations.into_iter()) {
            let idx = players.len();
            if let Some(ref c) = info.client_id {
                client_to_player.insert(c.clone(), idx);
            }
            id_to_player.insert(info.id.clone(), idx);
            let troops = b.config.start_manpower(info.player_type.into());
            players.push(Player::new(info, small, troops));
            small += 1;
        }

        let mut g = Self {
            config: b.config,
            map: b.map,
            mini_map: b.mini_map,
            players,
            ticks: 0,
            in_spawn_phase: !b.end_spawn_immediately,
            game_id: b.game_id,
            winner: None,
            next_unit_id: 1,
            next_attack_id: 1,
            executions: Vec::new(),
            client_to_player,
            id_to_player,
            water_components: None,
        };

        // Persistent executions
        for i in 0..g.players.len() {
            g.executions.push(Execution::PlayerEco { player_idx: i });
            if g.players[i].player_type == PlayerType::Nation {
                let seed = simple_hash(&g.players[i].id).wrapping_add(simple_hash(&g.game_id));
                let mut rng = PseudoRandom::new(seed);
                let interval = match g.config.difficulty {
                    Difficulty::Easy => 100,
                    Difficulty::Medium => 70,
                    Difficulty::Hard => 45,
                    Difficulty::Impossible => 30,
                };
                let next = rng.next_int(0.0, interval as f64) as u32;
                g.executions.push(Execution::Nation {
                    player_idx: i,
                    next_tick: next,
                });
            }
        }
        g.executions.push(Execution::WinCheck);
        g.water_components = Some(path_stub::WaterComponents::build(&g.map));
        g
    }

    pub fn end_spawn_phase(&mut self) {
        self.in_spawn_phase = false;
    }

    pub fn player_by_id(&self, id: &str) -> Option<usize> {
        self.id_to_player.get(id).copied()
    }

    pub fn player_by_client(&self, client: &str) -> Option<usize> {
        self.client_to_player.get(client).copied()
    }

    pub fn player_by_small(&self, sid: SmallId) -> Option<usize> {
        self.players.iter().position(|p| p.small_id == sid)
    }

    /// Spawn a player on a tile (assigns a disk of territory).
    pub fn spawn_player(&mut self, player_idx: usize, tile: TileRef) {
        if !self.map.is_land(tile) || self.map.is_impassable(tile) || self.map.has_owner(tile) {
            return;
        }
        let sid = self.players[player_idx].small_id;
        // Initial spawn radius similar to game (~few tiles)
        let cx = self.map.x(tile) as i32;
        let cy = self.map.y(tile) as i32;
        let radius = 4i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 {
                    continue;
                }
                let (x, y) = (x as u32, y as u32);
                if !self.map.is_valid_coord(x, y) {
                    continue;
                }
                let r = self.map.ref_xy(x, y);
                if self.map.is_land(r) && !self.map.is_impassable(r) && !self.map.has_owner(r) {
                    self.conquer_tile(player_idx, r);
                }
            }
        }
        self.players[player_idx].spawn_tile = Some(tile);
        let _ = sid;
        self.recompute_borders(player_idx);
    }

    pub fn conquer_tile(&mut self, player_idx: usize, tile: TileRef) {
        let prev = self.map.owner_id(tile);
        if prev != 0 {
            if let Some(pi) = self.player_by_small(prev) {
                self.players[pi].tiles.remove(&tile);
                self.players[pi].border_tiles.remove(&tile);
            }
        }
        let sid = self.players[player_idx].small_id;
        self.map.set_owner_id(tile, sid);
        self.players[player_idx].tiles.insert(tile);
        self.players[player_idx].last_tile_change = self.ticks;
    }

    pub fn recompute_borders(&mut self, player_idx: usize) {
        let tiles: Vec<TileRef> = self.players[player_idx].tiles.iter().copied().collect();
        let sid = self.players[player_idx].small_id;
        let mut borders = HashSet::new();
        let mut nbuf = [0u32; 4];
        for &t in &tiles {
            let n = self.map.neighbors4(t, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if self.map.is_land(nb)
                    && !self.map.is_impassable(nb)
                    && self.map.owner_id(nb) != sid
                {
                    borders.insert(t);
                    break;
                }
            }
        }
        self.players[player_idx].border_tiles = borders;
    }

    pub fn apply_intent(&mut self, client_id: &str, intent: &Intent) {
        let Some(pi) = self.player_by_client(client_id).or_else(|| {
            // Nations / bots stamp with their player id
            self.player_by_id(client_id)
        }) else {
            return;
        };
        match intent {
            Intent::Noop => {}
            Intent::Spawn { tile } => {
                if self.players[pi].tiles.is_empty() {
                    self.spawn_player(pi, *tile);
                }
            }
            Intent::Attack { target_id, troops } => {
                let target_small = match target_id {
                    None => 0u16, // terra nullius
                    Some(id) if id.is_empty() || id == "0" => 0,
                    Some(id) => self
                        .player_by_id(id)
                        .map(|i| self.players[i].small_id)
                        .unwrap_or(0),
                };
                let troops = troops.unwrap_or_else(|| {
                    self.config
                        .attack_amount(self.players[pi].player_type.into(), self.players[pi].troops)
                });
                self.start_attack(pi, target_small, troops, None);
            }
            Intent::Boat { troops, dst } => {
                self.start_boat(pi, *dst, *troops);
            }
            Intent::CancelAttack { attack_id } => {
                self.cancel_attack(pi, attack_id);
            }
            Intent::BuildUnit { unit, tile, .. } => {
                if let Some(ut) = parse_unit(unit) {
                    self.try_build(pi, ut, *tile);
                }
            }
            Intent::DeleteUnit { unit_id } => {
                self.delete_unit(pi, *unit_id);
            }
            Intent::AllianceRequest { recipient } => {
                if let Some(ri) = self.player_by_id(recipient) {
                    let rid = self.players[ri].id.clone();
                    self.players[pi].pending_alliance_out.insert(rid);
                }
            }
            Intent::BreakAlliance { recipient } => {
                if let Some(ri) = self.player_by_id(recipient) {
                    let rid = self.players[ri].id.clone();
                    let pid = self.players[pi].id.clone();
                    self.players[pi].allies.remove(&rid);
                    self.players[ri].allies.remove(&pid);
                    self.players[pi].is_traitor = true;
                    self.players[pi].traitor_until = self.ticks + 300;
                }
            }
            Intent::DonateGold { recipient, gold } => {
                if let Some(ri) = self.player_by_id(recipient) {
                    let amt = gold.unwrap_or(self.players[pi].gold / 3);
                    let amt = amt.min(self.players[pi].gold);
                    self.players[pi].gold -= amt;
                    self.players[ri].gold += amt;
                }
            }
            Intent::DonateTroops { recipient, troops } => {
                if let Some(ri) = self.player_by_id(recipient) {
                    let amt = troops.unwrap_or(self.players[pi].troops / 3.0);
                    let amt = amt.min(self.players[pi].troops);
                    self.players[pi].troops -= amt;
                    self.players[ri].troops += amt;
                }
            }
            Intent::Embargo { target_id, action } => {
                if let Some(ti) = self.player_by_id(target_id) {
                    let tid = self.players[ti].id.clone();
                    if action == "start" {
                        self.players[pi].embargoes.insert(tid);
                    } else {
                        self.players[pi].embargoes.remove(&tid);
                    }
                }
            }
            Intent::MoveWarship { unit_ids, tile } => {
                for &uid in unit_ids {
                    if let Some(u) = self.players[pi].units.iter_mut().find(|u| u.id == uid) {
                        if u.unit_type == UnitType::Warship {
                            u.target_tile = Some(*tile);
                        }
                    }
                }
            }
            Intent::UpgradeStructure { unit_id, .. } => {
                let (ut, level) = match self.players[pi]
                    .units
                    .iter()
                    .find(|u| u.id == *unit_id)
                {
                    Some(u) => (u.unit_type, u.level),
                    None => return,
                };
                let cost = self.config.unit_cost(ut, level);
                if self.players[pi].gold >= cost {
                    self.players[pi].gold -= cost;
                    if let Some(u) = self.players[pi]
                        .units
                        .iter_mut()
                        .find(|u| u.id == *unit_id)
                    {
                        u.level += 1;
                    }
                }
            }
            Intent::CancelBoat { unit_id } => {
                // Retreat transport: 25% malus
                if let Some(pos) = self.players[pi]
                    .units
                    .iter()
                    .position(|u| u.id == *unit_id && u.unit_type == UnitType::TransportShip)
                {
                    let u = self.players[pi].units.remove(pos);
                    let returned = u.troops.unwrap_or(0.0) * 0.75;
                    self.players[pi].troops += returned;
                }
            }
            Intent::AllianceReject { .. } | Intent::AllianceExtension { .. } => {
                // Handled at higher level for Nation; accept extension by default
            }
        }
    }

    pub fn start_attack(
        &mut self,
        attacker: usize,
        target_small: SmallId,
        troops: f64,
        source: Option<TileRef>,
    ) {
        let troops = troops.min(self.players[attacker].troops).max(0.0);
        if troops <= 0.0 {
            return;
        }
        if target_small != 0 {
            if let Some(ti) = self.player_by_small(target_small) {
                if self.players[attacker].allies.contains(&self.players[ti].id) {
                    return;
                }
            }
        }
        self.players[attacker].troops -= troops;
        let id = format!("atk-{}", self.next_attack_id);
        self.next_attack_id += 1;
        let exec = attack::AttackExecution::new(
            id,
            attacker,
            target_small,
            troops,
            source,
        );
        self.executions.push(Execution::Attack(exec));
    }

    pub fn start_boat(&mut self, attacker: usize, dst: TileRef, troops: f64) {
        let boats = self.players[attacker]
            .units
            .iter()
            .filter(|u| u.unit_type == UnitType::TransportShip)
            .count() as u32;
        if boats >= self.config.boat_max_number() {
            return;
        }
        let troops = troops.min(self.players[attacker].troops).max(0.0);
        if troops <= 0.0 || !self.map.is_land(dst) {
            return;
        }
        // Find shore deployment
        let Some(src) = self.best_shore_spawn(attacker, dst) else {
            return;
        };
        self.players[attacker].troops -= troops;
        let uid = self.next_unit_id;
        self.next_unit_id += 1;
        let mut unit = Unit::new(uid, UnitType::TransportShip, src, attacker);
        unit.troops = Some(troops);
        unit.target_tile = Some(dst);
        self.players[attacker].units.push(unit);
        self.executions.push(Execution::Boat(attack::BoatExecution {
            player_idx: attacker,
            unit_id: uid,
            dst,
            path_i: 0,
            path: self.water_path(src, dst).unwrap_or_default(),
            active: true,
        }));
    }

    fn best_shore_spawn(&self, player_idx: usize, dst: TileRef) -> Option<TileRef> {
        let mut best = None;
        let mut best_d = u32::MAX;
        for &t in &self.players[player_idx].tiles {
            if !self.map.is_shoreline(t) && !self.is_coastal_land(t) {
                continue;
            }
            // Prefer land tiles adjacent to water
            let mut water_nb = false;
            self.map.for_each_neighbor(t, |nb| {
                if self.map.is_water(nb) {
                    water_nb = true;
                }
            });
            if !water_nb {
                continue;
            }
            let d = self.map.manhattan_dist(t, dst);
            if d < best_d {
                best_d = d;
                best = Some(t);
            }
        }
        best
    }

    fn is_coastal_land(&self, t: TileRef) -> bool {
        if !self.map.is_land(t) {
            return false;
        }
        let mut w = false;
        self.map.for_each_neighbor(t, |nb| {
            if self.map.is_water(nb) {
                w = true;
            }
        });
        w
    }

    fn water_path(&self, from_land: TileRef, to_land: TileRef) -> Option<Vec<TileRef>> {
        // Step onto adjacent water, BFS to water near target, then to target.
        let start_water = self.adjacent_water(from_land)?;
        let end_water = self.adjacent_water(to_land)?;
        if let Some(ref wc) = self.water_components {
            if !wc.same(start_water, end_water) {
                return None;
            }
        }
        crate::path::bfs_water(&self.map, start_water, end_water)
    }

    fn adjacent_water(&self, land: TileRef) -> Option<TileRef> {
        let mut found = None;
        self.map.for_each_neighbor(land, |nb| {
            if found.is_none() && self.map.is_water(nb) {
                found = Some(nb);
            }
        });
        found
    }

    pub fn try_build(&mut self, player_idx: usize, ut: UnitType, tile: TileRef) {
        if !self.players[player_idx].tiles.contains(&tile) {
            return;
        }
        let owned = self.players[player_idx]
            .units
            .iter()
            .filter(|u| u.unit_type == ut)
            .count() as u32;
        let cost = self.config.unit_cost(ut, owned);
        if self.players[player_idx].gold < cost {
            return;
        }
        // Structure spacing
        if matches!(
            ut,
            UnitType::City
                | UnitType::Port
                | UnitType::Factory
                | UnitType::DefensePost
                | UnitType::SamLauncher
                | UnitType::MissileSilo
        ) {
            for u in &self.players[player_idx].units {
                if self.map.manhattan_dist(u.tile, tile) < 15 {
                    return;
                }
            }
        }
        if ut == UnitType::Port && !self.is_coastal_land(tile) {
            return;
        }
        self.players[player_idx].gold -= cost;
        let uid = self.next_unit_id;
        self.next_unit_id += 1;
        let mut unit = Unit::new(uid, ut, tile, player_idx);
        if matches!(
            ut,
            UnitType::City | UnitType::Port | UnitType::Factory | UnitType::SamLauncher | UnitType::MissileSilo
        ) {
            unit.level = 1;
        }
        if ut == UnitType::DefensePost {
            self.map.set_defense_bonus(tile, true);
        }
        // Nukes fire immediately toward map center of enemy — simplified: mark as projectile
        if matches!(ut, UnitType::AtomBomb | UnitType::HydrogenBomb | UnitType::Mirv) {
            unit.target_tile = self.players.iter().find(|p| {
                p.small_id != self.players[player_idx].small_id && !p.tiles.is_empty()
            }).and_then(|p| p.tiles.iter().next().copied());
        }
        self.players[player_idx].units.push(unit);
    }

    pub fn delete_unit(&mut self, player_idx: usize, unit_id: u32) {
        if let Some(pos) = self.players[player_idx]
            .units
            .iter()
            .position(|u| u.id == unit_id)
        {
            let u = self.players[player_idx].units.remove(pos);
            if u.unit_type == UnitType::DefensePost {
                self.map.set_defense_bonus(u.tile, false);
            }
        }
    }

    fn cancel_attack(&mut self, player_idx: usize, attack_id: &str) {
        for ex in &mut self.executions {
            if let Execution::Attack(a) = ex {
                if a.id == attack_id && a.attacker_idx == player_idx && a.active {
                    let ret = a.troops * 0.75; // malusForRetreat = 25
                    self.players[player_idx].troops += ret;
                    a.active = false;
                }
            }
        }
    }

    pub fn execute_next_tick(&mut self) {
        // Tick active executions
        let mut i = 0;
        while i < self.executions.len() {
            let run = !self.in_spawn_phase
                || matches!(self.executions[i], Execution::WinCheck);
            if !run {
                i += 1;
                continue;
            }
            match &self.executions[i] {
                Execution::PlayerEco { player_idx } => {
                    let pi = *player_idx;
                    self.tick_economy(pi);
                }
                Execution::Attack(_) => {
                    attack::tick_attack(self, i);
                }
                Execution::Boat(_) => {
                    attack::tick_boat(self, i);
                }
                Execution::Nation { player_idx, next_tick } => {
                    let pi = *player_idx;
                    let next = *next_tick;
                    if self.ticks >= next && !self.in_spawn_phase {
                        crate::nation::tick_nation(self, pi);
                        let interval = match self.config.difficulty {
                            Difficulty::Easy => 100,
                            Difficulty::Medium => 70,
                            Difficulty::Hard => 45,
                            Difficulty::Impossible => 30,
                        };
                        if let Execution::Nation { next_tick, .. } = &mut self.executions[i] {
                            *next_tick = self.ticks + interval;
                        }
                    }
                }
                Execution::WinCheck => {
                    if self.ticks % 10 == 0 {
                        self.check_win();
                    }
                }
            }
            i += 1;
        }

        // Decay traitor flags
        for p in &mut self.players {
            if p.is_traitor && self.ticks >= p.traitor_until {
                p.is_traitor = false;
            }
        }

        // Nuke projectiles (simplified): advance & detonate
        self.tick_nukes();

        // Warship simplified patrol/engage
        self.tick_warships();

        // Trade gold from ports
        if self.ticks % 10 == 0 {
            self.tick_trade();
        }

        self.executions.retain(|e| match e {
            Execution::Attack(a) => a.active,
            Execution::Boat(b) => b.active,
            _ => true,
        });

        self.ticks += 1;
    }

    fn tick_economy(&mut self, pi: usize) {
        if self.players[pi].tiles.is_empty() {
            return;
        }
        let pt = self.players[pi].player_type.into();
        let city_levels: f64 = self.players[pi]
            .units
            .iter()
            .filter(|u| u.unit_type == UnitType::City)
            .map(|u| u.level as f64)
            .sum();
        let max = self
            .config
            .max_troops(pt, self.players[pi].tiles.len() as u32, city_levels);
        let inc = self
            .config
            .troop_increase_rate(pt, self.players[pi].troops, max);
        self.players[pi].troops += inc;
        self.players[pi].gold += self.config.gold_addition_rate(pt);
    }

    fn check_win(&mut self) {
        let pct = self.config.percentage_tiles_owned_to_win();
        let land = self.map.num_land_tiles() as f64;
        for p in &self.players {
            if p.tiles.is_empty() {
                continue;
            }
            let share = (p.tiles.len() as f64) * 100.0 / land;
            if share >= pct {
                self.winner = Some(p.small_id);
                return;
            }
        }
        // Last standing
        let alive: Vec<_> = self
            .players
            .iter()
            .filter(|p| !p.tiles.is_empty() && p.player_type != PlayerType::Bot)
            .map(|p| p.small_id)
            .collect();
        if alive.len() == 1 {
            self.winner = Some(alive[0]);
        }
    }

    fn tick_nukes(&mut self) {
        let mut detonations: Vec<(usize, usize, UnitType, TileRef)> = Vec::new();
        for (pi, p) in self.players.iter().enumerate() {
            for (ui, u) in p.units.iter().enumerate() {
                if !matches!(
                    u.unit_type,
                    UnitType::AtomBomb | UnitType::HydrogenBomb | UnitType::Mirv
                ) {
                    continue;
                }
                if let Some(tgt) = u.target_tile {
                    if self.map.manhattan_dist(u.tile, tgt) <= 10 {
                        detonations.push((pi, ui, u.unit_type, tgt));
                    }
                }
            }
        }
        // Move nukes toward target
        for p in &mut self.players {
            for u in &mut p.units {
                if !matches!(
                    u.unit_type,
                    UnitType::AtomBomb | UnitType::HydrogenBomb | UnitType::Mirv
                ) {
                    continue;
                }
                if let Some(tgt) = u.target_tile {
                    let speed = match u.unit_type {
                        UnitType::Mirv => 15u32,
                        _ => 10,
                    };
                    // Step toward target
                    let cx = self.map.x(u.tile) as i32;
                    let cy = self.map.y(u.tile) as i32;
                    let tx = self.map.x(tgt) as i32;
                    let ty = self.map.y(tgt) as i32;
                    let dx = (tx - cx).signum() * speed.min(cx.abs_diff(tx) as u32) as i32;
                    let dy = (ty - cy).signum() * speed.min(cy.abs_diff(ty) as u32) as i32;
                    let nx = (cx + dx).clamp(0, self.map.width() as i32 - 1) as u32;
                    let ny = (cy + dy).clamp(0, self.map.height() as i32 - 1) as u32;
                    u.tile = self.map.ref_xy(nx, ny);
                }
            }
        }
        // SAM intercept
        detonations.retain(|(pi, ui, _, tgt)| {
            let nuke_tile = self.players[*pi].units.get(*ui).map(|u| u.tile).unwrap_or(*tgt);
            !self.sam_intercepts(nuke_tile)
        });
        // Apply fallout — remove ownership in blast
        let mut to_remove_units: Vec<(usize, u32)> = Vec::new();
        for (pi, ui, ut, tgt) in detonations {
            let (inner, outer) = self.config.nuke_magnitudes(if ut == UnitType::Mirv {
                UnitType::AtomBomb
            } else {
                ut
            });
            let uid = self.players[pi].units[ui].id;
            to_remove_units.push((pi, uid));
            let cx = self.map.x(tgt) as i32;
            let cy = self.map.y(tgt) as i32;
            let rmax = outer as i32;
            for dy in -rmax..=rmax {
                for dx in -rmax..=rmax {
                    let d2 = (dx * dx + dy * dy) as u32;
                    if d2 > outer * outer {
                        continue;
                    }
                    let x = cx + dx;
                    let y = cy + dy;
                    if x < 0 || y < 0 {
                        continue;
                    }
                    let (x, y) = (x as u32, y as u32);
                    if !self.map.is_valid_coord(x, y) {
                        continue;
                    }
                    let r = self.map.ref_xy(x, y);
                    if !self.map.is_land(r) {
                        continue;
                    }
                    if d2 <= inner * inner {
                        let owner = self.map.owner_id(r);
                        if owner != 0 {
                            if let Some(oi) = self.player_by_small(owner) {
                                self.players[oi].tiles.remove(&r);
                            }
                        }
                        self.map.set_owner_id(r, 0);
                        self.map.set_fallout(r, true);
                    } else {
                        self.map.set_fallout(r, true);
                    }
                }
            }
        }
        for (pi, uid) in to_remove_units {
            self.players[pi].units.retain(|u| u.id != uid);
        }
    }

    fn sam_intercepts(&self, nuke_tile: TileRef) -> bool {
        for p in &self.players {
            for u in &p.units {
                if u.unit_type == UnitType::SamLauncher
                    && self.map.manhattan_dist(u.tile, nuke_tile) < 70
                {
                    return true;
                }
            }
        }
        false
    }

    fn tick_warships(&mut self) {
        // Capture nearby enemy transport / trade — simplified engagement
        let mut sink: Vec<(usize, u32)> = Vec::new();
        let positions: Vec<(usize, u32, TileRef, UnitType, Option<f64>)> = self
            .players
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| {
                p.units.iter().map(move |u| {
                    (pi, u.id, u.tile, u.unit_type, u.troops)
                })
            })
            .collect();

        for (pi, p) in self.players.iter().enumerate() {
            for u in &p.units {
                if u.unit_type != UnitType::Warship {
                    continue;
                }
                for &(opi, oid, otile, otype, otroops) in &positions {
                    if opi == pi {
                        continue;
                    }
                    if otype != UnitType::TransportShip {
                        continue;
                    }
                    if self.map.manhattan_dist(u.tile, otile) <= 130 {
                        sink.push((opi, oid));
                        let _ = otroops;
                    }
                }
                // Move toward patrol target
            }
        }
        for (pi, uid) in sink {
            if let Some(pos) = self.players[pi].units.iter().position(|u| u.id == uid) {
                self.players[pi].units.remove(pos);
            }
            // Deactivate boat execs
            for ex in &mut self.executions {
                if let Execution::Boat(b) = ex {
                    if b.unit_id == uid {
                        b.active = false;
                    }
                }
            }
        }
        // Move warships one tile toward target
        let w = self.map.width();
        let h = self.map.height();
        for p in &mut self.players {
            for u in &mut p.units {
                if u.unit_type != UnitType::Warship {
                    continue;
                }
                let Some(tgt) = u.target_tile else { continue };
                let cx = (u.tile % w) as i32;
                let cy = (u.tile / w) as i32;
                let tx = (tgt % w) as i32;
                let ty = (tgt / w) as i32;
                let nx = (cx + (tx - cx).signum()).clamp(0, w as i32 - 1) as u32;
                let ny = (cy + (ty - cy).signum()).clamp(0, h as i32 - 1) as u32;
                let nr = ny * w + nx;
                // Prefer water
                u.tile = nr;
            }
        }
    }

    fn tick_trade(&mut self) {
        for pi in 0..self.players.len() {
            let ports: Vec<TileRef> = self.players[pi]
                .units
                .iter()
                .filter(|u| u.unit_type == UnitType::Port)
                .map(|u| u.tile)
                .collect();
            if ports.is_empty() {
                continue;
            }
            // Income bonus per port (simplified trade)
            let bonus = 500i64 * ports.len() as i64;
            self.players[pi].gold += bonus;
        }
    }

    /// Game hash matching TS: sum of player hashes every call.
    pub fn hash(&self) -> i64 {
        let mut hash: i64 = 1;
        for p in &self.players {
            hash += p.hash();
        }
        hash
    }

    pub fn compact_snapshot(&self) -> CompactSnapshot {
        CompactSnapshot {
            tick: self.ticks,
            hash: self.hash(),
            owners_checksum: self.map.owners_checksum(),
            players: self
                .players
                .iter()
                .map(|p| PlayerSnap {
                    id: p.id.clone(),
                    small_id: p.small_id,
                    gold: p.gold,
                    troops: p.troops,
                    tiles: p.tiles.len() as u32,
                    units: p.units.iter().map(|u| u.id).collect(),
                })
                .collect(),
            winner: self.winner,
        }
    }

    pub fn is_done(&self) -> bool {
        self.winner.is_some()
    }

    pub fn defense_post_near(&self, defender_small: SmallId, tile: TileRef) -> bool {
        let range = self.config.defense_post_range() as u32;
        for p in &self.players {
            if p.small_id != defender_small {
                continue;
            }
            for u in &p.units {
                if u.unit_type == UnitType::DefensePost
                    && self.map.manhattan_dist(u.tile, tile) <= range
                {
                    return true;
                }
            }
        }
        false
    }
}

fn parse_unit(s: &str) -> Option<UnitType> {
    match s {
        "City" => Some(UnitType::City),
        "Port" => Some(UnitType::Port),
        "Factory" => Some(UnitType::Factory),
        "DefensePost" => Some(UnitType::DefensePost),
        "SAMLauncher" | "SamLauncher" => Some(UnitType::SamLauncher),
        "MissileSilo" => Some(UnitType::MissileSilo),
        "Warship" => Some(UnitType::Warship),
        "TransportShip" => Some(UnitType::TransportShip),
        "AtomBomb" => Some(UnitType::AtomBomb),
        "HydrogenBomb" => Some(UnitType::HydrogenBomb),
        "MIRV" | "Mirv" => Some(UnitType::Mirv),
        _ => None,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactSnapshot {
    pub tick: u32,
    pub hash: i64,
    pub owners_checksum: u64,
    pub players: Vec<PlayerSnap>,
    pub winner: Option<SmallId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerSnap {
    pub id: String,
    pub small_id: SmallId,
    pub gold: i64,
    pub troops: f64,
    pub tiles: u32,
    pub units: Vec<u32>,
}
