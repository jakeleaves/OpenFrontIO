use crate::config::UnitType;
use crate::js_math::simple_hash;
use crate::map::TileRef;
use super::{PlayerInfo, PlayerType, SmallId};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Player {
    pub id: String,
    pub name: String,
    pub player_type: PlayerType,
    pub client_id: Option<String>,
    pub small_id: SmallId,
    pub troops: f64,
    pub gold: i64,
    pub tiles: HashSet<TileRef>,
    pub border_tiles: HashSet<TileRef>,
    pub units: Vec<super::Unit>,
    pub spawn_tile: Option<TileRef>,
    pub last_tile_change: u32,
    pub is_traitor: bool,
    pub traitor_until: u32,
    pub allies: HashSet<String>,
    pub embargoes: HashSet<String>,
    pub pending_alliance_out: HashSet<String>,
    pub relations: std::collections::HashMap<String, f64>,
}

impl Player {
    pub fn new(info: PlayerInfo, small_id: SmallId, troops: f64) -> Self {
        Self {
            id: info.id,
            name: info.name,
            player_type: info.player_type,
            client_id: info.client_id,
            small_id,
            troops,
            gold: 0,
            tiles: HashSet::new(),
            border_tiles: HashSet::new(),
            units: Vec::new(),
            spawn_tile: None,
            last_tile_change: 0,
            is_traitor: false,
            traitor_until: 0,
            allies: HashSet::new(),
            embargoes: HashSet::new(),
            pending_alliance_out: HashSet::new(),
            relations: std::collections::HashMap::new(),
        }
    }

    pub fn num_tiles_owned(&self) -> u32 {
        self.tiles.len() as u32
    }

    pub fn units_of(&self, t: UnitType) -> impl Iterator<Item = &super::Unit> {
        self.units.iter().filter(move |u| u.unit_type == t)
    }

    /// Port of PlayerImpl.hash()
    pub fn hash(&self) -> i64 {
        let unit_hash: i64 = self.units.iter().map(|u| u.hash()).sum();
        (simple_hash(&self.id) as i64) * ((self.troops as i64) + self.num_tiles_owned() as i64)
            + unit_hash
    }
}
