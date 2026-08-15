use crate::config::UnitType;
use crate::js_math::simple_hash;
use crate::map::TileRef;

#[derive(Clone, Debug)]
pub struct Unit {
    pub id: u32,
    pub unit_type: UnitType,
    pub tile: TileRef,
    pub owner_idx: usize,
    pub level: u32,
    pub troops: Option<f64>,
    pub target_tile: Option<TileRef>,
    pub health: f64,
}

impl Unit {
    pub fn new(id: u32, unit_type: UnitType, tile: TileRef, owner_idx: usize) -> Self {
        let health = match unit_type {
            UnitType::Warship => 1000.0,
            _ => 100.0,
        };
        Self {
            id,
            unit_type,
            tile,
            owner_idx,
            level: 0,
            troops: None,
            target_tile: None,
            health,
        }
    }

    /// Port of UnitImpl.hash(): tile + simpleHash(type) * id
    pub fn hash(&self) -> i64 {
        self.tile as i64 + (simple_hash(self.unit_type.as_str()) as i64) * (self.id as i64)
    }
}
