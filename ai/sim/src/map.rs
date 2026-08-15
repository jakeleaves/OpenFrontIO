//! Game map — port of `src/core/game/GameMap.ts` layout.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

pub type TileRef = u32;

const IS_LAND_BIT: u8 = 7;
const SHORELINE_BIT: u8 = 6;
const OCEAN_BIT: u8 = 5;
const MAGNITUDE_MASK: u8 = 0x1f;
const IMPASSABLE_MAGNITUDE: u8 = 31;

const PLAYER_ID_MASK: u16 = 0xfff;
const FALLOUT_BIT: u16 = 13;
const DEFENSE_BONUS_BIT: u16 = 14;

#[derive(Debug, Error)]
pub enum MapError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid terrain length {got} for {w}x{h}")]
    BadLength { got: usize, w: u32, h: u32 },
    #[error("{0}")]
    Msg(String),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MapMetadata {
    pub width: u32,
    pub height: u32,
    pub num_land_tiles: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MapManifest {
    pub name: String,
    pub map: MapMetadata,
    pub map4x: MapMetadata,
    pub map16x: MapMetadata,
    #[serde(default)]
    pub nations: Vec<NationSpawn>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct NationSpawn {
    pub name: String,
    pub coordinates: Option<[u32; 2]>,
    pub flag: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainType {
    Plains,
    Highland,
    Mountain,
    Impassable,
    Water,
}

#[derive(Clone, Debug)]
pub struct GameMap {
    width: u32,
    height: u32,
    num_land_tiles: u32,
    terrain: Vec<u8>,
    state: Vec<u16>,
    num_fallout: u32,
}

impl GameMap {
    pub fn new(width: u32, height: u32, terrain: Vec<u8>, num_land_tiles: u32) -> Result<Self, MapError> {
        let expected = (width as usize) * (height as usize);
        if terrain.len() != expected {
            return Err(MapError::BadLength {
                got: terrain.len(),
                w: width,
                h: height,
            });
        }
        Ok(Self {
            width,
            height,
            num_land_tiles,
            terrain,
            state: vec![0u16; expected],
            num_fallout: 0,
        })
    }

    pub fn from_bin(meta: &MapMetadata, data: &[u8]) -> Result<Self, MapError> {
        Self::new(meta.width, meta.height, data.to_vec(), meta.num_land_tiles)
    }

    pub fn load_pair(dir: &Path, compact: bool) -> Result<(Self, Self, MapManifest), MapError> {
        let manifest: MapManifest =
            serde_json::from_str(&fs::read_to_string(dir.join("manifest.json"))?)?;
        let (full_meta, full_name, mini_meta, mini_name) = if compact {
            (
                &manifest.map4x,
                "map4x.bin",
                &manifest.map16x,
                "map16x.bin",
            )
        } else {
            (&manifest.map, "map.bin", &manifest.map4x, "map4x.bin")
        };
        let full = Self::from_bin(full_meta, &fs::read(dir.join(full_name))?)?;
        let mini = Self::from_bin(mini_meta, &fs::read(dir.join(mini_name))?)?;
        Ok((full, mini, manifest))
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[inline]
    pub fn num_land_tiles(&self) -> u32 {
        self.num_land_tiles
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.terrain.len()
    }

    #[inline]
    pub fn ref_xy(&self, x: u32, y: u32) -> TileRef {
        debug_assert!(self.is_valid_coord(x, y));
        y * self.width + x
    }

    #[inline]
    pub fn is_valid_coord(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    #[inline]
    pub fn is_valid_ref(&self, r: TileRef) -> bool {
        (r as usize) < self.terrain.len()
    }

    #[inline]
    pub fn x(&self, r: TileRef) -> u32 {
        r % self.width
    }
    #[inline]
    pub fn y(&self, r: TileRef) -> u32 {
        r / self.width
    }

    #[inline]
    fn terr(&self, r: TileRef) -> u8 {
        self.terrain[r as usize]
    }

    #[inline]
    pub fn is_land(&self, r: TileRef) -> bool {
        (self.terr(r) >> IS_LAND_BIT) & 1 == 1
    }
    #[inline]
    pub fn is_water(&self, r: TileRef) -> bool {
        !self.is_land(r)
    }
    #[inline]
    pub fn is_shoreline(&self, r: TileRef) -> bool {
        (self.terr(r) >> SHORELINE_BIT) & 1 == 1
    }
    #[inline]
    pub fn is_ocean(&self, r: TileRef) -> bool {
        (self.terr(r) >> OCEAN_BIT) & 1 == 1
    }
    #[inline]
    pub fn magnitude(&self, r: TileRef) -> u8 {
        self.terr(r) & MAGNITUDE_MASK
    }
    #[inline]
    pub fn is_impassable(&self, r: TileRef) -> bool {
        self.is_land(r) && self.magnitude(r) == IMPASSABLE_MAGNITUDE
    }
    #[inline]
    pub fn terrain_byte(&self, r: TileRef) -> u8 {
        self.terr(r)
    }

    pub fn terrain_type(&self, r: TileRef) -> TerrainType {
        if !self.is_land(r) {
            return TerrainType::Water;
        }
        let mag = self.magnitude(r);
        if mag == IMPASSABLE_MAGNITUDE {
            TerrainType::Impassable
        } else if mag < 10 {
            TerrainType::Plains
        } else if mag < 20 {
            TerrainType::Highland
        } else {
            TerrainType::Mountain
        }
    }

    #[inline]
    pub fn owner_id(&self, r: TileRef) -> u16 {
        self.state[r as usize] & PLAYER_ID_MASK
    }
    #[inline]
    pub fn has_owner(&self, r: TileRef) -> bool {
        self.owner_id(r) != 0
    }
    pub fn set_owner_id(&mut self, r: TileRef, player_id: u16) {
        let s = &mut self.state[r as usize];
        *s = (*s & !PLAYER_ID_MASK) | (player_id & PLAYER_ID_MASK);
    }

    #[inline]
    pub fn has_fallout(&self, r: TileRef) -> bool {
        (self.state[r as usize] >> FALLOUT_BIT) & 1 == 1
    }
    pub fn set_fallout(&mut self, r: TileRef, value: bool) {
        let had = self.has_fallout(r);
        if value && !had {
            self.state[r as usize] |= 1 << FALLOUT_BIT;
            self.num_fallout += 1;
        } else if !value && had {
            self.state[r as usize] &= !(1 << FALLOUT_BIT);
            self.num_fallout = self.num_fallout.saturating_sub(1);
        }
    }
    pub fn num_tiles_with_fallout(&self) -> u32 {
        self.num_fallout
    }

    #[inline]
    pub fn has_defense_bonus(&self, r: TileRef) -> bool {
        (self.state[r as usize] >> DEFENSE_BONUS_BIT) & 1 == 1
    }
    pub fn set_defense_bonus(&mut self, r: TileRef, value: bool) {
        if value {
            self.state[r as usize] |= 1 << DEFENSE_BONUS_BIT;
        } else {
            self.state[r as usize] &= !(1 << DEFENSE_BONUS_BIT);
        }
    }

    #[inline]
    pub fn tile_state(&self, r: TileRef) -> u16 {
        self.state[r as usize]
    }

    pub fn tile_state_buffer(&self) -> &[u16] {
        &self.state
    }

    /// Cardinal neighbors N,S,W,E — same order as TS `neighbors()`.
    pub fn neighbors4(&self, r: TileRef, out: &mut [TileRef; 4]) -> usize {
        let x = self.x(r);
        let y = self.y(r);
        let mut n = 0;
        if y > 0 {
            out[n] = self.ref_xy(x, y - 1);
            n += 1;
        }
        if y + 1 < self.height {
            out[n] = self.ref_xy(x, y + 1);
            n += 1;
        }
        if x > 0 {
            out[n] = self.ref_xy(x - 1, y);
            n += 1;
        }
        if x + 1 < self.width {
            out[n] = self.ref_xy(x + 1, y);
            n += 1;
        }
        n
    }

    pub fn for_each_neighbor<F: FnMut(TileRef)>(&self, r: TileRef, mut f: F) {
        let mut buf = [0u32; 4];
        let n = self.neighbors4(r, &mut buf);
        for i in 0..n {
            f(buf[i]);
        }
    }

    pub fn manhattan_dist(&self, a: TileRef, b: TileRef) -> u32 {
        let dx = self.x(a).abs_diff(self.x(b));
        let dy = self.y(a).abs_diff(self.y(b));
        dx + dy
    }

    pub fn euclidean_dist_squared(&self, a: TileRef, b: TileRef) -> u64 {
        let dx = self.x(a) as i64 - self.x(b) as i64;
        let dy = self.y(a) as i64 - self.y(b) as i64;
        (dx * dx + dy * dy) as u64
    }

    /// Owner-id checksum for parity dumps (sum of ownerID over all tiles).
    pub fn owners_checksum(&self) -> u64 {
        self.state.iter().map(|&s| (s & PLAYER_ID_MASK) as u64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn load_plains() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/testdata/maps/plains");
        if !root.exists() {
            return;
        }
        let (map, mini, man) = GameMap::load_pair(&root, false).unwrap();
        assert_eq!(map.width(), 100);
        assert_eq!(map.height(), 100);
        assert_eq!(map.num_land_tiles(), 10000);
        assert_eq!(mini.width(), 50);
        assert_eq!(man.name, "Plains");
        assert!(map.is_land(map.ref_xy(50, 50)));
    }
}
