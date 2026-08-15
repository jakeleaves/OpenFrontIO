//! Hash helpers and parity dump format.

use crate::game::Game;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashCheckpoint {
    pub tick: u32,
    pub hash: i64,
}

pub fn collect_hash_if_due(game: &Game) -> Option<HashCheckpoint> {
    // TS emits hash when ticks % 10 == 0 *before* increment; after execute_next_tick
    // our ticks have already been incremented. Callers should snapshot mid-tick.
    if game.ticks % 10 == 0 {
        Some(HashCheckpoint {
            tick: game.ticks,
            hash: game.hash(),
        })
    } else {
        None
    }
}
