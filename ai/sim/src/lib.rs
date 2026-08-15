//! OpenFront deterministic simulation — Rust port of `src/core`.
//!
//! Parity with the TypeScript oracle is the hard gate. Float ops that must
//! match V8 go through [`js_math`].

pub mod action;
pub mod config;
pub mod game;
pub mod hash;
pub mod js_math;
pub mod map;
pub mod nation;
pub mod obs;
pub mod path;
pub mod rng;

pub use config::Config;
pub use game::{Game, GameBuilder, Intent, PlayerType};
pub use map::{GameMap, MapManifest, TileRef};
pub use rng::PseudoRandom;

/// Convenience re-exports used by FFI / training.
pub mod prelude {
    pub use crate::action::{decode_intent, legal_mask, shaped_reward, FactorizedAction};
    pub use crate::obs::{encode, Observation, GLOBAL_C, GLOBAL_H, GLOBAL_W, LOCAL_C, LOCAL_H, LOCAL_W, VECTOR_DIM};
    pub use crate::{Config, Game, GameBuilder, Intent, PseudoRandom};
}
