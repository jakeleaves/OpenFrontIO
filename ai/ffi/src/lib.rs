//! Python bindings — batch stepper for RL training.

use numpy::{PyArray1, PyArray3, PyArrayMethods};
use openfront_sim::action::{
    decode_intent, legal_mask, shaped_reward, ActionType, FactorizedAction, NUM_ACTION_TYPES,
    NUM_BUILD_TYPES, NUM_TARGET_PLAYERS, NUM_TROOP_FRACS, COARSE_H, COARSE_W,
};
use openfront_sim::config::Difficulty;
use openfront_sim::obs::{
    encode, Observation, GLOBAL_C, GLOBAL_H, GLOBAL_W, LOCAL_C, LOCAL_H, LOCAL_W, VECTOR_DIM,
};
use openfront_sim::{Game, GameBuilder};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

struct EnvState {
    game: Game,
    ego: usize,
    prev_tiles: u32,
    prev_troops_diff: f64,
    prev_gold: i64,
    done: bool,
}

#[pyclass]
struct SimBatch {
    envs: Vec<EnvState>,
    map_dir: PathBuf,
}

#[pymethods]
impl SimBatch {
    #[new]
    #[pyo3(signature = (n, map_dir, seed=1, difficulty="impossible"))]
    fn new(n: usize, map_dir: String, seed: i32, difficulty: &str) -> PyResult<Self> {
        let map_dir = PathBuf::from(map_dir);
        let diff = match difficulty {
            "easy" => Difficulty::Easy,
            "medium" => Difficulty::Medium,
            "hard" => Difficulty::Hard,
            _ => Difficulty::Impossible,
        };
        let mut envs = Vec::with_capacity(n);
        for i in 0..n {
            let mut game = GameBuilder::from_map_dir(&map_dir, false)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .with_human("Agent", &format!("human-{i}"))
                .with_nation("Nation", &format!("nation-{i}"))
                .difficulty(diff)
                .build();
            game.game_id = format!("batch-{seed}-{i}");
            // Auto-spawn both near opposite corners
            let w = game.map.width();
            let h = game.map.height();
            game.spawn_player(0, game.map.ref_xy(w / 4, h / 4));
            game.spawn_player(1, game.map.ref_xy(3 * w / 4, 3 * h / 4));
            game.end_spawn_phase();
            let ego = 0usize;
            let enemy_troops = game.players[1].troops;
            envs.push(EnvState {
                prev_tiles: game.players[ego].num_tiles_owned(),
                prev_troops_diff: game.players[ego].troops - enemy_troops,
                prev_gold: game.players[ego].gold,
                done: false,
                game,
                ego,
            });
        }
        Ok(Self { envs, map_dir })
    }

    fn n(&self) -> usize {
        self.envs.len()
    }

    fn obs_shapes<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("global", (GLOBAL_C, GLOBAL_H, GLOBAL_W)).unwrap();
        d.set_item("local", (LOCAL_C, LOCAL_H, LOCAL_W)).unwrap();
        d.set_item("vector", VECTOR_DIM).unwrap();
        d
    }

    /// Reset env i; returns nothing (call observe).
    fn reset(&mut self, i: usize) -> PyResult<()> {
        let seed = (i as i32).wrapping_mul(9973);
        let mut game = GameBuilder::from_map_dir(&self.map_dir, false)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .with_human("Agent", &format!("human-{i}"))
            .with_nation("Nation", &format!("nation-{i}"))
            .build();
        game.game_id = format!("reset-{seed}");
        let w = game.map.width();
        let h = game.map.height();
        game.spawn_player(0, game.map.ref_xy(w / 4, h / 4));
        game.spawn_player(1, game.map.ref_xy(3 * w / 4, 3 * h / 4));
        game.end_spawn_phase();
        let ego = 0;
        self.envs[i] = EnvState {
            prev_tiles: game.players[ego].num_tiles_owned(),
            prev_troops_diff: game.players[ego].troops - game.players[1].troops,
            prev_gold: game.players[ego].gold,
            done: false,
            game,
            ego,
        };
        Ok(())
    }

    fn observe<'py>(
        &self,
        py: Python<'py>,
        i: usize,
    ) -> PyResult<(
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let obs = encode(&self.envs[i].game, self.envs[i].ego);
        Ok(obs_to_numpy(py, &obs))
    }

    /// Step with factorized action ints: (atype, target, cx, cy, frac, build)
    fn step(
        &mut self,
        i: usize,
        action_type: u8,
        target_player: u8,
        cell_x: u8,
        cell_y: u8,
        troop_frac: u8,
        build_type: u8,
    ) -> PyResult<(f64, bool)> {
        let env = &mut self.envs[i];
        if env.done {
            return Ok((0.0, true));
        }
        let boats_before = env.game.players[env.ego]
            .units
            .iter()
            .filter(|u| u.unit_type == openfront_sim::config::UnitType::TransportShip)
            .count();

        let fa = FactorizedAction {
            action_type,
            target_player,
            cell_x,
            cell_y,
            troop_frac,
            build_type,
        };
        let intent = decode_intent(&env.game, env.ego, &fa);
        let client = env.game.players[env.ego]
            .client_id
            .clone()
            .unwrap_or_else(|| env.game.players[env.ego].id.clone());
        env.game.apply_intent(&client, &intent);

        // Advance several ticks (micro step)
        for _ in 0..5 {
            env.game.execute_next_tick();
            if env.game.is_done() {
                break;
            }
        }

        let boats_after = env.game.players[env.ego]
            .units
            .iter()
            .filter(|u| u.unit_type == openfront_sim::config::UnitType::TransportShip)
            .count();
        let boat_sunk = boats_after < boats_before;

        let reward = shaped_reward(
            env.prev_tiles,
            env.prev_troops_diff,
            env.prev_gold,
            &env.game,
            env.ego,
            boat_sunk,
        );
        env.prev_tiles = env.game.players[env.ego].num_tiles_owned();
        let enemy_troops: f64 = env
            .game
            .players
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != env.ego)
            .map(|(_, p)| p.troops)
            .sum();
        env.prev_troops_diff = env.game.players[env.ego].troops - enemy_troops;
        env.prev_gold = env.game.players[env.ego].gold;
        env.done = env.game.is_done() || env.game.ticks > 20_000;
        Ok((reward, env.done))
    }

    fn legal_action_types<'py>(
        &self,
        py: Python<'py>,
        i: usize,
    ) -> Bound<'py, PyArray1<bool>> {
        let m = legal_mask(&self.envs[i].game, self.envs[i].ego);
        PyArray1::from_slice(py, &m.action_type)
    }

    fn tick(&self, i: usize) -> u32 {
        self.envs[i].game.ticks
    }

    fn hash(&self, i: usize) -> i64 {
        self.envs[i].game.hash()
    }

    fn snapshot_json(&self, i: usize) -> String {
        serde_json::to_string(&self.envs[i].game.compact_snapshot()).unwrap_or_default()
    }
}

fn obs_to_numpy<'py>(
    py: Python<'py>,
    obs: &Observation,
) -> (
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<f32>>,
) {
    let g = PyArray1::from_slice(py, &obs.global)
        .reshape([GLOBAL_C, GLOBAL_H, GLOBAL_W])
        .unwrap();
    let l = PyArray1::from_slice(py, &obs.local)
        .reshape([LOCAL_C, LOCAL_H, LOCAL_W])
        .unwrap();
    let v = PyArray1::from_slice(py, &obs.vector);
    (g, l, v)
}

#[pyfunction]
fn action_space_sizes() -> (usize, usize, usize, usize, usize, usize) {
    (
        NUM_ACTION_TYPES,
        NUM_TARGET_PLAYERS,
        COARSE_W,
        COARSE_H,
        NUM_TROOP_FRACS,
        NUM_BUILD_TYPES,
    )
}

#[pyfunction]
fn noop_action_type() -> u8 {
    ActionType::Noop as u8
}

#[pymodule]
fn openfront_sim_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimBatch>()?;
    m.add_function(wrap_pyfunction!(action_space_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(noop_action_type, m)?)?;
    Ok(())
}
