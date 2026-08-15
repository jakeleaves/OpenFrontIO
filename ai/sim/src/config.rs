//! Balance formulas — port of `src/core/configuration/Config.ts`.

use crate::js_math::{pow, sigmoid, within};
use crate::map::{GameMap, TerrainType, TileRef};

const DEFENSE_DEBUFF_MIDPOINT: f64 = 150_000.0;
const DEFENSE_DEBUFF_DECAY_RATE: f64 = std::f64::consts::LN_2 / 50_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Impossible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameMode {
    Ffa,
    Team,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    Bot,
    Nation,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub difficulty: Difficulty,
    pub game_mode: GameMode,
    pub gold_multiplier: f64,
    pub infinite_gold: bool,
    pub infinite_troops: bool,
    pub instant_build: bool,
    pub bots: u32,
    pub percentage_to_win_ffa: f64,
    pub percentage_to_win_team: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            difficulty: Difficulty::Impossible,
            game_mode: GameMode::Ffa,
            gold_multiplier: 1.0,
            infinite_gold: false,
            infinite_troops: false,
            instant_build: false,
            bots: 0,
            percentage_to_win_ffa: 80.0,
            percentage_to_win_team: 95.0,
        }
    }
}

impl Config {
    pub fn percentage_tiles_owned_to_win(&self) -> f64 {
        match self.game_mode {
            GameMode::Team => self.percentage_to_win_team,
            GameMode::Ffa => self.percentage_to_win_ffa,
        }
    }

    pub fn city_troop_increase(&self) -> f64 {
        250_000.0
    }

    pub fn defense_post_range(&self) -> f64 {
        30.0
    }
    pub fn defense_post_defense_bonus(&self) -> f64 {
        5.0
    }
    pub fn defense_post_speed_bonus(&self) -> f64 {
        3.0
    }
    pub fn traitor_defense_debuff(&self) -> f64 {
        0.5
    }
    pub fn traitor_speed_debuff(&self) -> f64 {
        0.8
    }
    pub fn fallout_defense_modifier(&self, ratio: f64) -> f64 {
        within(5.0 - ratio * 2.0, 2.5, 5.0)
    }

    pub fn start_manpower(&self, pt: PlayerType) -> f64 {
        match pt {
            PlayerType::Bot => 10_000.0,
            PlayerType::Nation => match self.difficulty {
                Difficulty::Easy => 12_500.0,
                Difficulty::Medium => 18_750.0,
                Difficulty::Hard => 25_000.0,
                Difficulty::Impossible => 31_250.0,
            },
            PlayerType::Human => {
                if self.infinite_troops {
                    1_000_000.0
                } else {
                    25_000.0
                }
            }
        }
    }

    pub fn max_troops(
        &self,
        pt: PlayerType,
        num_tiles: u32,
        city_levels_sum: f64,
    ) -> f64 {
        let base = if pt == PlayerType::Human && self.infinite_troops {
            1_000_000_000.0
        } else {
            2.0 * (pow(num_tiles as f64, 0.6) * 1000.0 + 50_000.0)
                + city_levels_sum * self.city_troop_increase()
        };
        match pt {
            PlayerType::Bot => base / 3.0,
            PlayerType::Human => base,
            PlayerType::Nation => match self.difficulty {
                Difficulty::Easy => base * 0.5,
                Difficulty::Medium => base * 0.75,
                Difficulty::Hard => base,
                Difficulty::Impossible => base * 1.25,
            },
        }
    }

    pub fn troop_increase_rate(
        &self,
        pt: PlayerType,
        troops: f64,
        max: f64,
    ) -> f64 {
        let mut to_add = 10.0 + pow(troops, 0.73) / 4.0;
        let ratio = 1.0 - troops / max;
        to_add *= ratio;
        if pt == PlayerType::Bot {
            to_add *= 0.5;
        }
        if pt == PlayerType::Nation {
            to_add *= match self.difficulty {
                Difficulty::Easy => 0.9,
                Difficulty::Medium => 0.95,
                Difficulty::Hard => 1.0,
                Difficulty::Impossible => 1.05,
            };
        }
        (troops + to_add).min(max) - troops
    }

    pub fn gold_addition_rate(&self, pt: PlayerType) -> i64 {
        let base = if pt == PlayerType::Bot { 50i64 } else { 100i64 };
        ((base as f64) * self.gold_multiplier).floor() as i64
    }

    pub fn attack_amount(&self, attacker_type: PlayerType, troops: f64) -> f64 {
        if attacker_type == PlayerType::Bot {
            troops / 20.0
        } else {
            troops / 5.0
        }
    }

    /// Simplified attackLogic for TN / PvP without nearby unit scan (caller
    /// passes defense_post_nearby / fallout / traitor flags).
    pub fn attack_logic(
        &self,
        map: &GameMap,
        attack_troops: f64,
        attacker_type: PlayerType,
        attacker_tiles: u32,
        defender: AttackDefender,
        tile: TileRef,
        defense_post_nearby: bool,
    ) -> AttackResult {
        let (mut mag, mut speed) = match map.terrain_type(tile) {
            TerrainType::Plains => (80.0, 16.5),
            TerrainType::Highland => (100.0, 20.0),
            TerrainType::Mountain => (120.0, 25.0),
            TerrainType::Impassable => panic!("impassable"),
            TerrainType::Water => panic!("water"),
        };

        if let AttackDefender::Player { .. } = defender {
            if defense_post_nearby {
                mag *= self.defense_post_defense_bonus();
                speed *= self.defense_post_speed_bonus();
            }
        }

        if map.has_fallout(tile) {
            let ratio = map.num_tiles_with_fallout() as f64 / map.num_land_tiles() as f64;
            let m = self.fallout_defense_modifier(ratio);
            mag *= m;
            speed *= m;
        }

        match defender {
            AttackDefender::TerraNullius => AttackResult {
                attacker_troop_loss: if attacker_type == PlayerType::Bot {
                    mag / 10.0
                } else {
                    mag / 5.0
                },
                defender_troop_loss: 0.0,
                tiles_per_tick_used: within(
                    (2000.0 * speed.max(10.0)) / attack_troops,
                    5.0,
                    100.0,
                ),
            },
            AttackDefender::Player {
                troops: def_troops,
                tiles: def_tiles,
                is_bot,
                is_traitor,
                is_disconnected_same_team,
            } => {
                if is_disconnected_same_team {
                    mag = 0.0;
                }
                if (attacker_type == PlayerType::Human || attacker_type == PlayerType::Nation)
                    && is_bot
                {
                    mag *= 0.7;
                }

                let defense_sig = 1.0
                    - sigmoid(
                        def_tiles as f64,
                        DEFENSE_DEBUFF_DECAY_RATE,
                        DEFENSE_DEBUFF_MIDPOINT,
                    );
                let large_defender_speed_debuff = 0.7 + 0.3 * defense_sig;
                let large_defender_attack_debuff = 0.7 + 0.3 * defense_sig;

                let large_attack_bonus = if attacker_tiles > 100_000 {
                    pow((100_000.0 / attacker_tiles as f64).sqrt(), 0.7)
                } else {
                    1.0
                };
                let large_attacker_speed_bonus = if attacker_tiles > 100_000 {
                    pow(100_000.0 / attacker_tiles as f64, 0.6)
                } else {
                    1.0
                };

                let defender_troop_loss = def_troops / def_tiles.max(1) as f64;
                let traitor_mod = if is_traitor {
                    self.traitor_defense_debuff()
                } else {
                    1.0
                };
                let current_attacker_loss = within(def_troops / attack_troops, 0.6, 2.0)
                    * mag
                    * 0.8
                    * large_defender_attack_debuff
                    * large_attack_bonus
                    * traitor_mod;
                let alt_attacker_loss = 1.3 * defender_troop_loss * (mag / 100.0) * traitor_mod;
                let attacker_troop_loss = 0.6 * current_attacker_loss + 0.4 * alt_attacker_loss;

                AttackResult {
                    attacker_troop_loss,
                    defender_troop_loss,
                    tiles_per_tick_used: within(def_troops / (5.0 * attack_troops), 0.2, 1.5)
                        * speed
                        * large_defender_speed_debuff
                        * large_attacker_speed_bonus
                        * if is_traitor {
                            self.traitor_speed_debuff()
                        } else {
                            1.0
                        },
                }
            }
        }
    }

    pub fn unit_cost(&self, unit: UnitType, num_owned: u32) -> i64 {
        if self.instant_build && self.infinite_gold {
            return 0;
        }
        match unit {
            UnitType::City | UnitType::Factory | UnitType::Port => {
                ((2u64.pow(num_owned) as f64) * 125_000.0).min(1_000_000.0) as i64
            }
            UnitType::DefensePost => ((num_owned as i64 + 1) * 50_000).min(250_000),
            UnitType::SamLauncher => ((num_owned as i64 + 1) * 1_500_000).min(3_000_000),
            UnitType::MissileSilo => 1_000_000,
            UnitType::Warship => ((num_owned as i64 + 1) * 250_000).min(1_000_000),
            UnitType::AtomBomb => 750_000,
            UnitType::HydrogenBomb => 5_000_000,
            UnitType::Mirv => 25_000_000,
            UnitType::TransportShip => 0,
        }
    }

    pub fn boat_max_number(&self) -> u32 {
        3
    }

    pub fn nuke_magnitudes(&self, unit: UnitType) -> (u32, u32) {
        match unit {
            UnitType::Mirv => (12, 18), // warhead default; MIRV itself not used for blast
            UnitType::AtomBomb => (12, 30),
            UnitType::HydrogenBomb => (80, 100),
            _ => (0, 0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AttackDefender {
    TerraNullius,
    Player {
        troops: f64,
        tiles: u32,
        is_bot: bool,
        is_traitor: bool,
        is_disconnected_same_team: bool,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct AttackResult {
    pub attacker_troop_loss: f64,
    pub defender_troop_loss: f64,
    pub tiles_per_tick_used: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitType {
    City,
    Port,
    Factory,
    DefensePost,
    SamLauncher,
    MissileSilo,
    Warship,
    TransportShip,
    AtomBomb,
    HydrogenBomb,
    Mirv,
}

impl UnitType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::City => "City",
            Self::Port => "Port",
            Self::Factory => "Factory",
            Self::DefensePost => "DefensePost",
            Self::SamLauncher => "SAMLauncher",
            Self::MissileSilo => "MissileSilo",
            Self::Warship => "Warship",
            Self::TransportShip => "TransportShip",
            Self::AtomBomb => "AtomBomb",
            Self::HydrogenBomb => "HydrogenBomb",
            Self::Mirv => "MIRV",
        }
    }
}
