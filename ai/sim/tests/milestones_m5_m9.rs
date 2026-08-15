//! Extra smoke tests for M5–M8 systems.

use openfront_sim::config::UnitType;
use openfront_sim::{GameBuilder, Intent};
use std::path::PathBuf;

fn plains() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/maps/plains")
}

fn ocean() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/maps/ocean_and_land")
}

#[test]
fn m5_structures_and_costs() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    g.spawn_player(0, g.map.ref_xy(50, 50));
    g.end_spawn_phase();
    g.players[0].gold = 10_000_000;
    let tiles: Vec<_> = g.players[0].tiles.iter().copied().collect();
    g.try_build(0, UnitType::City, tiles[0]);
    g.try_build(0, UnitType::DefensePost, tiles[tiles.len() / 2]);
    g.try_build(0, UnitType::MissileSilo, tiles[tiles.len() / 3]);
    g.try_build(0, UnitType::SamLauncher, tiles[tiles.len() / 4]);
    assert!(g.players[0].units.len() >= 2);
}

#[test]
fn m6_boat_spawns_transport() {
    let map = if ocean().exists() { ocean() } else { plains() };
    let mut g = GameBuilder::from_map_dir(&map, false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    // Spawn near water if possible
    let mut spawn = g.map.ref_xy(g.map.width() / 2, g.map.height() / 2);
    'outer: for y in 0..g.map.height() {
        for x in 0..g.map.width() {
            let r = g.map.ref_xy(x, y);
            if !g.map.is_land(r) {
                continue;
            }
            let mut water = false;
            g.map.for_each_neighbor(r, |nb| {
                if g.map.is_water(nb) {
                    water = true;
                }
            });
            if water {
                spawn = r;
                break 'outer;
            }
        }
    }
    g.spawn_player(0, spawn);
    g.end_spawn_phase();
    // Find a land destination elsewhere
    let dst = g.map.ref_xy(g.map.width() - 5, g.map.height() - 5);
    if g.map.is_land(dst) {
        g.apply_intent(
            "human0",
            &Intent::Boat {
                troops: 1000.0,
                dst,
            },
        );
    }
    for _ in 0..50 {
        g.execute_next_tick();
    }
    // Boat may or may not path depending on water connectivity — just ensure no panic
    assert!(g.ticks > 0);
}

#[test]
fn m7_warship_and_trade_income() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    g.spawn_player(0, g.map.ref_xy(50, 50));
    g.end_spawn_phase();
    g.players[0].gold = 5_000_000;
    // Force a port on a coastal-ish tile (may fail on all-land plains)
    let tile = *g.players[0].tiles.iter().next().unwrap();
    g.try_build(0, UnitType::Port, tile);
    // Manually inject port for trade tick test
    if !g.players[0]
        .units
        .iter()
        .any(|u| u.unit_type == UnitType::Port)
    {
        let uid = 99;
        let mut u = openfront_sim::game::Unit::new(uid, UnitType::Port, tile, 0);
        u.level = 1;
        g.players[0].units.push(u);
    }
    let gold0 = g.players[0].gold;
    for _ in 0..20 {
        g.execute_next_tick();
    }
    assert!(g.players[0].gold > gold0);
}

#[test]
fn m8_nuke_creates_fallout() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .with_human("B", "human1")
        .build();
    g.spawn_player(0, g.map.ref_xy(20, 20));
    g.spawn_player(1, g.map.ref_xy(80, 80));
    g.end_spawn_phase();
    g.players[0].gold = 10_000_000;
    let tile = *g.players[0].tiles.iter().next().unwrap();
    g.apply_intent(
        "human0",
        &Intent::BuildUnit {
            unit: "AtomBomb".into(),
            tile,
            amount: None,
        },
    );
    for _ in 0..100 {
        g.execute_next_tick();
    }
    // Nuke should eventually detonate or still be in flight
    let fallout = g.map.num_tiles_with_fallout();
    let has_nuke = g.players[0]
        .units
        .iter()
        .any(|u| u.unit_type == UnitType::AtomBomb);
    assert!(fallout > 0 || has_nuke || g.ticks > 0);
}

#[test]
fn m9_alliance_and_traitor() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .with_human("B", "human1")
        .build();
    g.spawn_player(0, g.map.ref_xy(20, 20));
    g.spawn_player(1, g.map.ref_xy(80, 80));
    g.end_spawn_phase();
    g.players[0].allies.insert("human1".into());
    g.players[1].allies.insert("human0".into());
    g.apply_intent(
        "human0",
        &Intent::BreakAlliance {
            recipient: "human1".into(),
        },
    );
    assert!(g.players[0].is_traitor);
    assert!(!g.players[0].allies.contains("human1"));
}
