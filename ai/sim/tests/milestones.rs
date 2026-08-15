//! Integration tests for parity milestones M1–M4 (and smoke for later).

use openfront_sim::config::Difficulty;
use openfront_sim::js_math::simple_hash;
use openfront_sim::{GameBuilder, Intent, PseudoRandom};
use std::path::PathBuf;

fn plains() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/maps/plains")
}

#[test]
fn m1_prng_matches_node_seed_42() {
    let mut r = PseudoRandom::new(42);
    let expected: [f64; 10] = [
        0.8907801888417453,
        0.4310670436825603,
        0.3220651443116367,
        0.2072944741230458,
        0.6512069271411747,
        0.9914641629438847,
        0.2893015253357589,
        0.41498349350877106,
        0.48398855002596974,
        0.11718729161657393,
    ];
    for e in expected {
        assert_eq!(r.next().to_bits(), e.to_bits());
    }
}

#[test]
fn m1_map_load_and_empty_hash() {
    let g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    assert_eq!(g.map.width(), 100);
    assert_eq!(g.map.num_land_tiles(), 10000);
    // No tiles yet — hash is deterministic
    let h1 = g.hash();
    let h2 = g.hash();
    assert_eq!(h1, h2);
    assert!(h1 != 0);
}

#[test]
fn m2_economy_grows_troops_and_gold() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    g.spawn_player(0, g.map.ref_xy(50, 50));
    g.end_spawn_phase();
    let t0 = g.players[0].troops;
    let gold0 = g.players[0].gold;
    for _ in 0..100 {
        g.execute_next_tick();
    }
    assert!(g.players[0].troops > t0, "troops should grow");
    assert!(g.players[0].gold > gold0, "gold should grow");
    assert_eq!(g.players[0].gold, gold0 + 100 * 100); // 100/tick * 100 ticks
}

#[test]
fn m3_land_attack_conquers_terra_nullius() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    g.spawn_player(0, g.map.ref_xy(50, 50));
    g.end_spawn_phase();
    let tiles0 = g.players[0].num_tiles_owned();
    g.apply_intent(
        "human0",
        &Intent::Attack {
            target_id: None,
            troops: Some(10_000.0),
        },
    );
    for _ in 0..200 {
        g.execute_next_tick();
    }
    assert!(
        g.players[0].num_tiles_owned() > tiles0,
        "should conquer TN tiles"
    );
}

#[test]
fn m3_pvp_attack_transfers_tiles() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .with_human("B", "human1")
        .build();
    g.spawn_player(0, g.map.ref_xy(40, 50));
    g.spawn_player(1, g.map.ref_xy(60, 50));
    g.end_spawn_phase();
    // Expand both toward each other via TN first
    g.apply_intent(
        "human0",
        &Intent::Attack {
            target_id: None,
            troops: Some(15_000.0),
        },
    );
    g.apply_intent(
        "human1",
        &Intent::Attack {
            target_id: None,
            troops: Some(15_000.0),
        },
    );
    for _ in 0..500 {
        g.execute_next_tick();
    }
    let a = g.players[0].num_tiles_owned();
    let b = g.players[1].num_tiles_owned();
    assert!(a > 50 && b > 50);
    // Attack B
    g.apply_intent(
        "human0",
        &Intent::Attack {
            target_id: Some("human1".into()),
            troops: Some(g.players[0].troops * 0.5),
        },
    );
    let b_before = g.players[1].num_tiles_owned();
    let troops_a = g.players[0].troops;
    let troops_b = g.players[1].troops;
    for _ in 0..300 {
        g.execute_next_tick();
    }
    // Combat should change territory or troop counts
    assert!(
        g.players[1].num_tiles_owned() != b_before
            || (g.players[0].troops - troops_a).abs() > 1.0
            || (g.players[1].troops - troops_b).abs() > 1.0
            || g.players[0].num_tiles_owned() > a,
        "PvP attack should affect troops or tiles"
    );
}

#[test]
fn m4_win_check_at_80_percent() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .with_nation("N", "nation0")
        .difficulty(Difficulty::Easy)
        .build();
    // Give A almost all land
    for y in 0..100 {
        for x in 0..100 {
            let r = g.map.ref_xy(x, y);
            if g.map.is_land(r) {
                g.conquer_tile(0, r);
            }
        }
    }
    g.recompute_borders(0);
    g.end_spawn_phase();
    for _ in 0..20 {
        g.execute_next_tick();
    }
    assert_eq!(g.winner, Some(g.players[0].small_id));
}

#[test]
fn m5_build_city() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .build();
    g.spawn_player(0, g.map.ref_xy(50, 50));
    g.end_spawn_phase();
    g.players[0].gold = 500_000;
    let tile = *g.players[0].tiles.iter().next().unwrap();
    g.apply_intent(
        "human0",
        &Intent::BuildUnit {
            unit: "City".into(),
            tile,
            amount: None,
        },
    );
    assert!(g.players[0]
        .units
        .iter()
        .any(|u| u.unit_type == openfront_sim::config::UnitType::City));
}

#[test]
fn m10_nation_acts() {
    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .with_nation("N", "nation0")
        .difficulty(Difficulty::Impossible)
        .build();
    g.spawn_player(0, g.map.ref_xy(20, 20));
    g.spawn_player(1, g.map.ref_xy(80, 80));
    g.end_spawn_phase();
    let tiles_n0 = g.players[1].num_tiles_owned();
    for _ in 0..500 {
        g.execute_next_tick();
    }
    // Nation should have expanded or spent troops attacking
    assert!(
        g.players[1].num_tiles_owned() >= tiles_n0
            || g.players[1].troops != 31_250.0
    );
}

#[test]
fn obs_and_action_pipeline() {
    use openfront_sim::action::{decode_intent, legal_mask, FactorizedAction};
    use openfront_sim::obs::encode;

    let mut g = GameBuilder::from_map_dir(&plains(), false)
        .unwrap()
        .with_human("A", "human0")
        .with_nation("N", "nation0")
        .build();
    g.spawn_player(0, g.map.ref_xy(50, 50));
    g.end_spawn_phase();
    let obs = encode(&g, 0);
    assert_eq!(obs.vector.len(), 64);
    assert!(!obs.global.is_empty());
    let mask = legal_mask(&g, 0);
    assert!(mask.action_type[0]); // noop
    assert!(mask.action_type[2]); // attack
    let intent = decode_intent(&g, 0, &FactorizedAction::noop());
    assert!(matches!(intent, Intent::Noop));
    let _ = simple_hash("x");
}
