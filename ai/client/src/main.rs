//! Local / self-hosted OpenFront WebSocket bot client.
//!
//! Connects to a game server, maintains a Rust sim mirror, and emits intents.
//! Does NOT target public openfront.io matchmaking (Gatekeeper).

use anyhow::{Context, Result};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use openfront_sim::action::{decode_intent, FactorizedAction, ActionType};
use openfront_sim::obs::encode;
use openfront_sim::{GameBuilder, Intent};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Parser, Debug)]
#[command(name = "openfront-bot")]
#[command(about = "Headless OpenFront agent client (local/self-hosted only)")]
struct Args {
    /// WebSocket URL, e.g. ws://localhost:3000/w0/
    #[arg(long, default_value = "ws://127.0.0.1:3000/w0/")]
    url: String,

    /// Game ID to join
    #[arg(long)]
    game_id: Option<String>,

    /// Map directory for local mirror (testdata or resources/maps/<name>)
    #[arg(long)]
    map_dir: PathBuf,

    /// Dry-run: run local 1v1 vs Nation without WebSocket
    #[arg(long, default_value_t = false)]
    local_demo: bool,

    /// Ticks for local demo
    #[arg(long, default_value_t = 500)]
    demo_ticks: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.local_demo {
        return local_demo(&args);
    }
    run_ws(&args).await
}

fn local_demo(args: &Args) -> Result<()> {
    let mut game = GameBuilder::from_map_dir(&args.map_dir, false)?
        .with_human("Bot", "bot0")
        .with_nation("Nation", "nation0")
        .build();
    let w = game.map.width();
    let h = game.map.height();
    game.spawn_player(0, game.map.ref_xy(w / 4, h / 4));
    game.spawn_player(1, game.map.ref_xy(3 * w / 4, 3 * h / 4));
    game.end_spawn_phase();

    println!(
        "Local demo: {}x{}, {} ticks vs Impossible Nation",
        w, h, args.demo_ticks
    );
    for t in 0..args.demo_ticks {
        // Simple heuristic policy: attack TN early, then enemy
        let fa = if t % 40 == 0 {
            FactorizedAction {
                action_type: ActionType::Attack as u8,
                target_player: if t < 200 { 0 } else { 1 },
                cell_x: 16,
                cell_y: 8,
                troop_frac: 2,
                build_type: 0,
            }
        } else if t % 100 == 50 && game.players[0].gold > 200_000 {
            FactorizedAction {
                action_type: ActionType::Build as u8,
                target_player: 0,
                cell_x: 8,
                cell_y: 8,
                troop_frac: 0,
                build_type: 0, // City
            }
        } else {
            FactorizedAction::noop()
        };
        let intent = decode_intent(&game, 0, &fa);
        game.apply_intent("bot0", &intent);
        game.execute_next_tick();
        if t % 100 == 0 {
            let obs = encode(&game, 0);
            println!(
                "t={} tiles={} troops={:.0} gold={} vec0={:.2} winner={:?}",
                game.ticks,
                game.players[0].num_tiles_owned(),
                game.players[0].troops,
                game.players[0].gold,
                obs.vector[0],
                game.winner
            );
        }
        if game.is_done() {
            println!("Game over at tick {}", game.ticks);
            break;
        }
    }
    let snap = game.compact_snapshot();
    println!("{}", serde_json::to_string_pretty(&snap)?);
    Ok(())
}

async fn run_ws(args: &Args) -> Result<()> {
    let game_id = args
        .game_id
        .clone()
        .context("--game-id required unless --local-demo")?;
    println!("Connecting to {} for game {}", args.url, game_id);
    let (ws, _) = connect_async(&args.url)
        .await
        .context("websocket connect failed")?;
    let (mut write, mut read) = ws.split();

    // Dev join (no JWT) — only works with GAME_ENV=dev servers
    let join = json!({
        "type": "join",
        "gameID": game_id,
        "clientID": "ai-bot-client",
        "token": "",
        "username": "AIBot",
        "clanTag": null,
        "cosmetics": null,
        "persistentID": "ai-bot-persistent",
    });
    write
        .send(Message::Text(join.to_string().into()))
        .await?;

    let mut mirror = GameBuilder::from_map_dir(&args.map_dir, false)?
        .with_human("AIBot", "ai-bot-client")
        .build();
    mirror.end_spawn_phase();

    while let Some(msg) = read.next().await {
        let msg = msg?;
        let Message::Text(text) = msg else { continue };
        let v: Value = serde_json::from_str(&text)?;
        match v.get("type").and_then(|t| t.as_str()) {
            Some("turn") => {
                if let Some(turn) = v.get("turn") {
                    apply_turn_intents(&mut mirror, turn);
                    mirror.execute_next_tick();
                    let intent = policy_intent(&mirror);
                    if !matches!(intent, Intent::Noop) {
                        let wire = intent_to_wire(&intent);
                        write
                            .send(Message::Text(
                                json!({"type":"intent","intent": wire}).to_string().into(),
                            ))
                            .await?;
                    }
                }
            }
            Some("start") => {
                println!("Game started");
            }
            Some("error") => {
                eprintln!("Server error: {v}");
                break;
            }
            Some("ping") => {
                write
                    .send(Message::Text(json!({"type":"ping"}).to_string().into()))
                    .await?;
            }
            _ => {}
        }
        if mirror.is_done() {
            break;
        }
    }
    Ok(())
}

fn apply_turn_intents(game: &mut openfront_sim::Game, turn: &Value) {
    let Some(intents) = turn.get("intents").and_then(|i| i.as_array()) else {
        return;
    };
    for intent in intents {
        let client = intent
            .get("clientID")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        if let Some(parsed) = wire_to_intent(intent) {
            game.apply_intent(client, &parsed);
        }
    }
}

fn policy_intent(game: &openfront_sim::Game) -> Intent {
    let ego = 0;
    if game.players.is_empty() || game.players[ego].tiles.is_empty() {
        return Intent::Noop;
    }
    let fa = FactorizedAction {
        action_type: ActionType::Attack as u8,
        target_player: 0,
        cell_x: 16,
        cell_y: 8,
        troop_frac: 2,
        build_type: 0,
    };
    decode_intent(game, ego, &fa)
}

fn intent_to_wire(intent: &Intent) -> Value {
    match intent {
        Intent::Attack { target_id, troops } => json!({
            "type": "attack",
            "targetID": target_id,
            "troops": troops,
        }),
        Intent::Boat { troops, dst } => json!({
            "type": "boat",
            "troops": troops,
            "dst": dst,
        }),
        Intent::BuildUnit { unit, tile, amount } => json!({
            "type": "build_unit",
            "unit": unit,
            "tile": tile,
            "amount": amount,
        }),
        Intent::Spawn { tile } => json!({"type":"spawn","tile": tile}),
        Intent::Noop => json!({"type":"emoji","recipient":"ALL","emoji":"👍"}),
        other => json!({"type":"noop","debug": format!("{:?}", other)}),
    }
}

fn wire_to_intent(v: &Value) -> Option<Intent> {
    match v.get("type")?.as_str()? {
        "attack" => Some(Intent::Attack {
            target_id: v
                .get("targetID")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            troops: v.get("troops").and_then(|x| x.as_f64()),
        }),
        "boat" => Some(Intent::Boat {
            troops: v.get("troops")?.as_f64()?,
            dst: v.get("dst")?.as_u64()? as u32,
        }),
        "spawn" => Some(Intent::Spawn {
            tile: v.get("tile")?.as_u64()? as u32,
        }),
        "build_unit" => Some(Intent::BuildUnit {
            unit: v.get("unit")?.as_str()?.to_string(),
            tile: v.get("tile")?.as_u64()? as u32,
            amount: v.get("amount").and_then(|x| x.as_u64()).map(|x| x as u32),
        }),
        _ => None,
    }
}
