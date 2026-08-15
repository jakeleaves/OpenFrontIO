/**
 * Simpler oracle that mirrors tests/util/Setup.ts — direct Game tick,
 * SpawnExecution for placement, dumps gold/troops/tiles/hash.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { AttackExecution } from "../../src/core/execution/AttackExecution";
import { SpawnExecution } from "../../src/core/execution/SpawnExecution";
import { WinCheckExecution } from "../../src/core/execution/WinCheckExecution";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "../../src/core/game/Game";
import { createGame } from "../../src/core/game/GameImpl";
import {
  genTerrainFromBin,
  MapManifest,
} from "../../src/core/game/TerrainMapLoader";
import { Config } from "../../src/core/configuration/Config";
import { GameUpdateType, HashUpdate } from "../../src/core/game/GameUpdates";
import { GameConfig } from "../../src/core/Schemas";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

async function main() {
  console.debug = () => {};
  const mapName = process.argv[2] ?? "plains";
  const ticks = parseInt(process.argv[3] ?? "200", 10);
  const scenario = process.argv[4] ?? "economy";
  const out =
    process.argv[5] ??
    path.join(ROOT, `ai/fixtures/hashes/${mapName}_${scenario}.json`);

  const dir = path.join(ROOT, "tests/testdata/maps", mapName);
  const manifest = JSON.parse(
    fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
  ) as MapManifest;
  const gameMap = await genTerrainFromBin(
    manifest.map,
    fs.readFileSync(path.join(dir, "map.bin")),
  );
  const miniGameMap = await genTerrainFromBin(
    manifest.map4x,
    fs.readFileSync(path.join(dir, "map4x.bin")),
  );

  const gameConfig: GameConfig = {
    gameMap: GameMapType.Asia,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };
  const config = new Config(gameConfig, null, false);
  const info = new PlayerInfo("Agent", PlayerType.Human, null, "human0");
  const game = createGame([info], [], gameMap, miniGameMap, config);
  game.endSpawnPhase();

  const spawnTile = game.ref(
    Math.floor(game.width() / 2),
    Math.floor(game.height() / 2),
  );
  game.addExecution(new SpawnExecution("oracle-game", info, spawnTile));
  game.addExecution(new WinCheckExecution());
  // SpawnExecution adds PlayerExecution
  game.executeNextTick();
  game.executeNextTick();

  const player = game.player("human0");

  const hashes: { tick: number; hash: number }[] = [];
  const snapshots: unknown[] = [];

  for (let t = 0; t < ticks; t++) {
    if (scenario === "attack_tn" && t === 5) {
      game.addExecution(new AttackExecution(10000, player, game.terraNullius().id()));
    }
    const updates = game.executeNextTick();
    const hashUpdates = (updates[GameUpdateType.Hash] ?? []) as HashUpdate[];
    for (const hu of hashUpdates) {
      hashes.push({ tick: hu.tick, hash: hu.hash });
    }
    if (game.ticks() % 50 === 0 || t === ticks - 1) {
      snapshots.push({
        tick: game.ticks(),
        gold: Number(player.gold()),
        troops: player.troops(),
        tiles: player.numTilesOwned(),
        hash: hashes.length ? hashes[hashes.length - 1].hash : null,
      });
    }
  }

  const result = {
    map: mapName,
    scenario,
    ticks,
    hashes,
    snapshots,
    final: {
      tick: game.ticks(),
      gold: Number(player.gold()),
      troops: player.troops(),
      tiles: player.numTilesOwned(),
    },
  };
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result.final, null, 2));
  console.log(`Wrote ${out} (${hashes.length} hashes)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
