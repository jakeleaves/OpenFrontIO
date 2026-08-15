/**
 * TypeScript oracle — dumps every-10-tick hashes + compact snapshots.
 *
 * Usage:
 *   npx tsx ai/parity/oracle.ts --map plains --ticks 200 --out ai/fixtures/hashes/plains_economy.json
 *
 * Reads maps from tests/testdata/maps/<name>/ by default.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
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
import { Executor } from "../../src/core/execution/ExecutionManager";
import { GameRunner } from "../../src/core/GameRunner";
import { GameUpdateType, HashUpdate } from "../../src/core/game/GameUpdates";
import { PseudoRandom } from "../../src/core/PseudoRandom";
import { simpleHash } from "../../src/core/Util";
import { GameConfig } from "../../src/core/Schemas";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "../..");

interface Options {
  map: string;
  ticks: number;
  out: string;
  scenario: "empty" | "economy" | "attack_tn";
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    map: "plains",
    ticks: 200,
    out: "",
    scenario: "economy",
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "--map":
        opts.map = next();
        break;
      case "--ticks":
        opts.ticks = parseInt(next(), 10);
        break;
      case "--out":
        opts.out = next();
        break;
      case "--scenario":
        opts.scenario = next() as Options["scenario"];
        break;
    }
  }
  if (!opts.out) {
    opts.out = path.join(
      ROOT,
      "ai/fixtures/hashes",
      `${opts.map}_${opts.scenario}.json`,
    );
  }
  return opts;
}

async function loadMaps(mapName: string) {
  const dir = path.join(ROOT, "tests/testdata/maps", mapName);
  const manifest = JSON.parse(
    fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
  ) as MapManifest;
  const mapBin = fs.readFileSync(path.join(dir, "map.bin"));
  const miniBin = fs.readFileSync(path.join(dir, "map4x.bin"));
  const gameMap = await genTerrainFromBin(manifest.map, mapBin);
  const miniGameMap = await genTerrainFromBin(manifest.map4x, miniBin);
  return { gameMap, miniGameMap, manifest };
}

async function main() {
  console.debug = () => {};
  const opts = parseArgs(process.argv.slice(2));
  const { gameMap, miniGameMap } = await loadMaps(opts.map);

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
  const gameID = "oracle-game";
  const random = new PseudoRandom(simpleHash(gameID));

  const humans = [
    new PlayerInfo("Agent", PlayerType.Human, "client0", random.nextID()),
  ];
  if (opts.scenario !== "empty") {
    // keep single human for economy / attack
  }

  const game = createGame(humans, [], gameMap, miniGameMap, config);
  const hashes: { tick: number; hash: number }[] = [];
  const snapshots: unknown[] = [];

  const runner = new GameRunner(
    game,
    new Executor(game, gameID, "client0"),
    (gu) => {
      if ("errMsg" in gu) {
        console.error(gu.errMsg);
        return;
      }
      for (const hu of (gu.updates[GameUpdateType.Hash] ?? []) as HashUpdate[]) {
        hashes.push({ tick: hu.tick, hash: hu.hash });
      }
    },
  );
  runner.init();

  // End spawn + place player for economy/attack scenarios
  if (opts.scenario !== "empty") {
    game.endSpawnPhase();
    const mid = game.ref(Math.floor(game.width() / 2), Math.floor(game.height() / 2));
    // Use spawn intent via turn
    runner.addTurn({
      turnNumber: 0,
      intents: [
        {
          type: "spawn",
          clientID: "client0",
          tile: mid,
        },
      ],
    });
    runner.executeNextTick();
  }

  for (let t = 1; t <= opts.ticks; t++) {
    const intents: any[] = [];
    if (opts.scenario === "attack_tn" && t === 5) {
      intents.push({
        type: "attack",
        clientID: "client0",
        targetID: null,
        troops: 10000,
      });
    }
    runner.addTurn({ turnNumber: t, intents });
    if (!runner.executeNextTick()) break;

    if (game.ticks() % 10 === 0 || t === opts.ticks) {
      const p = game.player(humans[0].id);
      snapshots.push({
        tick: game.ticks(),
        hash: hashes.length ? hashes[hashes.length - 1]?.hash : null,
        gold: Number(p.gold()),
        troops: p.troops(),
        tiles: p.numTilesOwned(),
      });
    }
  }

  const out = {
    map: opts.map,
    scenario: opts.scenario,
    gameID,
    ticks: opts.ticks,
    hashes,
    snapshots,
    final: snapshots[snapshots.length - 1] ?? null,
  };
  fs.mkdirSync(path.dirname(opts.out), { recursive: true });
  fs.writeFileSync(opts.out, JSON.stringify(out, null, 2));
  console.log(`Wrote ${opts.out} (${hashes.length} hash checkpoints)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
