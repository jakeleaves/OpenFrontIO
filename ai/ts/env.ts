/**
 * Single-env wrapper around src/core: 1 Human + 1 Impossible Nation.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../src/core/configuration/Config";
import { Executor } from "../../src/core/execution/ExecutionManager";
import { NationExecution } from "../../src/core/execution/NationExecution";
import { SpawnExecution } from "../../src/core/execution/SpawnExecution";
import { WinCheckExecution } from "../../src/core/execution/WinCheckExecution";
import {
  Cell,
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Nation,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../src/core/game/Game";
import { createGame } from "../../src/core/game/GameImpl";
import {
  genTerrainFromBin,
  MapManifest,
} from "../../src/core/game/TerrainMapLoader";
import { GameRunner } from "../../src/core/GameRunner";
import { Intent, StampedIntent } from "../../src/core/Schemas";
import { decodeIntent, legalMask } from "./actions";
import { encodeObservation, obsToJson } from "./obs";
import {
  ActionMask,
  FactorizedAction,
  Observation,
  StepInfo,
  noopAction,
} from "./types";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

export type EnvOptions = {
  mapName?: string;
  mapsDir?: string;
  difficulty?: Difficulty;
  macroStride?: number;
  maxTicks?: number;
  seed?: number;
  /** If true, auto-spawn both players and end spawn phase on reset. */
  autoSpawn?: boolean;
};

export class GameEnv {
  private game!: Game;
  private runner!: GameRunner;
  private agent!: Player;
  private nation!: Player;
  private turn = 0;
  private done = false;
  private prevTiles = 0;
  private prevTroopDiff = 0;
  private prevGold = 0n;
  private boatsBefore = 0;
  private gameID = "ai-env";
  private readonly opts: Required<EnvOptions>;

  constructor(opts: EnvOptions = {}) {
    this.opts = {
      mapName: opts.mapName ?? "plains",
      mapsDir:
        opts.mapsDir ?? path.join(ROOT, "tests/testdata/maps"),
      difficulty: opts.difficulty ?? Difficulty.Impossible,
      macroStride: opts.macroStride ?? 20,
      maxTicks: opts.maxTicks ?? 20_000,
      seed: opts.seed ?? 1,
      autoSpawn: opts.autoSpawn ?? true,
    };
  }

  async reset(seed?: number): Promise<{
    obs: ReturnType<typeof obsToJson>;
    mask: ActionMask;
    info: StepInfo;
  }> {
    if (seed !== undefined) this.opts.seed = seed;
    console.debug = () => {};
    // Silence GameImpl timing logs that would break JSON-RPC on stdout.
    const origLog = console.log;
    console.log = () => {};

    try {
    const dir = path.join(this.opts.mapsDir, this.opts.mapName);
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

    this.gameID = `ai-env-${this.opts.seed}`;
    const humanInfo = new PlayerInfo(
      "Agent",
      PlayerType.Human,
      "agent-client",
      "agent",
    );
    // Opposite corners for 1v1
    const nx = Math.floor((manifest.map.width * 3) / 4);
    const ny = Math.floor((manifest.map.height * 3) / 4);
    const nationInfo = new PlayerInfo(
      "Nation",
      PlayerType.Nation,
      null,
      "nation",
    );
    const nationObj = new Nation(new Cell(nx, ny), nationInfo);

    const config = new Config(
      {
        gameMap: GameMapType.Asia,
        gameMapSize: GameMapSize.Normal,
        gameMode: GameMode.FFA,
        gameType: GameType.Singleplayer,
        difficulty: this.opts.difficulty,
        nations: 1,
        donateGold: false,
        donateTroops: false,
        bots: 0,
        infiniteGold: false,
        infiniteTroops: false,
        instantBuild: false,
        randomSpawn: false,
      },
      null,
      false,
    );

    this.game = createGame(
      [humanInfo],
      [nationObj],
      gameMap,
      miniGameMap,
      config,
    );

    this.runner = new GameRunner(
      this.game,
      new Executor(this.game, this.gameID, "agent-client"),
      () => {},
    );
    // Don't call runner.init() — it would spawn playlist tribes/nations again.
    // Wire Nation + win check ourselves.
    this.game.addExecution(new NationExecution(this.gameID, nationObj));
    this.game.addExecution(new WinCheckExecution());

    this.agent = this.game.player("agent");
    this.nation = this.game.player("nation");
    this.turn = 0;
    this.done = false;

    if (this.opts.autoSpawn) {
      this.game.endSpawnPhase();
      const ax = Math.floor(manifest.map.width / 4);
      const ay = Math.floor(manifest.map.height / 4);
      const aTile = this.findLandNear(ax, ay);
      const nTile = this.findLandNear(nx, ny);
      this.game.addExecution(
        new SpawnExecution(this.gameID, humanInfo, aTile),
        new SpawnExecution(this.gameID, nationInfo, nTile),
      );
      this.game.executeNextTick();
      this.game.executeNextTick();
    }

    this.prevTiles = this.agent.numTilesOwned();
    this.prevTroopDiff = this.agent.troops() - this.nation.troops();
    this.prevGold = this.agent.gold();
    this.boatsBefore = this.agent.units(UnitType.TransportShip).length;

    return {
      obs: obsToJson(this.observe()),
      mask: legalMask(this.game, this.agent),
      info: this.info(0),
    };
    } finally {
      console.log = origLog;
    }
  }

  private findLandNear(x: number, y: number) {
    const w = this.game.width();
    const h = this.game.height();
    for (let r = 0; r < Math.max(w, h); r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const t = this.game.ref(xx, yy);
          if (this.game.isLand(t) && !this.game.isImpassable(t)) return t;
        }
      }
    }
    return this.game.ref(
      Math.floor(w / 2),
      Math.floor(h / 2),
    );
  }

  observe(): Observation {
    return encodeObservation(this.game, this.agent);
  }

  mask(): ActionMask {
    return legalMask(this.game, this.agent);
  }

  step(action: FactorizedAction = noopAction()): {
    obs: ReturnType<typeof obsToJson>;
    mask: ActionMask;
    info: StepInfo;
  } {
    if (this.done) {
      return {
        obs: obsToJson(this.observe()),
        mask: this.mask(),
        info: this.info(0),
      };
    }

    const boatsBefore = this.agent.units(UnitType.TransportShip).length;

    const intent = decodeIntent(this.game, this.agent, action);
    if (intent) {
      this.applyAgentIntent(intent);
    }

    for (let i = 0; i < this.opts.macroStride; i++) {
      // Empty turn so NationExecution ticks; agent intent already applied
      this.runner.addTurn({ turnNumber: this.turn++, intents: [] });
      if (!this.runner.executeNextTick()) {
        this.game.executeNextTick();
      }
      if (this.game.getWinner() !== null) break;
      if (this.game.ticks() >= this.opts.maxTicks) break;
    }

    const boatsAfter = this.agent.units(UnitType.TransportShip).length;
    const boatSunk = boatsAfter < boatsBefore;
    const reward = this.shapedReward(boatSunk);

    this.prevTiles = this.agent.numTilesOwned();
    this.prevTroopDiff = this.agent.troops() - this.nation.troops();
    this.prevGold = this.agent.gold();

    const winner = this.game.getWinner();
    if (
      winner !== null ||
      !this.agent.isAlive() ||
      this.game.ticks() >= this.opts.maxTicks
    ) {
      this.done = true;
    }

    return {
      obs: obsToJson(this.observe()),
      mask: this.mask(),
      info: this.info(reward),
    };
  }

  private applyAgentIntent(intent: Intent) {
    const stamped: StampedIntent = {
      ...intent,
      clientID: "agent-client",
    } as StampedIntent;
    this.runner.addTurn({
      turnNumber: this.turn++,
      intents: [stamped],
    });
    this.runner.executeNextTick();
  }

  private shapedReward(boatSunk: boolean): number {
    const lambdaW = 10;
    const lambdaN = 1;
    const lambdaT = 0.3;
    const lambdaG = 0.05;
    const lambdaS = 0.5;

    const maxT = Math.max(1, this.game.config().maxTroops(this.agent));
    let r = 0;
    r += lambdaN * (this.agent.numTilesOwned() - this.prevTiles);
    r +=
      lambdaT *
      (this.agent.troops() - this.nation.troops() - this.prevTroopDiff) /
      maxT;
    const goldNow = this.agent.gold();
    r +=
      lambdaG *
      Number(goldNow - this.prevGold) /
      (1 + Number(goldNow));
    if (boatSunk) r -= lambdaS;

    const winner = this.game.getWinner();
    if (winner !== null) {
      const won =
        typeof winner === "object" &&
        "id" in winner &&
        (winner as Player).id?.() === this.agent.id();
      // Team winners are strings; FFA winner is Player
      const wonPlayer =
        winner === this.agent ||
        (typeof (winner as Player).id === "function" &&
          (winner as Player).id() === "agent");
      r += wonPlayer || won ? lambdaW : -lambdaW;
    } else if (!this.agent.isAlive()) {
      r -= lambdaW;
    }
    return r;
  }

  private info(reward: number): StepInfo {
    const winner = this.game.getWinner();
    let winnerId: string | null = null;
    if (winner && typeof (winner as Player).id === "function") {
      winnerId = (winner as Player).id();
    }
    return {
      tick: this.game.ticks(),
      done: this.done,
      reward,
      winner: winnerId,
      agentTiles: this.agent.numTilesOwned(),
      agentTroops: this.agent.troops(),
      agentGold: Number(this.agent.gold()),
      nationTiles: this.nation?.numTilesOwned?.() ?? 0,
    };
  }

  /** Record Nation-side attack intents for BC by observing outgoing attacks deltas — simplified: scripted expand demos use agent actions that mirror Nation heuristics. */
  snapshot() {
    return {
      tick: this.game.ticks(),
      agent: {
        tiles: this.agent.numTilesOwned(),
        troops: this.agent.troops(),
        gold: Number(this.agent.gold()),
      },
      nation: {
        tiles: this.nation.numTilesOwned(),
        troops: this.nation.troops(),
        gold: Number(this.nation.gold()),
      },
      winner: this.info(0).winner,
    };
  }
}
