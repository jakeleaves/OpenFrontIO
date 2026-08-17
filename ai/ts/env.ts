/**
 * World Solo FFA env: 1 Human vs all manifest nations at Impossible.
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
  Nation as ManifestNation,
  MapManifest,
} from "../../src/core/game/TerrainMapLoader";
import { Intent, StampedIntent } from "../../src/core/Schemas";
import {
  decodeIntent,
  DecodeIntentOptions,
  legalMask,
  LegalMaskOptions,
} from "./actions";
import { growthEfficiency } from "./growth";
import { createObservationBuffers, encodeObservation } from "./obs";
import {
  ActionMask,
  FactorizedAction,
  MAX_NATIONS,
  noopAction,
  Observation,
  StepInfo,
} from "./types";

export type RewardConfig = {
  lambdaW: number;
  lambdaN: number;
  lambdaT: number;
  lambdaG: number;
  lambdaS: number;
  lambdaNationTiles: number;
  /** Reward for non-city structures (SAM / MissileSilo). */
  lambdaStruct: number;
  /** One-time reward per newly completed City (count increase only). */
  lambdaCity: number;
  /** Penalty scale when troop ratio falls below reserveRatio. */
  lambdaReserve: number;
  reserveRatio: number;
  /** Reward for operating near peak troopIncreaseRate. */
  lambdaGrowth: number;
  /** Soft reward for improving placement (lower rank = better). */
  lambdaPlacement: number;
  /** Potential-based reward for tile lead vs the strongest enemy. */
  lambdaTileLead: number;
  /** One-time reward when a transport ship is launched. */
  lambdaBoatLaunch: number;
};

const DEFAULT_REWARD_CONFIG: RewardConfig = {
  lambdaW: 300,
  // A typical macro attack gains ~50-100 tiles. Scaling by land then by 20
  // makes useful expansion worth ~0.1-0.2 instead of effectively zero.
  lambdaN: 20,
  lambdaT: 0.5,
  lambdaG: 0.05,
  lambdaS: 0.5,
  lambdaNationTiles: 0.1,
  lambdaStruct: 2,
  // Bounded below terminal win (±300). Fires only on city-count increase.
  lambdaCity: 12,
  lambdaReserve: 8,
  reserveRatio: 0.35,
  // Potential-based: reward changes in efficiency, not time spent idling at
  // the optimum. This prevents the agent farming a positive reward via NOOP.
  lambdaGrowth: 1,
  // One rank improvement must not be drowned by hundreds of land-farm steps.
  lambdaPlacement: 20,
  // Same scale as lambdaN: catching the leader is as valuable as raw expansion.
  lambdaTileLead: 20,
  // Small nudge to try boats; far below terminal win.
  lambdaBoatLaunch: 1,
};

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

/** Cached raw map bytes — terrain is immutable; each reset builds fresh GameMapImpl state. */
type CachedMapAssets = {
  manifest: MapManifest;
  mapBin: Uint8Array;
  map4xBin: Uint8Array;
};

const MAP_ASSET_CACHE = new Map<string, CachedMapAssets>();

async function loadMapAssets(
  mapsDir: string,
  mapName: string,
): Promise<CachedMapAssets> {
  const key = `${mapsDir}::${mapName}`;
  const hit = MAP_ASSET_CACHE.get(key);
  if (hit) return hit;
  const dir = path.join(mapsDir, mapName);
  const manifest = JSON.parse(
    fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
  ) as MapManifest;
  const mapBin = new Uint8Array(fs.readFileSync(path.join(dir, "map.bin")));
  const map4xBin = new Uint8Array(fs.readFileSync(path.join(dir, "map4x.bin")));
  const assets = { manifest, mapBin, map4xBin };
  MAP_ASSET_CACHE.set(key, assets);
  return assets;
}

/** Clear cached map assets (tests). */
export function clearMapAssetCache(): void {
  MAP_ASSET_CACHE.clear();
}

export type StepTiming = {
  simMs: number;
  obsMs: number;
  maskMs: number;
  serializeMs: number;
};

export type EnvOptions = {
  mapName?: string;
  mapsDir?: string;
  difficulty?: Difficulty;
  /** Number of manifest nations to spawn. */
  nations?: number;
  /** Number of bot tribes to spawn. Nations + bots must fit target slots. */
  bots?: number;
  macroStride?: number;
  maxTicks?: number;
  seed?: number;
  /** If true, auto-spawn both players and end spawn phase on reset. */
  autoSpawn?: boolean;
  reward?: Partial<RewardConfig>;
};

export class GameEnv {
  private game!: Game;
  private executor!: Executor;
  private agent!: Player;
  /** Fixed nation player IDs for target slots 1..N (manifest order). */
  private opponentIds: string[] = [];
  private turn = 0;
  private done = false;
  private prevTiles = 0;
  private prevEnemyTilesTotal = 0;
  private prevStructCount = 0;
  private prevCityCount = 0;
  private prevTroopDiff = 0;
  private prevGold = 0n;
  private prevGrowthEfficiency = 0;
  private prevPlacement = 1;
  private prevTileLead = 0;
  private boatsBefore = 0;
  private gameID = "ai-env";
  /** Reused obs tensors — ~524 KB, zeroed each encode. */
  private readonly obsBuf = createObservationBuffers();
  private readonly opts: Omit<Required<EnvOptions>, "reward"> & {
    reward: RewardConfig;
  };

  constructor(opts: EnvOptions = {}) {
    this.opts = {
      mapName: opts.mapName ?? "world",
      mapsDir: opts.mapsDir ?? path.join(ROOT, "resources/maps"),
      difficulty: opts.difficulty ?? Difficulty.Impossible,
      nations: opts.nations ?? MAX_NATIONS,
      bots: opts.bots ?? 0,
      macroStride: opts.macroStride ?? 10,
      maxTicks: opts.maxTicks ?? 30_000,
      seed: opts.seed ?? 1,
      autoSpawn: opts.autoSpawn ?? true,
      reward: { ...DEFAULT_REWARD_CONFIG, ...opts.reward },
    };
    if (!Number.isInteger(this.opts.nations) || this.opts.nations < 1) {
      throw new Error("nations must be a positive integer");
    }
    if (!Number.isInteger(this.opts.bots) || this.opts.bots < 0) {
      throw new Error("bots must be a non-negative integer");
    }
    if (this.opts.nations + this.opts.bots > MAX_NATIONS) {
      throw new Error(`nations + bots must be <= ${MAX_NATIONS} target slots`);
    }
  }

  async reset(
    seed?: number,
    opts: { maskOptions?: LegalMaskOptions } = {},
  ): Promise<{
    obs: Observation;
    mask: ActionMask;
    info: StepInfo;
    timing?: StepTiming;
  }> {
    if (seed !== undefined) this.opts.seed = seed;
    console.debug = () => {};
    // Silence GameImpl timing logs that would break JSON-RPC on stdout.
    const origLog = console.log;
    console.log = () => {};

    try {
      const assets = await loadMapAssets(this.opts.mapsDir, this.opts.mapName);
      const manifest = assets.manifest;
      // Fresh GameMapImpl each reset (mutable ownership state); terrain bytes shared.
      const gameMap = await genTerrainFromBin(manifest.map, assets.mapBin);
      const miniGameMap = await genTerrainFromBin(
        manifest.map4x,
        assets.map4xBin,
      );

      this.gameID = `ai-env-${this.opts.seed}`;
      const humanInfo = new PlayerInfo(
        "Agent",
        PlayerType.Human,
        "agent-client",
        "agent",
      );

      const manifestNations = (manifest.nations ?? []).slice(
        0,
        Math.min(this.opts.nations, MAX_NATIONS),
      );
      const nationObjs = manifestNations.map((n, i) =>
        this.manifestToNation(n, i),
      );
      const nationIds = nationObjs.map((n) => n.playerInfo.id);

      const config = new Config(
        {
          gameMap: GameMapType.World,
          gameMapSize: GameMapSize.Normal,
          gameMode: GameMode.FFA,
          gameType: GameType.Singleplayer,
          difficulty: this.opts.difficulty,
          nations: "default",
          donateGold: false,
          donateTroops: false,
          bots: this.opts.bots,
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
        nationObjs,
        gameMap,
        miniGameMap,
        config,
      );
      this.game.setHeadless(true);

      this.executor = new Executor(this.game, this.gameID, "agent-client");
      // Headless path: tick GameImpl directly — no GameRunner placeName/wire work.

      this.agent = this.game.player("agent");
      this.turn = 0;
      this.done = false;

      if (this.opts.autoSpawn) {
        this.game.endSpawnPhase();
        const aTile = this.findAgentSpawn(nationObjs);
        this.game.addExecution(
          new SpawnExecution(this.gameID, humanInfo, aTile),
        );
        for (const nation of nationObjs) {
          const cell = nation.spawnCell;
          const nTile =
            cell !== undefined
              ? this.findLandNear(cell.x, cell.y)
              : this.findLandNear(
                  Math.floor(manifest.map.width / 2),
                  Math.floor(manifest.map.height / 2),
                );
          this.game.addExecution(
            new SpawnExecution(this.gameID, nation.playerInfo, nTile),
          );
        }
        if (this.opts.bots > 0) {
          this.game.addExecution(...this.executor.spawnTribes(this.opts.bots));
        }
        this.game.executeNextTick();
        this.game.executeNextTick();
      }

      const botIds = this.game
        .players()
        .filter((p) => p !== this.agent && p.type() === PlayerType.Bot)
        .map((p) => p.id())
        .sort();
      if (botIds.length !== this.opts.bots) {
        throw new Error(
          `expected ${this.opts.bots} spawned bots, got ${botIds.length}`,
        );
      }
      // Fixed policy target slots: nations first, then deterministic bot IDs.
      this.opponentIds = [...nationIds, ...botIds];

      // NationExecution.tick() deactivates if the player has 0 tiles. Add after
      // spawn has landed so Impossible Nations expand.
      for (const nation of nationObjs) {
        this.game.addExecution(new NationExecution(this.gameID, nation));
      }
      this.game.addExecution(new WinCheckExecution());

      const agg = this.aggregateEnemies();
      this.prevTiles = this.agent.numTilesOwned();
      this.prevEnemyTilesTotal = agg.enemyTilesTotal;
      this.prevStructCount = this.structCount(this.agent);
      this.prevCityCount = this.cityCount(this.agent);
      this.prevTroopDiff = this.agent.troops() - agg.enemyTroopsTotal;
      this.prevGold = this.agent.gold();
      this.prevPlacement = this.placement();
      this.prevTileLead =
        this.agent.numTilesOwned() - agg.strongestEnemyTiles;
      this.prevGrowthEfficiency = growthEfficiency(
        this.agent.troops(),
        Math.max(1, this.game.config().maxTroops(this.agent)),
      ).efficiency;
      this.boatsBefore = this.agent.units(UnitType.TransportShip).length;

      const tObs0 = performance.now();
      const obs = this.observe();
      const obsMs = performance.now() - tObs0;
      const tMask0 = performance.now();
      const mask = legalMask(
        this.game,
        this.agent,
        this.opponentIds,
        opts.maskOptions,
      );
      const maskMs = performance.now() - tMask0;
      const serializeMs = 0;

      return {
        obs,
        mask,
        info: this.info(0),
        timing: { simMs: 0, obsMs, maskMs, serializeMs },
      };
    } finally {
      console.log = origLog;
    }
  }

  private manifestToNation(n: ManifestNation, index: number): Nation {
    const id = `nation-${index}`;
    const info = new PlayerInfo(n.name, PlayerType.Nation, null, id);
    const cell =
      n.coordinates !== undefined
        ? new Cell(n.coordinates[0], n.coordinates[1])
        : undefined;
    return new Nation(cell, info);
  }

  /**
   * Seeded spawn: legal land maximizing distance from nation spawn cells.
   */
  private findAgentSpawn(nations: Nation[]): number {
    const w = this.game.width();
    const h = this.game.height();
    const spawns = nations
      .map((n) => n.spawnCell)
      .filter((c): c is Cell => c !== undefined);
    const minNationDist = 40;

    let rng = (this.opts.seed * 1103515245 + 12345) >>> 0;
    const next = () => {
      rng = (rng * 1103515245 + 12345) >>> 0;
      return rng / 0x100000000;
    };

    let bestTile = this.findLandNear(Math.floor(w / 2), Math.floor(h / 2));
    let bestScore = -Infinity;
    const trials = 400;
    for (let i = 0; i < trials; i++) {
      const x = Math.floor(next() * w);
      const y = Math.floor(next() * h);
      const t = this.game.ref(x, y);
      if (!this.game.isLand(t) || this.game.isImpassable(t)) continue;
      let minD = Infinity;
      for (const c of spawns) {
        const dx = x - c.x;
        const dy = y - c.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < minD) minD = d;
      }
      if (minD < minNationDist) continue;
      if (minD > bestScore) {
        bestScore = minD;
        bestTile = t;
      }
    }
    return bestTile;
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
    return this.game.ref(Math.floor(w / 2), Math.floor(h / 2));
  }

  private aliveOpponents(): Player[] {
    const out: Player[] = [];
    for (const id of this.opponentIds) {
      if (!this.game.hasPlayer(id)) continue;
      const p = this.game.player(id);
      if (p.isAlive() && p.numTilesOwned() > 0) out.push(p);
    }
    return out;
  }

  private aggregateEnemies(): {
    opponentsAlive: number;
    enemyTilesTotal: number;
    enemyTroopsTotal: number;
    strongestEnemyTiles: number;
    strongestEnemyTroops: number;
  } {
    const alive = this.aliveOpponents();
    let enemyTilesTotal = 0;
    let enemyTroopsTotal = 0;
    let strongestEnemyTiles = 0;
    let strongestEnemyTroops = 0;
    for (const p of alive) {
      const tiles = p.numTilesOwned();
      const troops = p.troops();
      enemyTilesTotal += tiles;
      enemyTroopsTotal += troops;
      if (tiles > strongestEnemyTiles) strongestEnemyTiles = tiles;
      if (troops > strongestEnemyTroops) strongestEnemyTroops = troops;
    }
    return {
      opponentsAlive: alive.length,
      enemyTilesTotal,
      enemyTroopsTotal,
      strongestEnemyTiles,
      strongestEnemyTroops,
    };
  }

  /** 1 = most land among alive non-bot players. */
  private placement(): number {
    const agentTiles = this.agent.numTilesOwned();
    let better = 0;
    for (const p of this.game.players()) {
      if (p === this.agent) continue;
      if (!p.isAlive()) continue;
      if (p.numTilesOwned() > agentTiles) better++;
    }
    return better + 1;
  }

  observe(): Observation {
    return encodeObservation(this.game, this.agent, this.obsBuf);
  }

  mask(opts: LegalMaskOptions = {}): ActionMask {
    return legalMask(this.game, this.agent, this.opponentIds, opts);
  }

  step(
    action: FactorizedAction = noopAction(),
    opts: {
      encodeObs?: boolean;
      encodeMask?: boolean;
      maskOptions?: LegalMaskOptions;
      decodeOptions?: DecodeIntentOptions;
    } = {},
  ): {
    obs: Observation;
    mask: ActionMask;
    info: StepInfo;
    timing: StepTiming;
  } {
    // WinCheckExecution / GameImpl print to console.log; keep stdout clean for RPC/viz.
    const origLog = console.log;
    console.log = () => {};
    try {
      return this.stepInner(action, opts);
    } finally {
      console.log = origLog;
    }
  }

  private stepInner(
    action: FactorizedAction,
    opts: {
      encodeObs?: boolean;
      encodeMask?: boolean;
      maskOptions?: LegalMaskOptions;
      decodeOptions?: DecodeIntentOptions;
    } = {},
  ): {
    obs: Observation;
    mask: ActionMask;
    info: StepInfo;
    timing: StepTiming;
  } {
    const encodeObs = opts.encodeObs !== false;
    const encodeMask = opts.encodeMask !== false;

    if (this.done) {
      const tObs0 = performance.now();
      const obs = encodeObs ? this.observe() : this.obsBuf;
      const obsMs = performance.now() - tObs0;
      const tMask0 = performance.now();
      const mask = encodeMask
        ? this.mask(opts.maskOptions)
        : {
            actionType: [] as boolean[],
            targetPlayer: [] as boolean[],
            cell: [] as boolean[],
            troopFrac: [] as boolean[],
            buildType: [] as boolean[],
          };
      const maskMs = performance.now() - tMask0;
      return {
        obs,
        mask,
        info: this.info(0),
        timing: { simMs: 0, obsMs, maskMs, serializeMs: 0 },
      };
    }

    const boatsBefore = this.agent.units(UnitType.TransportShip).length;

    const intent = decodeIntent(
      this.game,
      this.agent,
      action,
      this.opponentIds,
      opts.decodeOptions,
    );
    if (intent) {
      this.applyAgentIntent(intent);
    }

    const tSim0 = performance.now();
    for (let i = 0; i < this.opts.macroStride; i++) {
      // NationExecution ticks on empty turns; agent intent already applied.
      this.turn++;
      this.game.executeNextTick();
      if (this.game.getWinner() !== null) break;
      if (this.game.ticks() >= this.opts.maxTicks) break;
    }
    const simMs = performance.now() - tSim0;

    const boatsAfter = this.agent.units(UnitType.TransportShip).length;
    const boatSunk = boatsAfter < boatsBefore;
    const boatLaunched = boatsAfter > boatsBefore;
    const reward = this.shapedReward(boatSunk, boatLaunched);

    const agg = this.aggregateEnemies();
    this.prevTiles = this.agent.numTilesOwned();
    this.prevEnemyTilesTotal = agg.enemyTilesTotal;
    this.prevStructCount = this.structCount(this.agent);
    this.prevCityCount = this.cityCount(this.agent);
    this.prevTroopDiff = this.agent.troops() - agg.enemyTroopsTotal;
    this.prevGold = this.agent.gold();
    this.prevPlacement = this.placement();
    this.prevTileLead =
      this.agent.numTilesOwned() - agg.strongestEnemyTiles;

    const winner = this.game.getWinner();
    if (
      winner !== null ||
      !this.agent.isAlive() ||
      this.game.ticks() >= this.opts.maxTicks
    ) {
      this.done = true;
    }

    const tObs0 = performance.now();
    const obs = encodeObs ? this.observe() : this.obsBuf;
    const obsMs = performance.now() - tObs0;
    const tMask0 = performance.now();
    const mask = encodeMask
      ? this.mask(opts.maskOptions)
      : {
          actionType: [] as boolean[],
          targetPlayer: [] as boolean[],
          cell: [] as boolean[],
          troopFrac: [] as boolean[],
          buildType: [] as boolean[],
        };
    const maskMs = performance.now() - tMask0;

    return {
      obs,
      mask,
      info: this.info(reward),
      timing: { simMs, obsMs, maskMs, serializeMs: 0 },
    };
  }

  private applyAgentIntent(intent: Intent) {
    const stamped: StampedIntent = {
      ...intent,
      clientID: "agent-client",
    } as StampedIntent;
    this.game.addExecution(this.executor.createExec(stamped));
    this.turn++;
    this.game.executeNextTick();
  }

  private shapedReward(boatSunk: boolean, boatLaunched = false): number {
    const {
      lambdaW,
      lambdaN,
      lambdaT,
      lambdaG,
      lambdaS,
      lambdaNationTiles,
      lambdaStruct,
      lambdaCity,
      lambdaReserve,
      reserveRatio,
      lambdaGrowth,
      lambdaPlacement,
      lambdaTileLead,
      lambdaBoatLaunch,
    } = this.opts.reward;

    const maxT = Math.max(1, this.game.config().maxTroops(this.agent));
    const land = Math.max(1, this.game.numLandTiles());
    const ratio = this.agent.troops() / maxT;
    const agg = this.aggregateEnemies();
    const place = this.placement();
    let r = 0;
    r += (lambdaN * (this.agent.numTilesOwned() - this.prevTiles)) / land;
    // Reward reducing aggregate enemy land (FFA pressure).
    r +=
      lambdaNationTiles *
      Math.max(0, this.prevEnemyTilesTotal - agg.enemyTilesTotal);
    r +=
      (lambdaT *
        (this.agent.troops() - agg.enemyTroopsTotal - this.prevTroopDiff)) /
      maxT;
    const goldNow = this.agent.gold();
    r += (lambdaG * Number(goldNow - this.prevGold)) / (1 + Number(goldNow));
    const citiesNow = this.cityCount(this.agent);
    const dCities = Math.max(0, citiesNow - this.prevCityCount);
    r += lambdaCity * dCities;
    r += lambdaStruct * (this.structCount(this.agent) - this.prevStructCount);
    if (ratio < reserveRatio) {
      r -= lambdaReserve * (reserveRatio - ratio);
    }
    const currentGrowthEfficiency = growthEfficiency(
      this.agent.troops(),
      maxT,
    ).efficiency;
    r += lambdaGrowth * (currentGrowthEfficiency - this.prevGrowthEfficiency);
    this.prevGrowthEfficiency = currentGrowthEfficiency;
    // Lower placement number is better.
    r += lambdaPlacement * (this.prevPlacement - place);
    const tileLead =
      this.agent.numTilesOwned() - agg.strongestEnemyTiles;
    r += (lambdaTileLead * (tileLead - this.prevTileLead)) / land;
    if (boatSunk) r -= lambdaS;
    if (boatLaunched) r += lambdaBoatLaunch;

    const winner = this.game.getWinner();
    if (winner !== null) {
      const won =
        typeof winner === "object" &&
        "id" in winner &&
        (winner as Player).id?.() === this.agent.id();
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

  private cityCount(player: Player): number {
    return player.units(UnitType.City).length;
  }

  private structCount(player: Player): number {
    return (
      player.units(UnitType.SAMLauncher).length +
      player.units(UnitType.MissileSilo).length
    );
  }

  private info(reward: number): StepInfo {
    const winner = this.game.getWinner();
    let winnerId: string | null = null;
    if (winner && typeof (winner as Player).id === "function") {
      winnerId = (winner as Player).id();
    }
    const troopCap = Math.max(1, this.game.config().maxTroops(this.agent));
    const agentTroops = this.agent.troops();
    const growth = growthEfficiency(agentTroops, troopCap);
    const agg = this.aggregateEnemies();
    return {
      tick: this.game.ticks(),
      done: this.done,
      reward,
      winner: winnerId,
      agentTiles: this.agent.numTilesOwned(),
      agentTroops,
      agentGold: Number(this.agent.gold()),
      agentCities: this.cityCount(this.agent),
      nationTiles: agg.strongestEnemyTiles,
      troopCap,
      troopRatio: agentTroops / troopCap,
      growthEfficiency: growth.efficiency,
      optimalGrowthRatio: growth.optimalRatio,
      nationTroops: agg.strongestEnemyTroops,
      opponentsAlive: agg.opponentsAlive,
      enemyTilesTotal: agg.enemyTilesTotal,
      enemyTroopsTotal: agg.enemyTroopsTotal,
      strongestEnemyTiles: agg.strongestEnemyTiles,
      strongestEnemyTroops: agg.strongestEnemyTroops,
      placement: this.placement(),
      agentBoats: this.agent.units(UnitType.TransportShip).length,
      opponentIds: [...this.opponentIds],
    };
  }

  /** Expose game/agent for expert / tests (read-only use). */
  getGame(): Game {
    return this.game;
  }

  getAgent(): Player {
    return this.agent;
  }

  /** @deprecated Prefer getOpponents() — returns strongest alive nation. */
  getNation(): Player | null {
    const alive = this.aliveOpponents().sort(
      (a, b) => b.numTilesOwned() - a.numTilesOwned(),
    );
    return alive[0] ?? null;
  }

  getOpponents(): Player[] {
    return this.aliveOpponents();
  }

  getOpponentIds(): string[] {
    return [...this.opponentIds];
  }

  macroStride(): number {
    return this.opts.macroStride;
  }

  snapshot() {
    const agg = this.aggregateEnemies();
    return {
      tick: this.game.ticks(),
      agent: {
        tiles: this.agent.numTilesOwned(),
        troops: this.agent.troops(),
        gold: Number(this.agent.gold()),
        cities: this.cityCount(this.agent),
        boats: this.agent.units(UnitType.TransportShip).length,
      },
      enemies: {
        alive: agg.opponentsAlive,
        tiles: agg.enemyTilesTotal,
        troops: agg.enemyTroopsTotal,
        strongestTiles: agg.strongestEnemyTiles,
        strongestTroops: agg.strongestEnemyTroops,
      },
      placement: this.placement(),
      winner: this.info(0).winner,
    };
  }

  /**
   * Packed ownership grid for the live visualizer (downsampled on World).
   * cells are charCodes: 0 water, 1 unowned, 2 agent, 3+N = nation slot N.
   * When prevCells matches dimensions, returns sparse deltas if <25% changed.
   */
  mapFrame(
    maxDim = 500,
    prevCells?: string | null,
  ): {
    w: number;
    h: number;
    cells: string;
    deltas: { i: number; c: number }[] | null;
    units: {
      x: number;
      y: number;
      kind: string;
      owner: "agent" | "nation";
      nationIndex?: number;
    }[];
    attacks: { agent: number; nation: number };
    fullW: number;
    fullH: number;
    stride: number;
    nationCount: number;
  } {
    const fullW = this.game.width();
    const fullH = this.game.height();
    const stride = Math.max(
      1,
      Math.ceil(Math.max(fullW, fullH) / Math.max(64, maxDim)),
    );
    const w = Math.ceil(fullW / stride);
    const h = Math.ceil(fullH / stride);
    const agentSid = this.agent.smallID();
    const sidToNation = new Map<number, number>();
    for (let i = 0; i < this.opponentIds.length; i++) {
      const id = this.opponentIds[i];
      if (!this.game.hasPlayer(id)) continue;
      sidToNation.set(this.game.player(id).smallID(), i);
    }
    const chars: string[] = new Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const r = this.game.ref(
          Math.min(fullW - 1, x * stride),
          Math.min(fullH - 1, y * stride),
        );
        let code = 1;
        if (this.game.isWater(r)) {
          code = 0;
        } else {
          const o = this.game.ownerID(r);
          if (o === 0) code = 1;
          else if (o === agentSid) code = 2;
          else {
            const slot = sidToNation.get(o);
            code = slot !== undefined ? 3 + slot : 3;
          }
        }
        chars[y * w + x] = String.fromCharCode(code);
      }
    }
    const cells = chars.join("");
    let deltas: { i: number; c: number }[] | null = null;
    if (prevCells && prevCells.length === cells.length) {
      const changed: { i: number; c: number }[] = [];
      for (let i = 0; i < cells.length; i++) {
        if (cells.charCodeAt(i) !== prevCells.charCodeAt(i)) {
          changed.push({ i, c: cells.charCodeAt(i) });
        }
      }
      if (changed.length < cells.length * 0.25) {
        deltas = changed;
      }
    }
    const units: {
      x: number;
      y: number;
      kind: string;
      owner: "agent" | "nation";
      nationIndex?: number;
    }[] = [];
    for (const u of this.agent.units()) {
      units.push({
        x: Math.floor(this.game.x(u.tile()) / stride),
        y: Math.floor(this.game.y(u.tile()) / stride),
        kind: String(u.type()),
        owner: "agent",
      });
    }
    for (const p of this.aliveOpponents()) {
      const nationIndex = sidToNation.get(p.smallID());
      for (const u of p.units()) {
        units.push({
          x: Math.floor(this.game.x(u.tile()) / stride),
          y: Math.floor(this.game.y(u.tile()) / stride),
          kind: String(u.type()),
          owner: "nation",
          nationIndex,
        });
      }
    }
    const nationAttacks = this.aliveOpponents().reduce(
      (n, p) => n + p.outgoingAttacks().length,
      0,
    );
    return {
      w,
      h,
      cells,
      deltas,
      units,
      attacks: {
        agent: this.agent.outgoingAttacks().length,
        nation: nationAttacks,
      },
      fullW,
      fullH,
      stride,
      nationCount: this.opponentIds.length,
    };
  }
}
