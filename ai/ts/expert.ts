/**
 * Reserve-aware expert for BC demos (Nation-like ratios).
 * expand ~10–20%, reserve ~30–40%, trigger player attacks ~50–60% of maxTroops.
 * World FFA: multi-nation target slots + coastal boats.
 */
import { Game, Player, UnitType } from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
import { canBuildTransportShip } from "../../src/core/game/TransportShipUtils";
import { legalMask, resolveOpponentIds } from "./actions";
import {
  ActionType,
  COARSE_H,
  COARSE_W,
  FactorizedAction,
  TROOP_FRACS,
} from "./types";

export type ExpertStrategy = "default" | "city-economy";

export type ExpertOptions = {
  expandRatio?: number;
  reserveRatio?: number;
  triggerRatio?: number;
  strategy?: ExpertStrategy;
  targetCities?: number;
  opponentIds?: readonly string[];
};

export type ExpertDecision = {
  action: FactorizedAction;
  /** 0=expand, 1=crush, 2=eco, 3=nuke, 4=defend */
  macroGoal: number;
};

const DEFAULTS: Required<Omit<ExpertOptions, "opponentIds">> & {
  opponentIds?: readonly string[];
} = {
  expandRatio: 0.15,
  reserveRatio: 0.35,
  triggerRatio: 0.55,
  strategy: "default",
  targetCities: 3,
  opponentIds: undefined,
};

function tileToCoarse(
  game: Game,
  tile: TileRef,
): { cellX: number; cellY: number } {
  const x = game.x(tile);
  const y = game.y(tile);
  const cellX = Math.min(
    COARSE_W - 1,
    Math.max(0, Math.floor((x / Math.max(1, game.width())) * COARSE_W)),
  );
  const cellY = Math.min(
    COARSE_H - 1,
    Math.max(0, Math.floor((y / Math.max(1, game.height())) * COARSE_H)),
  );
  return { cellX, cellY };
}

function fracIndexForRatio(desired: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < TROOP_FRACS.length; i++) {
    const d = Math.abs(TROOP_FRACS[i] - desired);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function findExpandTile(game: Game, ego: Player): TileRef | null {
  for (const b of ego.borderTiles()) {
    for (const n of game.neighbors(b)) {
      if (game.isLand(n) && !game.hasOwner(n) && !game.isImpassable(n)) {
        return n;
      }
    }
  }
  return null;
}

/** Prefer weakest bordering enemy; returns tile + target slot (1..N). */
function findEnemyTarget(
  game: Game,
  ego: Player,
  opponentIds?: readonly string[],
): { tile: TileRef; targetPlayer: number } | null {
  const slots = resolveOpponentIds(game, ego, opponentIds);
  const idToSlot = new Map(slots.map((id, i) => [id, i + 1]));
  let best: { tile: TileRef; targetPlayer: number; troops: number } | null =
    null;
  for (const b of ego.borderTiles()) {
    for (const n of game.neighbors(b)) {
      if (!game.isLand(n) || !game.hasOwner(n)) continue;
      const owner = game.owner(n);
      if (!owner.isPlayer() || owner === ego || !owner.isAlive()) continue;
      const slot = idToSlot.get(owner.id());
      if (slot === undefined) continue;
      const troops = owner.troops();
      if (best === null || troops < best.troops) {
        best = { tile: n, targetPlayer: slot, troops };
      }
    }
  }
  return best;
}

/** Cap canBuild probes — each call runs a radius-15 BFS + sort. */
const BUILD_PROBE_BUDGET = 64;

function findBuildTile(
  game: Game,
  ego: Player,
  unitType: UnitType = UnitType.City,
): TileRef | null {
  // Prefer interior border-adjacent tiles without materializing all owned tiles
  // (O(territory) copies + canBuild was multi-second at 150k tiles).
  let probes = BUILD_PROBE_BUDGET;
  const borders = ego.borderTiles();
  for (const border of borders) {
    for (const n of game.neighbors(border)) {
      if (!game.isLand(n) || game.owner(n) !== ego) continue;
      if (borders.has(n)) continue;
      if (probes-- <= 0) return null;
      if (ego.canBuild(unitType, n) !== false) return n;
    }
  }
  for (const border of borders) {
    if (probes-- <= 0) return null;
    if (ego.canBuild(unitType, border) !== false) return border;
  }
  return null;
}

function findCoastalBuildTile(game: Game, ego: Player): TileRef | null {
  for (const tile of ego.borderTiles()) {
    for (const neighbor of game.neighbors(tile)) {
      if (game.isWater(neighbor)) return tile;
    }
  }
  return null;
}

/**
 * Boat destination cache + probe budget. canBuildTransportShip runs an unbounded
 * water BFS, so an exhaustive border scan costs seconds per call once the border
 * ring grows. Mirrors BOAT_MASK_REFRESH_TICKS in actions.ts.
 */
const BOAT_DEST_REFRESH_TICKS = 25;
/** Max canBuildTransportShip probes per search, across all three scans. */
const BOAT_PROBE_BUDGET = 400;

type BoatDestCache = {
  egoID: string;
  tickBucket: number;
  dst: TileRef | null;
};
let boatDestCache: BoatDestCache | null = null;

/** Prefer unowned coast, else enemy coast reachable by transport.
 *  Also samples distant TN / overseas enemies (not only the local water ring). */
function findBoatDestination(game: Game, ego: Player): TileRef | null {
  const egoID = ego.id();
  const tickBucket = Math.floor(game.ticks() / BOAT_DEST_REFRESH_TICKS);
  if (
    boatDestCache !== null &&
    boatDestCache.egoID === egoID &&
    boatDestCache.tickBucket === tickBucket
  ) {
    return boatDestCache.dst;
  }
  const dst = computeBoatDestination(game, ego);
  boatDestCache = { egoID, tickBucket, dst };
  return dst;
}

function computeBoatDestination(game: Game, ego: Player): TileRef | null {
  let enemyShore: TileRef | null = null;
  let localTn: TileRef | null = null;
  let probes = BOAT_PROBE_BUDGET;
  localRing: for (const border of ego.borderTiles()) {
    for (const water of game.neighbors(border)) {
      if (!game.isWater(water)) continue;
      for (const shore of game.neighbors(water)) {
        if (!game.isLand(shore) || game.isImpassable(shore)) continue;
        if (probes <= 0) break localRing;
        probes--;
        if (canBuildTransportShip(game, ego, shore) === false) continue;
        if (!game.hasOwner(shore)) {
          if (localTn === null) localTn = shore;
          continue;
        }
        const owner = game.owner(shore);
        if (owner.isPlayer() && owner !== ego && owner.isAlive()) {
          enemyShore = shore;
        }
      }
    }
  }

  // Distant TN continents — useful even when local land expand still exists.
  let overseasTn: TileRef | null = null;
  const stepX = Math.max(1, Math.floor(game.width() / 48));
  const stepY = Math.max(1, Math.floor(game.height() / 24));
  const borders = ego.borderTiles();
  for (let y = 0; y < game.height() && overseasTn === null; y += stepY) {
    for (let x = 0; x < game.width(); x += stepX) {
      const t = game.ref(x, y);
      if (!game.isLand(t) || game.hasOwner(t) || game.isImpassable(t)) continue;
      if (probes <= 0) break;
      probes--;
      if (canBuildTransportShip(game, ego, t) === false) continue;
      // Treat as overseas when not adjacent to ego borders.
      let near = false;
      for (const b of borders) {
        const dx = game.x(t) - game.x(b);
        const dy = game.y(t) - game.y(b);
        if (dx * dx + dy * dy < 40 * 40) {
          near = true;
          break;
        }
      }
      if (!near) {
        overseasTn = t;
        break;
      }
    }
  }

  // Prefer overseas TN (island hop) over local scrap, then local TN, then enemy.
  if (overseasTn !== null) return overseasTn;
  if (localTn !== null) return localTn;
  if (enemyShore !== null) return enemyShore;

  for (const p of game.players()) {
    if (p === ego || !p.isAlive()) continue;
    for (const tile of p.borderTiles()) {
      if (probes <= 0) return null;
      probes--;
      if (canBuildTransportShip(game, ego, tile) !== false) return tile;
    }
  }
  return null;
}

/** True when shore is not in the local coastal ring of ego. */
function isOverseasShore(game: Game, ego: Player, shore: TileRef): boolean {
  for (const border of ego.borderTiles()) {
    for (const water of game.neighbors(border)) {
      if (!game.isWater(water)) continue;
      for (const n of game.neighbors(water)) {
        if (n === shore) return false;
      }
    }
  }
  return true;
}

function actionAt(
  actionType: ActionType,
  tile: TileRef,
  game: Game,
  options: Partial<FactorizedAction> = {},
): FactorizedAction {
  const { cellX, cellY } = tileToCoarse(game, tile);
  return {
    actionType,
    targetPlayer: 0,
    cellX,
    cellY,
    troopFrac: 0,
    buildType: 0,
    ...options,
  };
}

export function expertDecision(
  game: Game,
  ego: Player,
  opts: ExpertOptions = {},
): ExpertDecision {
  const cfg = { ...DEFAULTS, ...opts };
  const mask = legalMask(game, ego, cfg.opponentIds);
  const maxT = Math.max(1, game.config().maxTroops(ego));
  const ratio = ego.troops() / maxT;

  const can = (t: ActionType) => mask.actionType[t] === true;
  const noop = (): ExpertDecision => ({
    action: {
      actionType: ActionType.NOOP,
      targetPlayer: 0,
      cellX: 0,
      cellY: 0,
      troopFrac: 0,
      buildType: 0,
    },
    macroGoal: 0,
  });
  const enemy = findEnemyTarget(game, ego, cfg.opponentIds);
  const expandTile = findExpandTile(game, ego);

  const cities = ego.units(UnitType.City).length;
  const cityEconomy = cfg.strategy === "city-economy";
  const cityCost = (() => {
    try {
      return Number(game.config().unitInfo(UnitType.City).cost(game, ego));
    } catch {
      return 125_000;
    }
  })();
  const canAffordCity = mask.buildType[0] === true;
  const gold = Number(ego.gold());
  const land = Math.max(1, game.numLandTiles());
  const tileShare = ego.numTilesOwned() / land;
  const nearCityCost = gold >= cityCost * 0.85;

  if (cityEconomy && can(ActionType.BUILD) && canAffordCity) {
    const tile = findBuildTile(game, ego, UnitType.City);
    if (tile !== null) {
      return {
        action: actionAt(ActionType.BUILD, tile, game, { buildType: 0 }),
        macroGoal: 2,
      };
    }
  }

  if (cityEconomy && cities < cfg.targetCities) {
    if (
      can(ActionType.ATTACK) &&
      expandTile !== null &&
      ratio >= cfg.reserveRatio * 0.9 &&
      !(nearCityCost || tileShare >= 0.7)
    ) {
      return {
        action: actionAt(ActionType.ATTACK, expandTile, game, {
          troopFrac: fracIndexForRatio(cfg.expandRatio),
        }),
        macroGoal: 0,
      };
    }
    if (!canAffordCity && (nearCityCost || tileShare >= 0.55)) {
      return noop();
    }
  }

  if (
    (!cityEconomy || cities >= cfg.targetCities) &&
    can(ActionType.BUILD) &&
    mask.buildType[7] &&
    ego.units(UnitType.MissileSilo).length > 0 &&
    enemy !== null
  ) {
    return {
      action: actionAt(
        ActionType.BUILD,
        findBuildTile(game, ego) ?? enemy.tile,
        game,
        { buildType: 7 },
      ),
      macroGoal: 3,
    };
  }

  if (can(ActionType.BUILD) && ratio >= cfg.reserveRatio) {
    const buildChoices: {
      buildType: number;
      tile: TileRef | null;
      macroGoal: number;
    }[] = cityEconomy
      ? [
          {
            buildType: 0,
            tile: findBuildTile(game, ego, UnitType.City),
            macroGoal: 2,
          },
          ...(cities >= cfg.targetCities &&
          ego.units(UnitType.Port).length === 0
            ? [
                {
                  buildType: 1,
                  tile: findCoastalBuildTile(game, ego),
                  macroGoal: 2,
                },
              ]
            : []),
          ...(cities >= cfg.targetCities &&
          ego.units(UnitType.DefensePost).length === 0
            ? [
                {
                  buildType: 2,
                  tile: findBuildTile(game, ego, UnitType.DefensePost),
                  macroGoal: 4,
                },
              ]
            : []),
          ...(cities >= cfg.targetCities &&
          ego.units(UnitType.MissileSilo).length === 0
            ? [
                {
                  buildType: 4,
                  tile: findBuildTile(game, ego, UnitType.MissileSilo),
                  macroGoal: 4,
                },
              ]
            : []),
        ]
      : [
          ...(ego.units(UnitType.Port).length === 0
            ? [
                {
                  buildType: 1,
                  tile: findCoastalBuildTile(game, ego),
                  macroGoal: 2,
                },
              ]
            : []),
          ...(ego.units(UnitType.MissileSilo).length === 0
            ? [
                {
                  buildType: 4,
                  tile: findBuildTile(game, ego, UnitType.MissileSilo),
                  macroGoal: 4,
                },
              ]
            : []),
          ...(ego.units(UnitType.DefensePost).length === 0
            ? [
                {
                  buildType: 2,
                  tile: findBuildTile(game, ego, UnitType.DefensePost),
                  macroGoal: 4,
                },
              ]
            : []),
          {
            buildType: 0,
            tile: findBuildTile(game, ego, UnitType.City),
            macroGoal: 2,
          },
        ];
    const choice = buildChoices.find(
      ({ buildType, tile }) => mask.buildType[buildType] && tile !== null,
    );
    if (choice?.tile !== null && choice !== undefined) {
      return {
        action: actionAt(ActionType.BUILD, choice.tile, game, {
          buildType: choice.buildType,
        }),
        macroGoal: choice.macroGoal,
      };
    }
  }

  if (cityEconomy && cities < cfg.targetCities) {
    if (
      can(ActionType.ATTACK) &&
      expandTile !== null &&
      ratio >= cfg.reserveRatio
    ) {
      return {
        action: actionAt(ActionType.ATTACK, expandTile, game, {
          troopFrac: fracIndexForRatio(Math.min(cfg.expandRatio, 0.1)),
        }),
        macroGoal: 0,
      };
    }
    if (can(ActionType.BOAT)) {
      const shore = findBoatDestination(game, ego);
      if (
        shore !== null &&
        (expandTile === null || isOverseasShore(game, ego, shore))
      ) {
        return {
          action: actionAt(ActionType.BOAT, shore, game, {
            troopFrac: fracIndexForRatio(cfg.expandRatio),
          }),
          macroGoal: 0,
        };
      }
    }
    return noop();
  }

  if (can(ActionType.ATTACK) && ratio >= cfg.triggerRatio && enemy !== null) {
    const spend = Math.min(cfg.expandRatio * 2, ratio - cfg.reserveRatio);
    if (mask.targetPlayer[enemy.targetPlayer]) {
      return {
        action: actionAt(
          ActionType.ATTACK,
          findBuildTile(game, ego) ?? enemy.tile,
          game,
          {
            targetPlayer: enemy.targetPlayer,
            troopFrac: fracIndexForRatio(Math.max(0.1, spend)),
          },
        ),
        macroGoal: 1,
      };
    }
  }

  // Useful overseas boats even when local land expand still exists.
  if (can(ActionType.BOAT) && ratio >= cfg.reserveRatio) {
    const shore = findBoatDestination(game, ego);
    if (shore !== null && isOverseasShore(game, ego, shore)) {
      return {
        action: actionAt(ActionType.BOAT, shore, game, {
          troopFrac: fracIndexForRatio(cfg.expandRatio),
        }),
        macroGoal: 0,
      };
    }
  }

  if (
    can(ActionType.ATTACK) &&
    ratio >= cfg.reserveRatio + cfg.expandRatio * 0.5 &&
    expandTile !== null &&
    mask.targetPlayer[0]
  ) {
    return {
      action: actionAt(ActionType.ATTACK, expandTile, game, {
        troopFrac: fracIndexForRatio(cfg.expandRatio),
      }),
      macroGoal: 0,
    };
  }

  if (can(ActionType.BOAT)) {
    const shore = findBoatDestination(game, ego);
    if (shore !== null) {
      return {
        action: actionAt(ActionType.BOAT, shore, game, {
          troopFrac: fracIndexForRatio(cfg.expandRatio),
        }),
        macroGoal: 0,
      };
    }
  }

  if (
    can(ActionType.ATTACK) &&
    ratio >= cfg.reserveRatio &&
    expandTile !== null &&
    mask.targetPlayer[0]
  ) {
    return {
      action: actionAt(ActionType.ATTACK, expandTile, game, { troopFrac: 0 }),
      macroGoal: 0,
    };
  }

  return noop();
}

export function expertAction(
  game: Game,
  ego: Player,
  opts: ExpertOptions = {},
): FactorizedAction {
  return expertDecision(game, ego, opts).action;
}

export function assertActionLegal(
  game: Game,
  ego: Player,
  action: FactorizedAction,
  opponentIds?: readonly string[],
): boolean {
  const mask = legalMask(game, ego, opponentIds);
  if (!mask.actionType[action.actionType]) return false;
  if (
    action.actionType === ActionType.ATTACK &&
    !mask.targetPlayer[action.targetPlayer]
  ) {
    return false;
  }
  const needsCell = [
    ActionType.SPAWN,
    ActionType.BOAT,
    ActionType.BUILD,
    ActionType.MOVE_WARSHIP,
  ].includes(action.actionType);
  if (needsCell) {
    const cellIdx = action.cellY * COARSE_W + action.cellX;
    if (cellIdx < 0 || cellIdx >= mask.cell.length || !mask.cell[cellIdx]) {
      return mask.cell.length === 0 || mask.cell[cellIdx] !== false;
    }
  }
  if (!mask.troopFrac[action.troopFrac]) {
    if (
      action.actionType === ActionType.ATTACK ||
      action.actionType === ActionType.BOAT
    ) {
      return false;
    }
  }
  if (
    action.actionType === ActionType.BUILD &&
    !mask.buildType[action.buildType]
  ) {
    return false;
  }
  return true;
}
