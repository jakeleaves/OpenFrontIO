import { Game, Player, Structures, UnitType } from "../../src/core/game/Game";
import {
  canBuildTransportShip,
  targetTransportTile,
} from "../../src/core/game/TransportShipUtils";
import { Intent } from "../../src/core/Schemas";
import {
  ActionMask,
  ActionType,
  BUILD_TYPES,
  COARSE_H,
  COARSE_W,
  FactorizedAction,
  MAX_NATIONS,
  NUM_ACTION_TYPES,
  NUM_BUILD_TYPES,
  NUM_TARGET_PLAYERS,
  NUM_TROOP_FRACS,
  TROOP_FRACS,
} from "./types";

/** Landing owner class for boat mask/decode round-trips. */
type BoatOwnerClass = "tn" | "enemy";

/** Full training decode vs budgeted viz decode for boat landings. */
export type BoatDecodeMode = "full" | "cheap";

/**
 * Full decoding may search outside the selected cell, but every probe invokes
 * water pathfinding. Keep that fallback bounded so an exploratory BOAT action
 * cannot stall a whole parallel rollout.
 */
const FULL_BOAT_FALLBACK_PROBE_BUDGET = 12;

const UNIT_ENUM: Record<string, UnitType> = {
  City: UnitType.City,
  Port: UnitType.Port,
  DefensePost: UnitType.DefensePost,
  SAMLauncher: UnitType.SAMLauncher,
  MissileSilo: UnitType.MissileSilo,
  Factory: UnitType.Factory,
  Warship: UnitType.Warship,
  AtomBomb: UnitType.AtomBomb,
  HydrogenBomb: UnitType.HydrogenBomb,
  MIRV: UnitType.MIRV,
};

/** True if the player borders any water (boats are impossible otherwise). */
function hasShoreAccess(game: Game, ego: Player): boolean {
  let found = false;
  ego.borderTiles().forEach((border) => {
    if (found) return;
    game.forEachNeighbor(border, (neighbor) => {
      if (found) return;
      if (game.isWater(neighbor)) found = true;
    });
  });
  return found;
}

/** True if ego borders unowned land that can still be expanded into. */
function hasUnownedExpandBorder(game: Game, ego: Player): boolean {
  let found = false;
  ego.borderTiles().forEach((border) => {
    if (found) return;
    game.forEachNeighbor(border, (neighbor) => {
      if (found) return;
      if (
        game.isLand(neighbor) &&
        !game.hasOwner(neighbor) &&
        !game.isImpassable(neighbor)
      ) {
        found = true;
      }
    });
  });
  return found;
}

/** SmallIDs of players that share a land border with ego. */
function borderingEnemyIds(game: Game, ego: Player): Set<string> {
  const ids = new Set<string>();
  ego.borderTiles().forEach((border) => {
    game.forEachNeighbor(border, (neighbor) => {
      if (!game.isLand(neighbor) || !game.hasOwner(neighbor)) return;
      const owner = game.owner(neighbor);
      if (owner.isPlayer() && owner !== ego && owner.isAlive()) {
        ids.add(owner.id());
      }
    });
  });
  return ids;
}

/**
 * Resolve fixed target slots. Prefer explicit opponentIds from the env;
 * otherwise fall back to sorted alive nations (legacy 1v1 demos / tests).
 */
export function resolveOpponentIds(
  game: Game,
  ego: Player,
  opponentIds?: readonly string[],
): string[] {
  if (opponentIds && opponentIds.length > 0) {
    return [...opponentIds].slice(0, MAX_NATIONS);
  }
  return game
    .players()
    .filter((p) => p !== ego && p.isAlive())
    .sort((a, b) => a.id().localeCompare(b.id()))
    .map((p) => p.id())
    .slice(0, MAX_NATIONS);
}

/** Owner class of a prospective boat landing shore (null = invalid). */
function boatOwnerClass(
  game: Game,
  ego: Player,
  tile: number,
): BoatOwnerClass | null {
  if (!game.isLand(tile) || game.isImpassable(tile)) return null;
  if (!game.hasOwner(tile)) return "tn";
  const owner = game.owner(tile);
  if (!owner.isPlayer() || owner === ego || !owner.isAlive()) return null;
  return "enemy";
}

/** Sample land tiles inside a coarse action cell (World cells are large). */
function sampleTilesInCoarseCell(
  game: Game,
  cellX: number,
  cellY: number,
): number[] {
  const cx = Math.min(COARSE_W - 1, Math.max(0, cellX | 0));
  const cy = Math.min(COARSE_H - 1, Math.max(0, cellY | 0));
  const x0 = Math.floor((cx / COARSE_W) * game.width());
  const x1 = Math.max(x0 + 1, Math.floor(((cx + 1) / COARSE_W) * game.width()));
  const y0 = Math.floor((cy / COARSE_H) * game.height());
  const y1 = Math.max(
    y0 + 1,
    Math.floor(((cy + 1) / COARSE_H) * game.height()),
  );
  // Sparse sample — canBuildTransportShip is SpatialQuery-heavy.
  const stepX = Math.max(1, Math.floor((x1 - x0) / 4));
  const stepY = Math.max(1, Math.floor((y1 - y0) / 4));
  const out: number[] = [];
  for (let y = y0; y < y1; y += stepY) {
    for (let x = x0; x < x1; x += stepX) {
      const xx = Math.min(game.width() - 1, x);
      const yy = Math.min(game.height() - 1, y);
      out.push(game.ref(xx, yy));
    }
  }
  out.push(coarseToTile(game, cx, cy));
  // Expand water samples to neighboring land (coarse centers often sit offshore).
  const extra: number[] = [];
  for (const t of out) {
    if (!game.isWater(t)) continue;
    for (const n of game.neighbors(t)) {
      if (game.isLand(n) && !game.isImpassable(n)) extra.push(n);
    }
  }
  return out.concat(extra);
}

/**
 * Resolve a legal boat landing shore inside a coarse cell.
 * Prefers destination owner class matching land inside the cell (enemy vs TN);
 * never returns a water tile as dst.
 */
function resolveBoatInCell(
  game: Game,
  ego: Player,
  cellX: number,
  cellY: number,
): number | null {
  const hint = coarseToTile(game, cellX, cellY);
  const hx = game.x(hint);
  const hy = game.y(hint);

  let prefer: BoatOwnerClass | null = null;
  for (const t of sampleTilesInCoarseCell(game, cellX, cellY)) {
    const cls = boatOwnerClass(game, ego, t);
    if (cls === null) continue;
    if (canBuildTransportShip(game, ego, t) === false) continue;
    if (cls === "enemy") {
      prefer = "enemy";
      break;
    }
    if (prefer === null) prefer = "tn";
  }

  const tryDst = (tile: number): number | null => {
    if (canBuildTransportShip(game, ego, tile) === false) return null;
    const dst = targetTransportTile(game, ego, tile);
    if (dst === null) return null;
    if (canBuildTransportShip(game, ego, dst) === false) return null;
    return dst;
  };

  let bestPreferred: number | null = null;
  let bestPreferredDist = Infinity;
  let bestAny: number | null = null;
  let bestAnyDist = Infinity;

  for (const tile of sampleTilesInCoarseCell(game, cellX, cellY)) {
    const dst = tryDst(tile);
    if (dst === null) continue;
    const dx = game.x(dst) - hx;
    const dy = game.y(dst) - hy;
    const d = dx * dx + dy * dy;
    const cls = boatOwnerClass(game, ego, dst);
    if (prefer !== null && cls === prefer && d < bestPreferredDist) {
      bestPreferredDist = d;
      bestPreferred = dst;
    }
    if (d < bestAnyDist) {
      bestAnyDist = d;
      bestAny = dst;
    }
  }

  return bestPreferred ?? bestAny;
}

/**
 * Viz-only boat landing: sample the coarse cell with a tiny SpatialQuery budget.
 * Returns null (NOOP) instead of scanning the whole map for a fallback shore.
 */
function resolveBoatInCellCheap(
  game: Game,
  ego: Player,
  cellX: number,
  cellY: number,
  probeBudget = 3,
): number | null {
  const samples = sampleTilesInCoarseCell(game, cellX, cellY);
  let remaining = Math.max(1, probeBudget | 0);
  let bestEnemy: number | null = null;
  let bestTn: number | null = null;

  for (const tile of samples) {
    if (remaining <= 0) break;
    const cls = boatOwnerClass(game, ego, tile);
    if (cls === null) continue;
    // One canBuildTransportShip ≈ two water searches; keep the budget tiny.
    remaining -= 1;
    if (canBuildTransportShip(game, ego, tile) === false) continue;
    const dst = targetTransportTile(game, ego, tile);
    if (dst === null) continue;
    const dstCls = boatOwnerClass(game, ego, dst);
    if (dstCls === "enemy") {
      bestEnemy = dst;
      break;
    }
    if (dstCls === "tn" && bestTn === null) bestTn = dst;
  }

  return bestEnemy ?? bestTn;
}

/**
 * Resolve a legal boat landing for a coarse action. Uses in-cell search first,
 * then nearest legal shore to the coarse center (not first border shore).
 */
export function resolveBoatDestination(
  game: Game,
  ego: Player,
  cellX: number,
  cellY: number,
  opts: { mode?: BoatDecodeMode } = {},
): number | null {
  if ((opts.mode ?? "full") === "cheap") {
    return resolveBoatInCellCheap(game, ego, cellX, cellY);
  }
  const inCell = resolveBoatInCell(game, ego, cellX, cellY);
  if (inCell !== null) return inCell;

  const hint = coarseToTile(game, cellX, cellY);
  const hx = game.x(hint);
  const hy = game.y(hint);

  let prefer: BoatOwnerClass | null = null;
  for (const t of sampleTilesInCoarseCell(game, cellX, cellY)) {
    const cls = boatOwnerClass(game, ego, t);
    if (cls === "enemy") {
      prefer = "enemy";
      break;
    }
    if (cls === "tn" && prefer === null) prefer = "tn";
  }

  let bestPreferred: number | null = null;
  let bestPreferredDist = Infinity;
  let bestAny: number | null = null;
  let bestAnyDist = Infinity;
  const probesPerFallbackScan = Math.max(
    1,
    Math.floor(FULL_BOAT_FALLBACK_PROBE_BUDGET / 3),
  );
  let remainingProbes = probesPerFallbackScan;

  const considerFar = (tile: number) => {
    if (remainingProbes <= 0) return;
    remainingProbes--;
    if (canBuildTransportShip(game, ego, tile) === false) return;
    const dst = targetTransportTile(game, ego, tile);
    if (dst === null) return;
    if (canBuildTransportShip(game, ego, dst) === false) return;
    const dx = game.x(dst) - hx;
    const dy = game.y(dst) - hy;
    const d = dx * dx + dy * dy;
    const cls = boatOwnerClass(game, ego, dst);
    if (prefer !== null && cls === prefer && d < bestPreferredDist) {
      bestPreferredDist = d;
      bestPreferred = dst;
    }
    if (d < bestAnyDist) {
      bestAnyDist = d;
      bestAny = dst;
    }
  };

  localFallback: for (const border of ego.borderTiles()) {
    for (const water of game.neighbors(border)) {
      if (!game.isWater(water)) continue;
      for (const shore of game.neighbors(water)) {
        if (!game.isLand(shore) || game.isImpassable(shore)) continue;
        considerFar(shore);
        if (remainingProbes <= 0) break localFallback;
      }
    }
  }
  remainingProbes = probesPerFallbackScan;
  enemyFallback: for (const p of game.players()) {
    if (p === ego || !p.isAlive()) continue;
    let n = 0;
    for (const tile of p.borderTiles()) {
      considerFar(tile);
      if (remainingProbes <= 0) break enemyFallback;
      if (++n >= 24) break;
    }
  }
  remainingProbes = probesPerFallbackScan;
  const stepX = Math.max(1, Math.floor(game.width() / 48));
  const stepY = Math.max(1, Math.floor(game.height() / 24));
  gridFallback: for (let y = 0; y < game.height(); y += stepY) {
    for (let x = 0; x < game.width(); x += stepX) {
      const t = game.ref(x, y);
      if (!game.isLand(t) || game.hasOwner(t) || game.isImpassable(t)) continue;
      considerFar(t);
      if (remainingProbes <= 0) break gridFallback;
    }
  }

  return bestPreferred ?? bestAny;
}

/**
 * Boat cell mask cache. SpatialQuery inside canBuildTransportShip dominates
 * World FFA step time (~3s/mask). Recompute every BOAT_MASK_REFRESH_TICKS.
 */
const BOAT_MASK_REFRESH_TICKS = 25;
const GEOMETRY_BOAT_MASK_REFRESH_TICKS = 100;
export type BoatMaskMode = "verified" | "geometry";
type BoatMaskCache = {
  egoID: string;
  mode: BoatMaskMode;
  refreshTicks: number;
  tickBucket: number;
  tilesOwned: number;
  cells: boolean[];
  any: boolean;
};
const boatMaskByGame = new WeakMap<object, BoatMaskCache>();

/** Clear boat mask cache (tests / env reset). */
export function clearBoatMaskCache(): void {
  // WeakMap entries drop when Game is GC'd; nothing global to clear.
}

/**
 * Cheap shore plausibility without SpatialQuery: land, not ego-owned,
 * borders water, and has a valid owner class for landing.
 */
function isPlausibleBoatShore(game: Game, ego: Player, shore: number): boolean {
  if (!game.isLand(shore) || game.isImpassable(shore)) return false;
  if (boatOwnerClass(game, ego, shore) === null) return false;
  for (const n of game.neighbors(shore)) {
    if (game.isWater(n)) return true;
  }
  return false;
}

/** Mark coarse cells that look like reachable boat landing shores. */
function markBoatLandingCells(
  game: Game,
  ego: Player,
  markCell: (tile: number) => void,
  mode: BoatMaskMode,
  refreshTicks: number,
): boolean {
  const egoID = ego.id();
  const tick = game.ticks();
  const tickBucket = Math.floor(tick / refreshTicks);
  const tilesOwned = ego.numTilesOwned();
  const cached = boatMaskByGame.get(game);

  if (
    cached &&
    cached.egoID === egoID &&
    cached.mode === mode &&
    cached.refreshTicks === refreshTicks &&
    cached.tickBucket === tickBucket &&
    // Verified training masks pick up large territory changes promptly.
    // Geometry playback masks deliberately stay stale within their bucket.
    (mode === "geometry" ||
      Math.abs(cached.tilesOwned - tilesOwned) <
        Math.max(50, tilesOwned * 0.05))
  ) {
    let any = false;
    for (let i = 0; i < cached.cells.length; i++) {
      if (!cached.cells[i]) continue;
      any = true;
      const cy = (i / COARSE_W) | 0;
      const cx = i % COARSE_W;
      markCell(coarseToTile(game, cx, cy));
    }
    return any || cached.any;
  }

  const marked = new Set<number>();
  let any = false;
  // Cap expensive SpatialQuery verifications — mask may be slightly loose;
  // decode still runs resolveBoatDestination and will NOOP illegal boats.
  let verifyBudget = mode === "verified" ? 12 : 0;

  const tryMark = (shore: number, verify: boolean) => {
    if (!isPlausibleBoatShore(game, ego, shore)) return;
    const cellX = Math.min(
      COARSE_W - 1,
      Math.floor((game.x(shore) / game.width()) * COARSE_W),
    );
    const cellY = Math.min(
      COARSE_H - 1,
      Math.floor((game.y(shore) / game.height()) * COARSE_H),
    );
    const key = cellY * COARSE_W + cellX;
    if (marked.has(key)) {
      any = true;
      return;
    }
    if (mode === "verified" && verify && verifyBudget > 0) {
      verifyBudget -= 1;
      if (canBuildTransportShip(game, ego, shore) === false) return;
    }
    marked.add(key);
    markCell(shore);
    any = true;
  };

  // Local ring: ego border → water → nearby shores (geometry only + sparse verify).
  for (const border of ego.borderTiles()) {
    for (const water of game.neighbors(border)) {
      if (!game.isWater(water)) continue;
      for (const shore of game.neighbors(water)) {
        tryMark(shore, true);
      }
    }
  }

  // Sample enemy border shores (capped; verify sparsely).
  for (const p of game.players()) {
    if (p === ego || !p.isAlive()) continue;
    let n = 0;
    for (const tile of p.borderTiles()) {
      tryMark(tile, n < 4);
      if (++n >= 12) break;
    }
  }

  // Sparse TN coastal shores — sample water cells, mark adjacent unowned land.
  // Geometry only (no SpatialQuery). Decode verifies reachability.
  const stepX = Math.max(16, Math.floor(game.width() / 16));
  const stepY = Math.max(16, Math.floor(game.height() / 8));
  let tnMarks = 0;
  for (let y = 0; y < game.height() && tnMarks < 48; y += stepY) {
    for (let x = 0; x < game.width() && tnMarks < 48; x += stepX) {
      const t = game.ref(x, y);
      if (!game.isWater(t)) continue;
      for (const shore of game.neighbors(t)) {
        if (
          !game.isLand(shore) ||
          game.hasOwner(shore) ||
          game.isImpassable(shore)
        ) {
          continue;
        }
        const before = marked.size;
        tryMark(shore, false);
        if (marked.size > before) {
          tnMarks += 1;
          break;
        }
      }
    }
  }

  const cells = Array(COARSE_H * COARSE_W).fill(false) as boolean[];
  for (const key of marked) cells[key] = true;
  boatMaskByGame.set(game, {
    egoID,
    mode,
    refreshTicks,
    tickBucket,
    tilesOwned,
    cells,
    any,
  });
  return any;
}

export function coarseToTile(game: Game, cellX: number, cellY: number): number {
  const cx = Math.min(COARSE_W - 1, Math.max(0, cellX | 0));
  const cy = Math.min(COARSE_H - 1, Math.max(0, cellY | 0));
  const x = Math.min(
    game.width() - 1,
    Math.floor(((cx + 0.5) / COARSE_W) * game.width()),
  );
  const y = Math.min(
    game.height() - 1,
    Math.floor(((cy + 0.5) / COARSE_H) * game.height()),
  );
  return game.ref(x, y);
}

/** Resolve a legal structure spawn near the coarse cell (structureMinDist-aware). */
export function resolveBuildTile(
  game: Game,
  ego: Player,
  unit: UnitType,
  hintTile: number,
): number | null {
  const direct = ego.canBuild(unit, hintTile);
  if (direct !== false) return direct;

  // Do NOT scan every owned tile — canBuild runs a radius-15 BFS+sort and that
  // is multi-second at 100k+ territory. Probe near the hint, then borders.
  const hx = game.x(hintTile);
  const hy = game.y(hintTile);
  let best: number | null = null;
  let bestDist = Infinity;
  let probes = 64;

  const consider = (tile: number) => {
    if (probes-- <= 0) return false;
    if (game.owner(tile) !== ego) return true;
    const spawn = ego.canBuild(unit, tile);
    if (spawn === false) return true;
    const dx = game.x(spawn) - hx;
    const dy = game.y(spawn) - hy;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      best = spawn;
    }
    return true;
  };

  // Spiral-ish neighborhood around the coarse-cell center.
  for (let r = 1; r <= 24 && probes > 0; r++) {
    for (let dy = -r; dy <= r && probes > 0; dy++) {
      for (let dx = -r; dx <= r && probes > 0; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = hx + dx;
        const y = hy + dy;
        if (!game.isValidCoord(x, y)) continue;
        if (!consider(game.ref(x, y))) break;
      }
    }
    if (best !== null) return best;
  }

  for (const border of ego.borderTiles()) {
    if (!consider(border)) break;
    if (best !== null && probes < 32) break;
  }
  return best;
}

export type LegalMaskOptions = {
  /**
   * Set false for display-only policy playback. Boat landing validation
   * performs expensive water reachability searches that grow with territory.
   * Omit to preserve the complete training/action mask.
   */
  includeBoatActions?: boolean;
  /**
   * "geometry" marks plausible coastal cells without running water
   * reachability. It is intended for responsive policy visualization; decode
   * still validates a selected BOAT action before creating an intent.
   */
  boatMaskMode?: BoatMaskMode;
  /** Override boat-mask cache duration. Defaults to 25 verified / 100 geometry. */
  boatMaskRefreshTicks?: number;
};

export type DecodeIntentOptions = {
  /**
   * "cheap" only probes a few tiles inside the selected coarse cell.
   * Used by live viz so a loose geometry boat mask cannot stall playback
   * on full-map water reachability searches.
   */
  boatDecodeMode?: BoatDecodeMode;
};

export function legalMask(
  game: Game,
  ego: Player,
  opponentIds?: readonly string[],
  opts: LegalMaskOptions = {},
): ActionMask {
  const actionType = Array(NUM_ACTION_TYPES).fill(false) as boolean[];
  const targetPlayer = Array(NUM_TARGET_PLAYERS).fill(false) as boolean[];
  const cell = Array(COARSE_H * COARSE_W).fill(false) as boolean[];
  const troopFrac = Array(NUM_TROOP_FRACS).fill(true) as boolean[];
  const buildType = Array(NUM_BUILD_TYPES).fill(false) as boolean[];

  actionType[ActionType.NOOP] = true;

  if (game.inSpawnPhase() && !ego.hasSpawned()) {
    actionType[ActionType.SPAWN] = true;
    cell.fill(true);
    return { actionType, targetPlayer, cell, troopFrac, buildType };
  }

  if (!ego.isAlive() || ego.numTilesOwned() === 0) {
    return { actionType, targetPlayer, cell, troopFrac, buildType };
  }

  const markCell = (tile: number) => {
    const x = Math.min(
      COARSE_W - 1,
      Math.floor((game.x(tile) / game.width()) * COARSE_W),
    );
    const y = Math.min(
      COARSE_H - 1,
      Math.floor((game.y(tile) / game.height()) * COARSE_H),
    );
    cell[y * COARSE_W + x] = true;
  };

  const canExpand = hasUnownedExpandBorder(game, ego);
  const bordering = borderingEnemyIds(game, ego);
  const slots = resolveOpponentIds(game, ego, opponentIds);

  // Mark coarse cells from borders + expand neighbors + structures/boats —
  // not every owned tile (O(territory) on World).
  ego.borderTiles().forEach((border) => {
    markCell(border);
    game.forEachNeighbor(border, (neighbor) => {
      if (
        game.isLand(neighbor) &&
        !game.hasOwner(neighbor) &&
        !game.isImpassable(neighbor)
      ) {
        markCell(neighbor);
      }
    });
  });
  for (const u of ego.units()) {
    if (Structures.has(u.type()) || u.type() === UnitType.TransportShip) {
      markCell(u.tile());
    }
  }

  const maxTroops = Math.max(1, game.config().maxTroops(ego));
  const ratio = ego.troops() / maxTroops;
  const EXPAND = 0.15;
  const RESERVE = 0.35;
  const TRIGGER = 0.55;
  const bank = ratio >= RESERVE ? RESERVE : EXPAND;
  const maxSpend = ratio > bank + 1e-6 ? 1 - bank / ratio : 0;

  // TN expand (target 0) only while unowned land remains on the border.
  targetPlayer[0] = canExpand;
  // Per-nation slots: alive + bordering + above war trigger.
  let anyEnemy = false;
  for (let i = 0; i < slots.length; i++) {
    const id = slots[i];
    if (!game.hasPlayer(id)) continue;
    const p = game.player(id);
    if (!p.isAlive() || p.numTilesOwned() === 0) continue;
    if (!bordering.has(id)) continue;
    if (ratio < TRIGGER) continue;
    targetPlayer[i + 1] = true;
    anyEnemy = true;
  }

  actionType[ActionType.ATTACK] =
    maxSpend > 0.02 && (targetPlayer[0] || anyEnemy);

  const boats = ego.units(UnitType.TransportShip).length;
  if (
    opts.includeBoatActions !== false &&
    boats < game.config().boatMaxNumber() &&
    maxSpend > 0.02 &&
    hasShoreAccess(game, ego)
  ) {
    const boatMaskMode = opts.boatMaskMode ?? "verified";
    const defaultRefreshTicks =
      boatMaskMode === "geometry"
        ? GEOMETRY_BOAT_MASK_REFRESH_TICKS
        : BOAT_MASK_REFRESH_TICKS;
    const refreshTicks = Math.max(
      1,
      Math.floor(opts.boatMaskRefreshTicks ?? defaultRefreshTicks),
    );
    const boatCells = markBoatLandingCells(
      game,
      ego,
      markCell,
      boatMaskMode,
      refreshTicks,
    );
    actionType[ActionType.BOAT] = boatCells;
  }

  if (!cell.some(Boolean)) cell.fill(true);

  actionType[ActionType.BUILD] = true;
  if (ego.units(UnitType.Warship).length > 0) {
    actionType[ActionType.MOVE_WARSHIP] = true;
  }
  // Never enable DELETE: decode used to fall back to units()[0] (often City).
  if (ego.units().length > 0) {
    actionType[ActionType.UPGRADE] = true;
  }
  if (ego.outgoingAttacks().length > 0) {
    actionType[ActionType.CANCEL_ATTACK] = true;
  }
  if (boats > 0) {
    actionType[ActionType.CANCEL_BOAT] = true;
  }

  troopFrac.fill(false);
  for (let i = 0; i < TROOP_FRACS.length; i++) {
    troopFrac[i] = TROOP_FRACS[i] <= maxSpend + 1e-6;
  }
  if (actionType[ActionType.ATTACK] && !troopFrac.some(Boolean)) {
    troopFrac[0] = true;
  }
  if (!troopFrac.some(Boolean)) troopFrac[0] = true;

  for (let i = 0; i < BUILD_TYPES.length; i++) {
    const ut = UNIT_ENUM[BUILD_TYPES[i]];
    if (!ut || game.config().isUnitDisabled(ut)) continue;
    try {
      const cost = Number(game.config().unitInfo(ut).cost(game, ego));
      buildType[i] = Number(ego.gold()) >= cost;
    } catch {
      buildType[i] = false;
    }
  }

  return { actionType, targetPlayer, cell, troopFrac, buildType };
}

export function decodeIntent(
  game: Game,
  ego: Player,
  a: FactorizedAction,
  opponentIds?: readonly string[],
  opts: DecodeIntentOptions = {},
): Intent | null {
  const tile = coarseToTile(game, a.cellX, a.cellY);
  const frac = TROOP_FRACS[a.troopFrac % NUM_TROOP_FRACS] ?? 0.35;
  const maxTroops = Math.max(1, game.config().maxTroops(ego));
  const bank =
    a.actionType === ActionType.ATTACK && a.targetPlayer !== 0 ? 0.35 : 0.15;
  const reserve = maxTroops * bank;
  const troops = Math.max(
    0,
    Math.min(ego.troops() * frac, ego.troops() - reserve),
  );
  const slots = resolveOpponentIds(game, ego, opponentIds);

  switch (a.actionType) {
    case ActionType.NOOP:
      return null;
    case ActionType.SPAWN:
      return { type: "spawn", tile };
    case ActionType.ATTACK: {
      if (troops <= 0) return null;
      let targetID: string | null = null;
      if (a.targetPlayer !== 0) {
        const slot = (a.targetPlayer | 0) - 1;
        // Legacy 1v1 demos use targetPlayer=1 with a single enemy.
        const id =
          slot >= 0 && slot < slots.length ? slots[slot] : (slots[0] ?? null);
        if (id === null || !game.hasPlayer(id)) return null;
        const enemy = game.player(id);
        if (!enemy.isAlive() || enemy.numTilesOwned() === 0) return null;
        if (!borderingEnemyIds(game, ego).has(id)) return null;
        targetID = id;
      } else if (!hasUnownedExpandBorder(game, ego)) {
        return null;
      }
      return { type: "attack", targetID, troops };
    }
    case ActionType.BOAT: {
      if (troops <= 0) return null;
      // Resolve a real landing shore (never coarse water center as dst).
      const dst = resolveBoatDestination(game, ego, a.cellX, a.cellY, {
        mode: opts.boatDecodeMode ?? "full",
      });
      if (dst === null) return null;
      return { type: "boat", troops, dst };
    }
    case ActionType.BUILD: {
      const name = BUILD_TYPES[a.buildType % NUM_BUILD_TYPES];
      const unit = UNIT_ENUM[name];
      if (!unit) return null;
      const buildTile = resolveBuildTile(game, ego, unit, tile);
      if (buildTile === null) return null;
      return { type: "build_unit", unit, tile: buildTile };
    }
    case ActionType.CANCEL_ATTACK: {
      const atk = ego.outgoingAttacks()[0];
      if (!atk) return null;
      return { type: "cancel_attack", attackID: atk.id() };
    }
    case ActionType.CANCEL_BOAT: {
      const boat = ego.units(UnitType.TransportShip)[0];
      if (!boat) return null;
      return { type: "cancel_boat", unitID: boat.id() };
    }
    case ActionType.MOVE_WARSHIP: {
      const ships = ego.units(UnitType.Warship);
      if (ships.length === 0) return null;
      return {
        type: "move_warship",
        unitIds: ships.map((s) => s.id()),
        tile,
      };
    }
    case ActionType.DELETE:
      return null;
    case ActionType.UPGRADE: {
      const u = ego
        .units()
        .filter(
          (x) =>
            (x.type() === UnitType.City ||
              x.type() === UnitType.Port ||
              x.type() === UnitType.Factory ||
              x.type() === UnitType.SAMLauncher ||
              x.type() === UnitType.MissileSilo) &&
            ego.canUpgradeUnit(x),
        )
        .sort((a, b) => a.level() - b.level())[0];
      if (!u) return null;
      return {
        type: "upgrade_structure",
        unit: u.type(),
        unitId: u.id(),
        amount: 1,
      };
    }
    default:
      return null;
  }
}
