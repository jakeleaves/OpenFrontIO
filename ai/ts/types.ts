/** Shared obs / action constants for the TS train env. */

export const GLOBAL_C = 12;
export const GLOBAL_H = 64;
export const GLOBAL_W = 128;
export const LOCAL_C = 8;
export const LOCAL_H = 64;
export const LOCAL_W = 64;
export const VECTOR_DIM = 64;

export const COARSE_W = 32;
export const COARSE_H = 16;

/** World Solo FFA: TN expand + up to 72 nation attack slots. */
export const MAX_NATIONS = 72;
/** Index 0 = terra nullius; 1..MAX_NATIONS = fixed nation slots. */
export const NUM_TARGET_PLAYERS = 1 + MAX_NATIONS;

/** Discrete spend labels: 1%, 2%, …, 100% of current troops (decode may bank-clamp). */
export const TROOP_FRACS: readonly number[] = Array.from(
  { length: 100 },
  (_, i) => (i + 1) / 100,
);

export enum ActionType {
  NOOP = 0,
  SPAWN = 1,
  ATTACK = 2,
  BOAT = 3,
  CANCEL_ATTACK = 4,
  CANCEL_BOAT = 5,
  BUILD = 6,
  UPGRADE = 7,
  MOVE_WARSHIP = 8,
  DELETE = 9,
}

export const NUM_ACTION_TYPES = 10;
export const NUM_TROOP_FRACS = TROOP_FRACS.length;
export const NUM_BUILD_TYPES = 10;

export const BUILD_TYPES = [
  "City",
  "Port",
  "DefensePost",
  "SAMLauncher",
  "MissileSilo",
  "Factory",
  "Warship",
  "AtomBomb",
  "HydrogenBomb",
  "MIRV",
] as const;

export type FactorizedAction = {
  actionType: number;
  targetPlayer: number;
  cellX: number;
  cellY: number;
  troopFrac: number;
  buildType: number;
};

export type Observation = {
  global: Float32Array; // C*H*W
  local: Float32Array;
  vector: Float32Array;
};

export type ActionMask = {
  actionType: boolean[];
  targetPlayer: boolean[];
  cell: boolean[];
  troopFrac: boolean[];
  buildType: boolean[];
};

export type StepInfo = {
  tick: number;
  done: boolean;
  reward: number;
  winner: string | null;
  agentTiles: number;
  agentTroops: number;
  agentGold: number;
  /** Number of Cities currently owned by the agent. */
  agentCities: number;
  /** @deprecated Prefer enemyTilesTotal — strongest single opponent tiles. */
  nationTiles: number;
  /** Agent max troop capacity (config.maxTroops). */
  troopCap: number;
  /** agentTroops / troopCap. */
  troopRatio: number;
  /** Current troopIncreaseRate / peak rate for this cap (0–1). */
  growthEfficiency: number;
  /** Ratio that maximizes absolute regen for current troopCap. */
  optimalGrowthRatio: number;
  /** @deprecated Prefer enemyTroopsTotal — strongest single opponent troops. */
  nationTroops: number;
  /** Alive Nation opponents. */
  opponentsAlive: number;
  /** Sum of all opponent land tiles. */
  enemyTilesTotal: number;
  /** Sum of all opponent troops. */
  enemyTroopsTotal: number;
  /** Strongest alive opponent tile count. */
  strongestEnemyTiles: number;
  /** Strongest alive opponent troop count. */
  strongestEnemyTroops: number;
  /** Agent tile-rank among alive non-bot players (1 = most land). */
  placement: number;
  /** Transport ships owned by the agent. */
  agentBoats: number;
  /** Fixed opponent IDs for target slots 1..N (empty slots padded). */
  opponentIds: string[];
};

export function noopAction(): FactorizedAction {
  return {
    actionType: ActionType.NOOP,
    targetPlayer: 0,
    cellX: 0,
    cellY: 0,
    troopFrac: 0,
    buildType: 0,
  };
}
