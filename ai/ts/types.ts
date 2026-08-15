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

export const TROOP_FRACS = [0.1, 0.2, 0.35, 0.5, 0.75] as const;

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
export const NUM_TARGET_PLAYERS = 2; // TN, Enemy
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
  nationTiles: number;
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
