import {
  Game,
  Player,
  UnitType,
} from "../../src/core/game/Game";
import { Intent } from "../../src/core/Schemas";
import {
  ActionMask,
  ActionType,
  BUILD_TYPES,
  COARSE_H,
  COARSE_W,
  FactorizedAction,
  NUM_ACTION_TYPES,
  NUM_BUILD_TYPES,
  NUM_TARGET_PLAYERS,
  NUM_TROOP_FRACS,
  TROOP_FRACS,
} from "./types";

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

export function coarseToTile(
  game: Game,
  cellX: number,
  cellY: number,
): number {
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

export function legalMask(game: Game, ego: Player): ActionMask {
  const actionType = Array(NUM_ACTION_TYPES).fill(false) as boolean[];
  const targetPlayer = Array(NUM_TARGET_PLAYERS).fill(true) as boolean[];
  const cell = Array(COARSE_H * COARSE_W).fill(true) as boolean[];
  const troopFrac = Array(NUM_TROOP_FRACS).fill(true) as boolean[];
  const buildType = Array(NUM_BUILD_TYPES).fill(false) as boolean[];

  actionType[ActionType.NOOP] = true;

  if (game.inSpawnPhase() && !ego.hasSpawned()) {
    actionType[ActionType.SPAWN] = true;
    return { actionType, targetPlayer, cell, troopFrac, buildType };
  }

  if (!ego.isAlive() || ego.numTilesOwned() === 0) {
    return { actionType, targetPlayer, cell, troopFrac, buildType };
  }

  actionType[ActionType.ATTACK] = true;
  const boats = ego.units(UnitType.TransportShip).length;
  if (boats < game.config().boatMaxNumber()) {
    actionType[ActionType.BOAT] = true;
  }
  actionType[ActionType.BUILD] = true;
  if (ego.units(UnitType.Warship).length > 0) {
    actionType[ActionType.MOVE_WARSHIP] = true;
  }
  if (ego.units().length > 0) {
    actionType[ActionType.DELETE] = true;
    actionType[ActionType.UPGRADE] = true;
  }
  if (ego.outgoingAttacks().length > 0) {
    actionType[ActionType.CANCEL_ATTACK] = true;
  }
  if (boats > 0) {
    actionType[ActionType.CANCEL_BOAT] = true;
  }

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
): Intent | null {
  const tile = coarseToTile(game, a.cellX, a.cellY);
  const frac = TROOP_FRACS[a.troopFrac % NUM_TROOP_FRACS] ?? 0.35;
  const troops = ego.troops() * frac;

  switch (a.actionType) {
    case ActionType.NOOP:
      return null;
    case ActionType.SPAWN:
      return { type: "spawn", tile };
    case ActionType.ATTACK: {
      let targetID: string | null = null;
      if (a.targetPlayer !== 0) {
        const enemies = game
          .players()
          .filter((p) => p !== ego && p.isAlive());
        targetID = enemies[0]?.id() ?? null;
      }
      return { type: "attack", targetID, troops };
    }
    case ActionType.BOAT:
      return { type: "boat", troops, dst: tile };
    case ActionType.BUILD: {
      const name = BUILD_TYPES[a.buildType % NUM_BUILD_TYPES];
      const unit = UNIT_ENUM[name];
      if (!unit) return null;
      return { type: "build_unit", unit, tile };
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
    case ActionType.DELETE: {
      const u = ego.units()[0];
      if (!u) return null;
      return { type: "delete_unit", unitId: u.id() };
    }
    case ActionType.UPGRADE: {
      const u = ego
        .units()
        .find(
          (x) =>
            x.type() === UnitType.City ||
            x.type() === UnitType.Port ||
            x.type() === UnitType.Factory ||
            x.type() === UnitType.SAMLauncher ||
            x.type() === UnitType.MissileSilo,
        );
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
