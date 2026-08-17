/**
 * Convert observable Nation state changes into approximate factorized actions
 * for behavior-cloning demos. The core does not retain an intent history.
 */
import { Game, Player, UnitType } from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
import { resolveOpponentIds } from "./actions";
import {
  ActionType,
  BUILD_TYPES,
  COARSE_H,
  COARSE_W,
  FactorizedAction,
  TROOP_FRACS,
  noopAction,
} from "./types";

export type NationSnapshot = {
  attacks: number;
  units: number;
  gold: number;
  troops: number;
  tiles: number;
  unitCounts: Partial<Record<UnitType, number>>;
};

const BUILD_INDEX: Partial<Record<UnitType, number>> = {
  [UnitType.City]: BUILD_TYPES.indexOf("City"),
  [UnitType.Port]: BUILD_TYPES.indexOf("Port"),
  [UnitType.DefensePost]: BUILD_TYPES.indexOf("DefensePost"),
  [UnitType.SAMLauncher]: BUILD_TYPES.indexOf("SAMLauncher"),
  [UnitType.MissileSilo]: BUILD_TYPES.indexOf("MissileSilo"),
  [UnitType.Factory]: BUILD_TYPES.indexOf("Factory"),
  [UnitType.Warship]: BUILD_TYPES.indexOf("Warship"),
  [UnitType.AtomBomb]: BUILD_TYPES.indexOf("AtomBomb"),
  [UnitType.HydrogenBomb]: BUILD_TYPES.indexOf("HydrogenBomb"),
  [UnitType.MIRV]: BUILD_TYPES.indexOf("MIRV"),
};

export function snapshotNationState(player: Player): NationSnapshot {
  const unitCounts: Partial<Record<UnitType, number>> = {};
  for (const unit of player.units()) {
    unitCounts[unit.type()] = (unitCounts[unit.type()] ?? 0) + 1;
  }
  return {
    attacks: player.outgoingAttacks().length,
    units: player.units().length,
    gold: Number(player.gold()),
    troops: player.troops(),
    tiles: player.numTilesOwned(),
    unitCounts,
  };
}

function tileToCoarse(
  game: Game,
  tile: TileRef,
): Pick<FactorizedAction, "cellX" | "cellY"> {
  return {
    cellX: Math.min(
      COARSE_W - 1,
      Math.floor((game.x(tile) / Math.max(1, game.width())) * COARSE_W),
    ),
    cellY: Math.min(
      COARSE_H - 1,
      Math.floor((game.y(tile) / Math.max(1, game.height())) * COARSE_H),
    ),
  };
}

function troopFracIndex(before: number, after: number): number {
  const spent = Math.max(0.1, (before - after) / Math.max(1, before));
  return TROOP_FRACS.reduce(
    (best, fraction, index) =>
      Math.abs(fraction - spent) < Math.abs(TROOP_FRACS[best] - spent)
        ? index
        : best,
    0,
  );
}

function attackTargetSlot(
  game: Game,
  nation: Player,
  opponentIds?: readonly string[],
): number {
  const atk = nation.outgoingAttacks()[0];
  if (atk) {
    const target = atk.target();
    if (target.isPlayer()) {
      const id = target.id();
      const slots = resolveOpponentIds(game, nation, opponentIds);
      const idx = slots.indexOf(id);
      if (idx >= 0) return idx + 1;
      // Attacking the human agent (not in nation slots) → slot 1 as "enemy".
      return 1;
    }
    return 0; // terra nullius
  }
  const hasEnemy = game
    .players()
    .some((player) => player !== nation && player.isAlive());
  return hasEnemy ? 1 : 0;
}

/**
 * Approximate the Nation's most visible action over one macro stride.
 * Attack deltas take precedence, followed by newly constructed supported units.
 */
export function labelNationDelta(
  game: Game,
  nation: Player,
  before: Pick<NationSnapshot, "attacks" | "units" | "gold" | "troops"> &
    Partial<Pick<NationSnapshot, "unitCounts">>,
  after: NationSnapshot,
  opponentIds?: readonly string[],
): FactorizedAction {
  const fallbackTile = [...nation.tiles()][0] ?? game.ref(0, 0);
  const cell = tileToCoarse(game, fallbackTile);

  if (after.attacks > before.attacks) {
    return {
      actionType: ActionType.ATTACK,
      targetPlayer: attackTargetSlot(game, nation, opponentIds),
      ...cell,
      troopFrac: troopFracIndex(before.troops, after.troops),
      buildType: 0,
    };
  }

  if (after.units > before.units) {
    const builtType = Object.values(UnitType).find(
      (type) =>
        (after.unitCounts[type] ?? 0) > (before.unitCounts?.[type] ?? 0),
    );
    if (builtType === UnitType.TransportShip) {
      const boat = nation.units(UnitType.TransportShip)[0];
      return {
        actionType: ActionType.BOAT,
        ...tileToCoarse(game, boat?.tile() ?? fallbackTile),
        targetPlayer: 0,
        troopFrac: troopFracIndex(before.troops, after.troops),
        buildType: 0,
      };
    }
    const buildType =
      builtType === undefined ? undefined : BUILD_INDEX[builtType];
    if (buildType !== undefined) {
      const unit = nation.units(builtType)[0];
      return {
        actionType: ActionType.BUILD,
        ...tileToCoarse(game, unit?.tile() ?? fallbackTile),
        targetPlayer: 0,
        troopFrac: 0,
        buildType,
      };
    }
  }

  return noopAction();
}
