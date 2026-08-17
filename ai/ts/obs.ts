import { Game, Player, PlayerType, UnitType } from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
import {
  GLOBAL_C,
  GLOBAL_H,
  GLOBAL_W,
  LOCAL_C,
  LOCAL_H,
  LOCAL_W,
  Observation,
  VECTOR_DIM,
} from "./types";
import { growthEfficiency } from "./growth";

function set(
  buf: Float32Array,
  c: number,
  y: number,
  x: number,
  C: number,
  H: number,
  W: number,
  v: number,
) {
  buf[c * H * W + y * W + x] = v;
}

/** Centroid of border tiles without allocating a spread copy of the set. */
export function borderCenter(game: Game, player: Player): { x: number; y: number } {
  const borders = player.borderTiles();
  if (borders.size === 0) {
    if (player.numTilesOwned() === 0) {
      return {
        x: Math.floor(game.width() / 2),
        y: Math.floor(game.height() / 2),
      };
    }
    // First owned tile without spreading the full TileSet.
    let first: TileRef | null = null;
    player.tiles().forEach((t) => {
      if (first === null) first = t;
    });
    if (first === null) {
      return {
        x: Math.floor(game.width() / 2),
        y: Math.floor(game.height() / 2),
      };
    }
    return { x: game.x(first), y: game.y(first) };
  }
  let sx = 0;
  let sy = 0;
  let n = 0;
  borders.forEach((t) => {
    sx += game.x(t);
    sy += game.y(t);
    n++;
  });
  return {
    x: Math.floor(sx / n),
    y: Math.floor(sy / n),
  };
}

export function createObservationBuffers(): Observation {
  return {
    global: new Float32Array(GLOBAL_C * GLOBAL_H * GLOBAL_W),
    local: new Float32Array(LOCAL_C * LOCAL_H * LOCAL_W),
    vector: new Float32Array(VECTOR_DIM),
  };
}

/**
 * Encode observation into `into` when provided (zeroed then filled), otherwise
 * allocate fresh buffers. Training should pass a reused Observation.
 */
export function encodeObservation(
  game: Game,
  ego: Player,
  into?: Observation,
): Observation {
  const global = into?.global ?? new Float32Array(GLOBAL_C * GLOBAL_H * GLOBAL_W);
  const local = into?.local ?? new Float32Array(LOCAL_C * LOCAL_H * LOCAL_W);
  const vector = into?.vector ?? new Float32Array(VECTOR_DIM);
  if (into) {
    global.fill(0);
    local.fill(0);
    vector.fill(0);
  }

  const mw = game.width();
  const mh = game.height();
  const egoSid = ego.smallID();
  const allies = new Set(ego.allies().map((a) => a.smallID()));

  for (let gy = 0; gy < GLOBAL_H; gy++) {
    for (let gx = 0; gx < GLOBAL_W; gx++) {
      const mx = Math.min(mw - 1, Math.floor(((gx + 0.5) / GLOBAL_W) * mw));
      const my = Math.min(mh - 1, Math.floor(((gy + 0.5) / GLOBAL_H) * mh));
      const r = game.ref(mx, my);
      if (game.isWater(r)) {
        set(global, 4, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
        continue;
      }
      const owner = game.ownerID(r);
      if (owner === 0) {
        set(global, 3, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
      } else if (owner === egoSid) {
        set(global, 0, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
      } else if (allies.has(owner)) {
        set(global, 2, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
      } else {
        set(global, 1, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
      }
      set(
        global,
        5,
        gy,
        gx,
        GLOBAL_C,
        GLOBAL_H,
        GLOBAL_W,
        game.magnitude(r) / 31,
      );
      if (game.hasFallout(r)) {
        set(global, 6, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
      }
    }
  }

  for (const p of game.players()) {
    const chan = p === ego ? 8 : 9;
    for (const u of p.units()) {
      const gx = Math.min(
        GLOBAL_W - 1,
        Math.floor((game.x(u.tile()) / mw) * GLOBAL_W),
      );
      const gy = Math.min(
        GLOBAL_H - 1,
        Math.floor((game.y(u.tile()) / mh) * GLOBAL_H),
      );
      let c = chan;
      if (
        u.type() === UnitType.Warship ||
        u.type() === UnitType.TransportShip
      ) {
        c = 10;
      } else if (
        u.type() === UnitType.SAMLauncher ||
        u.type() === UnitType.MissileSilo
      ) {
        c = 11;
      }
      set(global, c, gy, gx, GLOBAL_C, GLOBAL_H, GLOBAL_W, 1);
    }
  }

  const { x: cx, y: cy } = borderCenter(game, ego);
  const half = Math.floor(LOCAL_W / 2);
  const borderTiles = ego.borderTiles();
  for (let ly = 0; ly < LOCAL_H; ly++) {
    for (let lx = 0; lx < LOCAL_W; lx++) {
      const mx = cx + lx - half;
      const my = cy + ly - half;
      if (mx < 0 || my < 0 || mx >= mw || my >= mh) {
        set(local, 4, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
        continue;
      }
      const r = game.ref(mx, my);
      if (game.isWater(r)) {
        set(local, 4, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
        continue;
      }
      const owner = game.ownerID(r);
      if (owner === egoSid) set(local, 0, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
      else if (owner === 0) set(local, 3, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
      else set(local, 1, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
      set(local, 5, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, game.magnitude(r) / 31);
      if (game.hasFallout(r)) {
        set(local, 6, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
      }
      if (borderTiles.has(r)) {
        set(local, 7, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
      }
    }
  }

  const maxT = Math.max(1, game.config().maxTroops(ego));
  const land = Math.max(1, game.numLandTiles());
  const opponents: Player[] = [];
  for (const p of game.players()) {
    if (p !== ego && p.type() !== PlayerType.Bot && p.isAlive()) {
      opponents.push(p);
    }
  }
  let enemy: Player | undefined;
  let bestTroops = -1;
  for (const p of opponents) {
    const t = p.troops();
    if (t > bestTroops) {
      bestTroops = t;
      enemy = p;
    }
  }

  let enemyTilesTotal = 0;
  let enemyTroopsTotal = 0;
  let adjacentEnemies = 0;
  const bordering = new Set<string>();
  borderTiles.forEach((border) => {
    game.forEachNeighbor(border, (neighbor) => {
      if (!game.isLand(neighbor) || !game.hasOwner(neighbor)) return;
      const owner = game.owner(neighbor);
      if (owner.isPlayer() && owner !== ego && owner.isAlive()) {
        bordering.add(owner.id());
      }
    });
  });
  for (const p of opponents) {
    enemyTilesTotal += p.numTilesOwned();
    enemyTroopsTotal += p.troops();
    if (bordering.has(p.id())) adjacentEnemies++;
  }

  let better = 0;
  for (const p of opponents) {
    if (p.numTilesOwned() > ego.numTilesOwned()) better++;
  }
  const placement = better + 1;
  const contenders = opponents.length + (ego.isAlive() ? 1 : 0);

  vector[0] = Math.log1p(Number(ego.gold()));
  vector[1] = Number(game.config().goldAdditionRate(ego));
  vector[2] = ego.troops() / maxT;
  vector[3] = ego.troops() / 100_000;
  vector[4] = ego.numTilesOwned() / land;
  vector[5] = enemy ? enemy.troops() / Math.max(1, ego.troops()) : 0;
  vector[6] = enemy ? enemy.numTilesOwned() / land : 0;
  vector[7] = ego.units(UnitType.City).length;
  vector[8] = ego.units(UnitType.Port).length;
  vector[9] = ego.units(UnitType.SAMLauncher).length;
  vector[10] = ego.units(UnitType.MissileSilo).length;
  vector[11] = ego.units(UnitType.TransportShip).length;
  vector[12] = ego.units(UnitType.Warship).length;
  vector[13] = game.inSpawnPhase() ? 1 : 0;
  vector[14] = game.ticks() / 10_000;
  vector[15] = ego.isTraitor() ? 1 : 0;
  vector[16] = ego.incomingAttacks().length;
  vector[17] = ego.outgoingAttacks().length;
  vector[18] = enemy ? enemy.troops() / Math.max(1, ego.troops()) : 0;
  vector[19] = enemy ? enemy.numTilesOwned() / land : 0;
  vector[20] = enemy?.outgoingAttacks().length ?? 0;
  vector[21] = enemy?.units(UnitType.SAMLauncher).length ?? 0;
  vector[22] = enemy?.units(UnitType.MissileSilo).length ?? 0;
  if (enemy) {
    const enemyCenter = borderCenter(game, enemy);
    const dx = cx - enemyCenter.x;
    const dy = cy - enemyCenter.y;
    vector[23] = Math.sqrt(dx * dx + dy * dy) / Math.hypot(mw, mh);
  }

  const growth = growthEfficiency(ego.troops(), maxT);
  vector[24] = growth.rate / Math.max(1, maxT * 0.01); // normalized regen
  vector[25] = growth.efficiency; // 1 at peak growth ratio
  vector[26] = growth.optimalRatio;
  vector[27] = ego.troops() / maxT - growth.optimalRatio; // signed error vs peak

  // FFA pressure features (slots beyond legacy 1v1 vector layout).
  vector[28] = opponents.length / 72;
  vector[29] = enemyTilesTotal / land;
  vector[30] = enemyTroopsTotal / Math.max(1, ego.troops() + enemyTroopsTotal);
  vector[31] = adjacentEnemies / Math.max(1, opponents.length);
  vector[32] = placement / Math.max(1, contenders);
  vector[33] =
    ego.numTilesOwned() / Math.max(1, enemyTilesTotal + ego.numTilesOwned());
  vector[34] =
    ego.incomingAttacks().reduce((n, a) => n + a.troops(), 0) /
    Math.max(1, ego.troops());

  return { global, local, vector };
}

function f32ToBase64(arr: Float32Array): string {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString(
    "base64",
  );
}

/** Encode observation as base64 Float32 tensors (wire format v2). */
export function obsToJson(obs: Observation) {
  return {
    encoding: "f32b64" as const,
    global: f32ToBase64(obs.global),
    local: f32ToBase64(obs.local),
    vector: f32ToBase64(obs.vector),
    shapes: {
      global: [GLOBAL_C, GLOBAL_H, GLOBAL_W],
      local: [LOCAL_C, LOCAL_H, LOCAL_W],
      vector: VECTOR_DIM,
    },
  };
}

/** Legacy Array-of-floats encoding (tests / demos that need JSON numbers). */
export function obsToJsonLegacy(obs: Observation) {
  return {
    global: Array.from(obs.global),
    local: Array.from(obs.local),
    vector: Array.from(obs.vector),
    shapes: {
      global: [GLOBAL_C, GLOBAL_H, GLOBAL_W],
      local: [LOCAL_C, LOCAL_H, LOCAL_W],
      vector: VECTOR_DIM,
    },
  };
}

export type { TileRef };
