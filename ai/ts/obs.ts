import {
  Game,
  Player,
  PlayerType,
  UnitType,
} from "../../src/core/game/Game";
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

function borderCenter(game: Game, player: Player): { x: number; y: number } {
  const borders = [...player.borderTiles()];
  if (borders.length === 0) {
    const tiles = [...player.tiles()];
    if (tiles.length === 0) {
      return { x: Math.floor(game.width() / 2), y: Math.floor(game.height() / 2) };
    }
    const t = tiles[0];
    return { x: game.x(t), y: game.y(t) };
  }
  let sx = 0;
  let sy = 0;
  for (const t of borders) {
    sx += game.x(t);
    sy += game.y(t);
  }
  return {
    x: Math.floor(sx / borders.length),
    y: Math.floor(sy / borders.length),
  };
}

export function encodeObservation(game: Game, ego: Player): Observation {
  const global = new Float32Array(GLOBAL_C * GLOBAL_H * GLOBAL_W);
  const local = new Float32Array(LOCAL_C * LOCAL_H * LOCAL_W);
  const vector = new Float32Array(VECTOR_DIM);

  const mw = game.width();
  const mh = game.height();
  const egoSid = ego.smallID();
  const allies = new Set(ego.allies().map((a) => a.smallID()));

  for (let gy = 0; gy < GLOBAL_H; gy++) {
    for (let gx = 0; gx < GLOBAL_W; gx++) {
      const mx = Math.min(
        mw - 1,
        Math.floor(((gx + 0.5) / GLOBAL_W) * mw),
      );
      const my = Math.min(
        mh - 1,
        Math.floor(((gy + 0.5) / GLOBAL_H) * mh),
      );
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
      if (ego.borderTiles().has(r)) {
        set(local, 7, ly, lx, LOCAL_C, LOCAL_H, LOCAL_W, 1);
      }
    }
  }

  const maxT = Math.max(1, game.config().maxTroops(ego));
  const land = Math.max(1, game.numLandTiles());
  const enemy = game
    .players()
    .filter((p) => p !== ego && p.type() !== PlayerType.Bot && p.isAlive())
    .sort((a, b) => b.troops() - a.troops())[0];

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

  return { global, local, vector };
}

export function obsToJson(obs: Observation) {
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
