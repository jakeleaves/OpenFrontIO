/**
 * Focused boat decode/mask probe (diagnosis regressions).
 *
 *   npx tsx ai/ts/boat_decode_smoke.ts
 *
 * Asserts:
 * 1) Enemy shore BOAT decode does not remap to TN/water
 * 2) Distant reachable TN shores can appear in the boat cell mask
 */
import { canBuildTransportShip } from "../../src/core/game/TransportShipUtils";
import {
  coarseToTile,
  decodeIntent,
  legalMask,
  resolveBoatDestination,
} from "./actions";
import { GameEnv } from "./env";
import { expertDecision } from "./expert";
import { ActionType, COARSE_H, COARSE_W } from "./types";

function tileToCoarse(
  game: {
    x: (t: number) => number;
    y: (t: number) => number;
    width: () => number;
    height: () => number;
  },
  tile: number,
): { cellX: number; cellY: number } {
  return {
    cellX: Math.min(
      COARSE_W - 1,
      Math.floor((game.x(tile) / game.width()) * COARSE_W),
    ),
    cellY: Math.min(
      COARSE_H - 1,
      Math.floor((game.y(tile) / game.height()) * COARSE_H),
    ),
  };
}

async function main() {
  console.debug = () => {};
  // Smaller FFA for a fast coastal setup; decode/mask logic is map-agnostic.
  const env = new GameEnv({
    mapName: "onion",
    macroStride: 5,
    maxTicks: 4_000,
    seed: 7,
    nations: 4,
    bots: 0,
  });
  await env.reset(7);
  const ids = env.getOpponentIds();

  let boatLegal = false;
  for (let i = 0; i < 160; i++) {
    const game = env.getGame();
    const agent = env.getAgent();
    const mask = legalMask(game, agent, ids);
    if (mask.actionType[ActionType.BOAT]) boatLegal = true;
    const decision = expertDecision(game, agent, { opponentIds: ids });
    const { info } = env.step(decision.action);
    if (info.done) break;
    if (boatLegal) break;
  }

  const game = env.getGame();
  const agent = env.getAgent();
  const mask = legalMask(game, agent, ids);
  if (!mask.actionType[ActionType.BOAT]) {
    throw new Error("BOAT never became legal — cannot probe decode");
  }
  const geometryMask = legalMask(game, agent, ids, {
    boatMaskMode: "geometry",
    boatMaskRefreshTicks: 100,
  });
  if (!geometryMask.actionType[ActionType.BOAT]) {
    throw new Error("geometry BOAT mask did not expose a coastal destination");
  }
  if (!geometryMask.cell.some(Boolean)) {
    throw new Error("geometry BOAT mask did not mark any coarse cells");
  }

  let enemyShore: number | null = null;
  for (const p of game.players()) {
    if (p === agent || !p.isAlive()) continue;
    for (const tile of p.borderTiles()) {
      if (canBuildTransportShip(game, agent, tile) !== false) {
        enemyShore = tile;
        break;
      }
    }
    if (enemyShore !== null) break;
  }
  if (enemyShore === null) {
    throw new Error("no reachable enemy shore for decode probe");
  }

  const { cellX, cellY } = tileToCoarse(game, enemyShore);
  const center = coarseToTile(game, cellX, cellY);
  const cheapResolved = resolveBoatDestination(game, agent, cellX, cellY, {
    mode: "cheap",
  });
  // Cheap mode may NOOP when the selected cell has no nearby legal shore —
  // that is intentional. It must never throw or scan the whole map.
  if (cheapResolved !== null && !game.isLand(cheapResolved)) {
    throw new Error(`cheap boat decode returned non-land dst=${cheapResolved}`);
  }
  console.error(
    `PASS cheap decode: cell=(${cellX},${cellY}) dst=${
      cheapResolved === null
        ? "null"
        : `(${game.x(cheapResolved)},${game.y(cheapResolved)})`
    }`,
  );
  const resolved = resolveBoatDestination(game, agent, cellX, cellY);
  if (resolved === null) {
    throw new Error(
      `resolveBoatDestination null for enemy cell (${cellX},${cellY}) shore=${enemyShore}`,
    );
  }
  if (!game.isLand(resolved)) {
    throw new Error(`resolved boat dst is not land: ${resolved}`);
  }
  if (!game.hasOwner(resolved)) {
    throw new Error(
      `enemy cell decode remapped to TN shore ${resolved} ` +
        `(center=${center} water=${game.isWater(center)} shore=${enemyShore})`,
    );
  }
  const owner = game.owner(resolved);
  if (!owner.isPlayer() || owner === agent) {
    throw new Error(`resolved dst owner is not enemy player`);
  }

  const intent = decodeIntent(
    game,
    agent,
    {
      actionType: ActionType.BOAT,
      targetPlayer: 0,
      cellX,
      cellY,
      troopFrac: 14,
      buildType: 0,
    },
    ids,
  );
  if (!intent || intent.type !== "boat") {
    throw new Error(
      `decodeIntent BOAT failed for enemy cell (${cellX},${cellY})`,
    );
  }
  if (!game.hasOwner(intent.dst)) {
    throw new Error(
      `decodeIntent remapped enemy boat to TN/unowned dst=${intent.dst}`,
    );
  }
  const intentOwner = game.owner(intent.dst);
  if (!intentOwner.isPlayer() || intentOwner === agent) {
    throw new Error(`decodeIntent boat dst not enemy-owned`);
  }

  console.error(
    `PASS enemy decode: cell=(${cellX},${cellY}) center=(${game.x(center)},${game.y(center)}) ` +
      `water=${game.isWater(center)} dst=(${game.x(intent.dst)},${game.y(intent.dst)}) ` +
      `owner=${intentOwner.id()}`,
  );

  let distantTnMasked = false;
  let distantTnReachable = 0;
  const stepX = Math.max(4, Math.floor(game.width() / 24));
  const stepY = Math.max(4, Math.floor(game.height() / 12));
  for (let y = 0; y < game.height(); y += stepY) {
    for (let x = 0; x < game.width(); x += stepX) {
      const t = game.ref(x, y);
      if (!game.isLand(t) || game.hasOwner(t) || game.isImpassable(t)) continue;
      if (canBuildTransportShip(game, agent, t) === false) continue;
      let near = false;
      for (const b of agent.borderTiles()) {
        const dx = game.x(t) - game.x(b);
        const dy = game.y(t) - game.y(b);
        if (dx * dx + dy * dy < 25 * 25) {
          near = true;
          break;
        }
      }
      if (near) continue;
      distantTnReachable += 1;
      const c = tileToCoarse(game, t);
      if (mask.cell[c.cellY * COARSE_W + c.cellX]) distantTnMasked = true;
    }
  }
  if (distantTnReachable === 0) {
    // Onion may be one landmass — fall back: any reachable TN cell marked.
    for (let y = 0; y < game.height(); y += stepY) {
      for (let x = 0; x < game.width(); x += stepX) {
        const t = game.ref(x, y);
        if (!game.isLand(t) || game.hasOwner(t) || game.isImpassable(t)) {
          continue;
        }
        if (canBuildTransportShip(game, agent, t) === false) continue;
        distantTnReachable += 1;
        const c = tileToCoarse(game, t);
        if (mask.cell[c.cellY * COARSE_W + c.cellX]) distantTnMasked = true;
      }
    }
  }
  if (distantTnReachable === 0) {
    throw new Error("no reachable TN shores sampled — unexpected");
  }
  if (!distantTnMasked) {
    throw new Error(
      `TN reachable (${distantTnReachable}) but none masked in legalMask.cell`,
    );
  }
  console.error(
    `PASS TN mask: reachable=${distantTnReachable} maskedCell=true boatLegal=${mask.actionType[ActionType.BOAT]}`,
  );
  console.error("boat_decode_smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
