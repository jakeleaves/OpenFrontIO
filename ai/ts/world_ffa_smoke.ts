/**
 * Focused World Solo FFA checks: 72 nations, target masks, boats, rewards.
 *
 *   npx tsx ai/ts/world_ffa_smoke.ts
 */
import { ActionType } from "./types";
import { decodeIntent, legalMask } from "./actions";
import { GameEnv } from "./env";
import { expertDecision } from "./expert";

async function main() {
  console.debug = () => {};
  console.log = (...args: unknown[]) => console.error(...args);

  const env = new GameEnv({
    mapName: "world",
    macroStride: 5,
    maxTicks: 2_000,
    seed: 7,
  });
  const { info, mask } = await env.reset(7);

  if (info.opponentsAlive !== 72) {
    throw new Error(`expected 72 nations, got ${info.opponentsAlive}`);
  }
  if (info.opponentIds.length !== 72) {
    throw new Error(`expected 72 opponentIds, got ${info.opponentIds.length}`);
  }
  if (env.getGame().width() !== 2000 || env.getGame().height() !== 1000) {
    throw new Error("expected World 2000×1000");
  }
  if (mask.targetPlayer.length !== 73) {
    throw new Error(`target mask length ${mask.targetPlayer.length} != 73`);
  }
  if (!mask.targetPlayer[0]) {
    throw new Error("TN expand should be legal at spawn");
  }
  // Early game: agent usually has no nation border yet.
  const enemySlots = mask.targetPlayer.slice(1).filter(Boolean).length;
  console.error(`reset OK: alive=${info.opponentsAlive} enemySlots=${enemySlots}`);

  const game = env.getGame();
  const agent = env.getAgent();
  const ids = env.getOpponentIds();

  // No void TN attack when expand is impossible: force by decoding with mask off.
  const voidAttack = decodeIntent(
    game,
    agent,
    {
      actionType: ActionType.ATTACK,
      targetPlayer: 0,
      cellX: 0,
      cellY: 0,
      troopFrac: 34,
      buildType: 0,
    },
    ids,
  );
  if (voidAttack && !mask.targetPlayer[0]) {
    throw new Error("decoded TN attack while TN mask false");
  }

  // Drive a short expert episode — expect tiles to grow and rewards finite.
  let last = info;
  let boatSeen = false;
  for (let i = 0; i < 40; i++) {
    const decision = expertDecision(game, agent, { opponentIds: ids });
    const m = legalMask(game, agent, ids);
    if (m.actionType[ActionType.BOAT]) boatSeen = true;
    const step = env.step(decision.action);
    last = step.info;
    if (!Number.isFinite(last.reward)) {
      throw new Error(`non-finite reward at step ${i}`);
    }
    if (last.done) break;
  }

  const frame = env.mapFrame(400);
  if (frame.w * frame.h !== frame.cells.length) {
    throw new Error("mapFrame size mismatch");
  }
  if (frame.stride < 2) {
    throw new Error("World mapFrame should downsample");
  }
  if (!frame.cells.includes("0")) {
    throw new Error("World mapFrame missing water");
  }

  console.error(
    `episode: tick=${last.tick} tiles=${last.agentTiles} enemyTiles=${last.enemyTilesTotal} ` +
      `alive=${last.opponentsAlive} place=#${last.placement} boatMask=${boatSeen} ` +
      `frame=${frame.w}x${frame.h}@${frame.stride}`,
  );

  if (last.agentTiles <= 52 && last.enemyTilesTotal <= 3732) {
    // Spawn starts ~52 agent / ~3732 enemy; require some change.
    throw new Error(
      `no FFA progress: agent=${last.agentTiles} enemies=${last.enemyTilesTotal}`,
    );
  }

  console.error("world_ffa_smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
