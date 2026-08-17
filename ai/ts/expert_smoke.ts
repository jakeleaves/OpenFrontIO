/**
 * Expert reserve-aware smoke: expand without draining troops to near-zero.
 *
 *   npx tsx ai/ts/expert_smoke.ts
 */
import { GameEnv } from "./env";
import { assertActionLegal, expertAction } from "./expert";
import { ActionType } from "./types";

async function main() {
  console.debug = () => {};
  const env = new GameEnv({
    mapName: "world",
    macroStride: 10,
    maxTicks: 2000,
  });
  let { info } = await env.reset(7);
  let minRatio = 1;
  let expands = 0;
  let illegal = 0;
  const ids = () => env.getOpponentIds();

  for (let i = 0; i < 40; i++) {
    const action = expertAction(env.getGame(), env.getAgent(), {
      opponentIds: ids(),
    });
    if (
      !assertActionLegal(env.getGame(), env.getAgent(), action, ids())
    ) {
      illegal++;
      break;
    }
    if (action.actionType === ActionType.ATTACK && action.targetPlayer === 0) {
      expands++;
    }
    const step = env.step(action);
    info = step.info;
    minRatio = Math.min(minRatio, info.troopRatio);
    if (info.done) break;
  }

  console.log("final", {
    tick: info.tick,
    tiles: info.agentTiles,
    enemies: info.enemyTilesTotal,
    alive: info.opponentsAlive,
    troopRatio: info.troopRatio,
    minRatio,
    expands,
    illegal,
  });

  if (illegal > 0) throw new Error("expert produced illegal action");
  if (info.agentTiles <= 52) throw new Error("expert did not expand");
  if (minRatio < 0.08) {
    throw new Error(`expert drained troops too low: minRatio=${minRatio}`);
  }
  console.log("expert smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
