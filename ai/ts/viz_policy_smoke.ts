/**
 * Headless smoke: heuristic policy expands in GameEnv (same loop as viz).
 *
 *   npx tsx ai/ts/viz_policy_smoke.ts
 */
import { GameEnv } from "./env";
import { obsToJson } from "./obs";
import { heuristicPolicy } from "./policy";

async function main() {
  console.debug = () => {};
  const env = new GameEnv({
    mapName: "plains",
    macroStride: 10,
    maxTicks: 2000,
  });
  let { obs, mask, info } = await env.reset(11);
  const policy = heuristicPolicy();
  await policy.reset?.();
  const startTiles = info.agentTiles;

  for (let i = 0; i < 25; i++) {
    const action = await policy.act({
      tick: info.tick,
      gold: info.agentGold,
      troopRatio: info.troopRatio,
      obs,
      mask,
      game: env.getGame(),
      agent: env.getAgent(),
    });
    const step = env.step(action);
    info = step.info;
    obs = step.obs;
    mask = step.mask;
    if (info.done) break;
  }

  console.log("viz policy smoke", {
    startTiles,
    endTiles: info.agentTiles,
    troopRatio: info.troopRatio,
    tick: info.tick,
  });
  if (info.agentTiles <= startTiles) {
    throw new Error("policy did not expand territory");
  }
  // Ensure obs round-trip stays shaped
  const j = obsToJson(env.observe());
  if (j.global.length !== 12 * 64 * 128) throw new Error("bad global obs");
  console.log("viz_policy_smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
