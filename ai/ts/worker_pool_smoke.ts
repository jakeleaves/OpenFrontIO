import { EnvWorkerPool } from "./worker_pool";
import { ActionType } from "./types";

async function main() {
  const pool = new EnvWorkerPool(2, { mapName: "plains", macroStride: 10 });
  await pool.start();
  const r0 = await pool.reset(0, 1);
  const r1 = await pool.reset(1, 2);
  console.log("w0", (r0.info as { agentTiles: number }).agentTiles);
  console.log("w1", (r1.info as { agentTiles: number }).agentTiles);
  const s0 = await pool.step(0, {
    actionType: ActionType.ATTACK,
    targetPlayer: 0,
    cellX: 16,
    cellY: 8,
    troopFrac: 2,
    buildType: 0,
  });
  console.log("step0", s0.info);
  await pool.close();
  console.log("pool smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
