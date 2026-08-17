import { ActionType } from "./types";
import { GameEnv } from "./env";

async function main() {
  console.debug = () => {};
  const env = new GameEnv({
    mapName: "world",
    macroStride: 10,
    maxTicks: 2_000,
  });
  const { info, mask } = await env.reset(42);
  console.log("reset", {
    opponentsAlive: info.opponentsAlive,
    agentTiles: info.agentTiles,
    enemyTilesTotal: info.enemyTilesTotal,
    placement: info.placement,
  });
  console.log(
    "legal actions",
    mask.actionType
      .map((v, i) => (v ? ActionType[i] ?? i : null))
      .filter(Boolean),
  );

  let last = info;
  for (let i = 0; i < 20; i++) {
    const { info: stepInfo } = env.step({
      actionType: ActionType.ATTACK,
      targetPlayer: 0,
      cellX: 16,
      cellY: 8,
      troopFrac: 14,
      buildType: 0,
    });
    last = stepInfo;
    if (i % 5 === 0) {
      console.log(
        `step ${i}: tick=${stepInfo.tick} tiles=${stepInfo.agentTiles}/` +
          `${stepInfo.enemyTilesTotal} alive=${stepInfo.opponentsAlive} ` +
          `reward=${stepInfo.reward.toFixed(2)} done=${stepInfo.done}`,
      );
    }
    if (stepInfo.done) break;
  }
  console.log("final", env.snapshot());
  if (last.agentTiles <= 52) {
    throw new Error(
      `no expansion after ${last.tick} ticks: agent=${last.agentTiles}`,
    );
  }
  console.log("smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
