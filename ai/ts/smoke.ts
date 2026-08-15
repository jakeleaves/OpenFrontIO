import { ActionType } from "./types";
import { GameEnv } from "./env";

async function main() {
  console.debug = () => {};
  const env = new GameEnv({ mapName: "plains", macroStride: 10, maxTicks: 500 });
  const { info, mask } = await env.reset(42);
  console.log("reset", info);
  console.log(
    "legal actions",
    mask.actionType
      .map((v, i) => (v ? ActionType[i] ?? i : null))
      .filter(Boolean),
  );

  let last = info;
  for (let i = 0; i < 30; i++) {
    const { info: stepInfo } = env.step({
      actionType: ActionType.ATTACK,
      targetPlayer: i < 10 ? 0 : 1,
      cellX: 16,
      cellY: 8,
      troopFrac: 2,
      buildType: 0,
    });
    last = stepInfo;
    if (i % 5 === 0) {
      console.log(
        `step ${i}: tick=${stepInfo.tick} tiles=${stepInfo.agentTiles}/${stepInfo.nationTiles} troops=${stepInfo.agentTroops.toFixed(0)} reward=${stepInfo.reward.toFixed(2)} done=${stepInfo.done}`,
      );
    }
    if (stepInfo.done) break;
  }
  console.log("final", env.snapshot());
  console.log("smoke OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
