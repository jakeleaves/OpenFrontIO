/**
 * Record reserve-aware expert trajectories for BC.
 *
 *   npx tsx ai/ts/record_demos.ts --out ai/fixtures/demos/plains.jsonl --episodes 4 --steps 80
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Difficulty } from "../../src/core/game/Game";
import { legalMask } from "./actions";
import { GameEnv } from "./env";
import { assertActionLegal, expertDecision, ExpertStrategy } from "./expert";
import { labelNationDelta, snapshotNationState } from "./nation_label";
import { encodeObservation, obsToJson } from "./obs";
import { noopAction } from "./types";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function parseArgs() {
  const a = process.argv.slice(2);
  const opts = {
    out: path.join(ROOT, "ai/fixtures/demos/world_ffa.jsonl"),
    episodes: 40,
    steps: 200,
    map: "world",
    stride: 5,
    seed: 1,
    teacher: "expert" as "expert" | "nation",
    difficulty: Difficulty.Impossible,
    strategy: "default" as ExpertStrategy,
    targetCities: 3,
    nations: 72,
    bots: 0,
  };
  for (let i = 0; i < a.length; i++) {
    if (a[i] === "--out") opts.out = path.resolve(a[++i]);
    else if (a[i] === "--episodes") opts.episodes = parseInt(a[++i], 10);
    else if (a[i] === "--steps") opts.steps = parseInt(a[++i], 10);
    else if (a[i] === "--map") opts.map = a[++i];
    else if (a[i] === "--stride") opts.stride = parseInt(a[++i], 10);
    else if (a[i] === "--seed") opts.seed = parseInt(a[++i], 10);
    else if (a[i] === "--teacher") {
      const teacher = a[++i];
      if (teacher !== "expert" && teacher !== "nation") {
        throw new Error("--teacher must be expert or nation");
      }
      opts.teacher = teacher;
    } else if (a[i] === "--difficulty") {
      const difficulty = a[++i];
      if (!Object.values(Difficulty).includes(difficulty as Difficulty)) {
        throw new Error(
          "--difficulty must be Easy, Medium, Hard, or Impossible",
        );
      }
      opts.difficulty = difficulty as Difficulty;
    } else if (a[i] === "--strategy") {
      const strategy = a[++i];
      if (strategy !== "default" && strategy !== "city-economy") {
        throw new Error("--strategy must be default or city-economy");
      }
      opts.strategy = strategy;
    } else if (a[i] === "--target-cities") {
      opts.targetCities = parseInt(a[++i], 10);
    } else if (a[i] === "--nations") {
      opts.nations = parseInt(a[++i], 10);
    } else if (a[i] === "--bots") {
      opts.bots = parseInt(a[++i], 10);
    }
  }
  return opts;
}

async function main() {
  console.debug = () => {};
  const opts = parseArgs();
  fs.mkdirSync(path.dirname(opts.out), { recursive: true });
  // Sync append avoids WriteStream writev failures on multi-GB JSONL files.
  fs.writeFileSync(opts.out, "");
  let written = 0;
  let illegal = 0;

  const env = new GameEnv({
    mapName: opts.map,
    macroStride: opts.stride,
    maxTicks: 20_000,
    difficulty: opts.difficulty,
    nations: opts.nations,
    bots: opts.bots,
  });

  for (let ep = 0; ep < opts.episodes; ep++) {
    let { obs, mask, info } = await env.reset(opts.seed + ep);
    for (let t = 0; t < opts.steps; t++) {
      const game = env.getGame();
      const agent = env.getAgent();
      const opponentIds = env.getOpponentIds();
      const nation = env.getNation();
      if (opts.teacher === "nation" && !nation) {
        break;
      }
      const nationBefore =
        opts.teacher === "nation" && nation
          ? snapshotNationState(nation)
          : undefined;
      const obsBefore =
        opts.teacher === "nation" && nation
          ? obsToJson(encodeObservation(game, nation))
          : obsToJson(obs);
      const maskBefore =
        opts.teacher === "nation" && nation
          ? legalMask(game, nation, opponentIds)
          : mask;
      const decision =
        opts.teacher === "expert"
          ? expertDecision(game, agent, {
              strategy: opts.strategy,
              targetCities: opts.targetCities,
              opponentIds,
            })
          : { action: noopAction(), macroGoal: undefined };
      if (
        opts.teacher === "expert" &&
        !assertActionLegal(game, agent, decision.action, opponentIds)
      ) {
        illegal++;
        break;
      }
      const step = env.step(decision.action);
      const action =
        opts.teacher === "nation" && nationBefore && nation
          ? labelNationDelta(
              game,
              nation,
              nationBefore,
              snapshotNationState(nation),
              opponentIds,
            )
          : decision.action;
      fs.appendFileSync(
        opts.out,
        JSON.stringify({
          episode: ep,
          step: t,
          obs: obsBefore,
          mask: maskBefore,
          action,
          troopFracBins: 100,
          strategy: opts.strategy,
          ...(decision.macroGoal !== undefined
            ? { macroGoal: decision.macroGoal }
            : {}),
          reward: step.info.reward,
          done: step.info.done,
          info: step.info,
        }) + "\n",
      );
      written++;
      obs = step.obs;
      mask = step.mask;
      info = step.info;
      if (info.done) break;
    }
    console.log(
      `episode ${ep}: tick=${info.tick} tiles=${info.agentTiles}/${info.nationTiles} ratio=${info.troopRatio.toFixed(2)}`,
    );
  }

  console.log(
    `wrote ${written} transitions → ${opts.out} (illegal=${illegal})`,
  );
  if (illegal > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
