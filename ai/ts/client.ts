/**
 * Local / self-hosted OpenFront intent client.
 *
 * Modes:
 *   --demo     Run GameEnv offline heuristic (no WebSocket)
 *   --ws URL   Connect to a GAME_ENV=dev server (no public matchmaking)
 *
 * Optional: load ONNX via onnxruntime-node if --onnx is set (demo path uses heuristic otherwise).
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GameEnv } from "./env";
import { ActionType, FactorizedAction } from "./types";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

function parseArgs() {
  const a = process.argv.slice(2);
  const opts: {
    demo: boolean;
    ticks: number;
    map: string;
    ws?: string;
    gameId?: string;
    onnx?: string;
  } = { demo: false, ticks: 400, map: "plains" };
  for (let i = 0; i < a.length; i++) {
    if (a[i] === "--demo") opts.demo = true;
    else if (a[i] === "--ticks") opts.ticks = parseInt(a[++i], 10);
    else if (a[i] === "--map") opts.map = a[++i];
    else if (a[i] === "--ws") opts.ws = a[++i];
    else if (a[i] === "--game-id") opts.gameId = a[++i];
    else if (a[i] === "--onnx") opts.onnx = a[++i];
  }
  return opts;
}

function heuristic(tick: number, gold: number): FactorizedAction {
  if (tick % 40 === 0) {
    return {
      actionType: ActionType.ATTACK,
      targetPlayer: tick < 200 ? 0 : 1,
      cellX: 16,
      cellY: 8,
      troopFrac: 2,
      buildType: 0,
    };
  }
  if (tick % 100 === 50 && gold > 200_000) {
    return {
      actionType: ActionType.BUILD,
      targetPlayer: 0,
      cellX: 8,
      cellY: 8,
      troopFrac: 0,
      buildType: 0,
    };
  }
  return {
    actionType: ActionType.NOOP,
    targetPlayer: 0,
    cellX: 0,
    cellY: 0,
    troopFrac: 0,
    buildType: 0,
  };
}

async function runDemo(opts: ReturnType<typeof parseArgs>) {
  console.debug = () => {};
  const env = new GameEnv({
    mapName: opts.map,
    macroStride: 5,
    maxTicks: opts.ticks,
  });
  let { info } = await env.reset(1);
  console.log("demo start", info);
  if (opts.onnx && fs.existsSync(opts.onnx)) {
    console.log(
      `ONNX checkpoint present at ${opts.onnx} — wire onnxruntime for live infer; using heuristic this run`,
    );
  }
  for (let t = 0; t < opts.ticks; t += 5) {
    const action = heuristic(info.tick, info.agentGold);
    const step = env.step(action);
    info = step.info;
    if (t % 50 === 0) {
      console.log(
        `t=${info.tick} tiles=${info.agentTiles}/${info.nationTiles} gold=${info.agentGold} reward=${info.reward.toFixed(1)}`,
      );
    }
    if (info.done) break;
  }
  console.log("demo final", env.snapshot());
}

async function runWs(opts: ReturnType<typeof parseArgs>) {
  if (!opts.ws || !opts.gameId) {
    throw new Error("--ws and --game-id required");
  }
  // Dynamic import so demo mode needs no ws dependency beyond Node built-ins
  const { default: WebSocket } = await import("ws");
  console.log(`Connecting ${opts.ws} game=${opts.gameId} (dev servers only)`);
  const ws = new WebSocket(opts.ws);
  await new Promise<void>((resolve, reject) => {
    ws.once("open", () => resolve());
    ws.once("error", reject);
  });
  ws.send(
    JSON.stringify({
      type: "join",
      gameID: opts.gameId,
      clientID: "ai-bot-client",
      token: "",
      username: "AIBot",
      clanTag: null,
      cosmetics: null,
      persistentID: "ai-bot-persistent",
    }),
  );

  // Mirror env for obs; live game still runs src/core on server/clients
  const mirror = new GameEnv({ mapName: opts.map, autoSpawn: false });
  await mirror.reset(1);

  ws.on("message", (data) => {
    const v = JSON.parse(data.toString());
    if (v.type === "turn") {
      const action = heuristic(v.turn?.turnNumber ?? 0, 0);
      // Emit attack intents only — decode to wire schema
      if (action.actionType === ActionType.ATTACK) {
        ws.send(
          JSON.stringify({
            type: "intent",
            intent: {
              type: "attack",
              targetID: action.targetPlayer === 0 ? null : null,
              troops: null,
            },
          }),
        );
      }
    } else if (v.type === "ping") {
      ws.send(JSON.stringify({ type: "ping" }));
    } else if (v.type === "error") {
      console.error("server error", v);
      ws.close();
    } else if (v.type === "start") {
      console.log("game started");
    }
  });
}

async function main() {
  const opts = parseArgs();
  if (opts.ws) {
    await runWs(opts);
    return;
  }
  opts.demo = true;
  await runDemo(opts);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
