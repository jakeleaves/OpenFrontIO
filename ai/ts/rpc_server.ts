/**
 * Stdin/stdout JSON-RPC gateway for Python trainers and the live visualizer.
 * One line in → one line out.
 *
 * Commands: reset, step, ping, snapshot, map_frame, expert_step,
 *           viz_reset, viz_step, quit.
 *
 * Usage: npx tsx ai/ts/rpc_server.ts --map world --stride 5 --nations 24 --bots 48
 */
import { createInterface } from "readline";
import { Difficulty, isDifficulty } from "../../src/core/game/Game";
import { GameEnv } from "./env";
import { expertDecision } from "./expert";
import { obsToJson } from "./obs";
import { ActionMask, FactorizedAction, Observation, noopAction } from "./types";

type EnvConfig = {
  mapName: string;
  stride: number;
  difficulty: Difficulty;
  nations: number;
  bots: number;
  maxTicks: number;
  lambdaW?: number;
};

function parseArgs(): EnvConfig {
  const args = process.argv.slice(2);
  let mapName = "world";
  let stride = 10;
  let difficulty: Difficulty = Difficulty.Impossible;
  let nations = 72;
  let bots = 0;
  let maxTicks = 30_000;
  let lambdaW: number | undefined;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--map") mapName = args[++i];
    if (args[i] === "--stride") stride = parseInt(args[++i], 10);
    if (args[i] === "--nations") nations = parseInt(args[++i], 10);
    if (args[i] === "--bots") bots = parseInt(args[++i], 10);
    if (args[i] === "--max-ticks") maxTicks = parseInt(args[++i], 10);
    if (args[i] === "--difficulty") {
      const value = args[++i];
      if (!isDifficulty(value)) {
        throw new Error(`invalid difficulty: ${value}`);
      }
      difficulty = value;
    }
    if (args[i] === "--lambda-w") {
      lambdaW = parseFloat(args[++i]);
      if (!Number.isFinite(lambdaW)) {
        throw new Error("--lambda-w must be numeric");
      }
    }
  }
  return { mapName, stride, difficulty, nations, bots, maxTicks, lambdaW };
}

function makeEnv(cfg: EnvConfig): GameEnv {
  return new GameEnv({
    mapName: cfg.mapName,
    macroStride: cfg.stride,
    difficulty: cfg.difficulty,
    nations: cfg.nations,
    bots: cfg.bots,
    maxTicks: cfg.maxTicks,
    reward: cfg.lambdaW === undefined ? undefined : { lambdaW: cfg.lambdaW },
  });
}

function wireResult(
  result: {
    obs: Observation;
    mask: ActionMask;
    info: unknown;
    timing?: unknown;
    [key: string]: unknown;
  },
  opts: { includeObs?: boolean } = {},
) {
  const includeObs = opts.includeObs !== false;
  const t0 = performance.now();
  const obs = includeObs ? obsToJson(result.obs) : null;
  const serializeMs = performance.now() - t0;
  const timing =
    result.timing && typeof result.timing === "object"
      ? { ...(result.timing as object), serializeMs }
      : { serializeMs };
  const { obs: _rawObs, ...rest } = result;
  return { ...rest, obs, timing };
}

async function main() {
  // Keep stdout JSON-only for the Python RPC client.
  console.log = (...args: unknown[]) => console.error(...args);
  console.debug = () => {};
  console.info = (...args: unknown[]) => console.error(...args);
  console.warn = (...args: unknown[]) => console.error(...args);

  const envConfig = parseArgs();
  let env = makeEnv(envConfig);
  const rl = createInterface({ input: process.stdin, terminal: false });

  process.stdout.write(
    JSON.stringify({
      ready: true,
      map: envConfig.mapName,
      stride: envConfig.stride,
      difficulty: envConfig.difficulty,
      nations: envConfig.nations,
      bots: envConfig.bots,
      maxTicks: envConfig.maxTicks,
    }) + "\n",
  );

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg: {
      id?: number;
      cmd: string;
      seed?: number;
      action?: FactorizedAction;
      difficulty?: string;
      includeMap?: boolean;
      includeObs?: boolean;
      /** Use a cached geometry-only boat mask for display-policy playback. */
      liteMask?: boolean;
      useExpert?: boolean;
      maxDim?: number;
    };
    try {
      msg = JSON.parse(line);
    } catch {
      process.stdout.write(
        JSON.stringify({ ok: false, error: "bad json" }) + "\n",
      );
      continue;
    }
    try {
      let result: unknown;
      switch (msg.cmd) {
        case "ping":
          result = { pong: true };
          break;
        case "reset":
          if (msg.difficulty && isDifficulty(msg.difficulty)) {
            if (msg.difficulty !== envConfig.difficulty) {
              envConfig.difficulty = msg.difficulty;
              env = makeEnv(envConfig);
            }
          }
          result = wireResult(await env.reset(msg.seed));
          break;
        case "viz_reset": {
          if (msg.difficulty && isDifficulty(msg.difficulty)) {
            if (msg.difficulty !== envConfig.difficulty) {
              envConfig.difficulty = msg.difficulty;
              env = makeEnv(envConfig);
            }
          }
          const reset = await env.reset(msg.seed, {
            maskOptions: {
              boatMaskMode: "geometry",
              boatMaskRefreshTicks: 100,
            },
          });
          result = {
            ...wireResult(reset),
            snapshot: env.snapshot(),
            map: env.mapFrame(msg.maxDim ?? 256),
          };
          break;
        }
        case "step":
          result = wireResult(env.step(msg.action ?? noopAction()));
          break;
        case "viz_step": {
          let action = msg.action ?? noopAction();
          if (msg.useExpert) {
            const decision = expertDecision(env.getGame(), env.getAgent(), {
              opponentIds: env.getOpponentIds(),
            });
            action = decision.action;
          }
          // Display path: skip obs/mask unless the python policy needs them.
          // Python viz keeps boats via a loose, cached coastal-geometry mask
          // and a budgeted in-cell boat decode (no full-map water search).
          // Training and normal RPC retain the verified mask/decode.
          const includeObs = msg.includeObs !== false;
          const stepped = env.step(action, {
            encodeObs: includeObs,
            encodeMask: includeObs,
            maskOptions: msg.liteMask
              ? {
                  boatMaskMode: "geometry",
                  boatMaskRefreshTicks: 100,
                }
              : undefined,
            decodeOptions: msg.liteMask
              ? { boatDecodeMode: "cheap" }
              : undefined,
          });
          result = {
            ...wireResult(stepped, { includeObs }),
            action,
            snapshot: env.snapshot(),
            map: msg.includeMap ? env.mapFrame(msg.maxDim ?? 256) : null,
          };
          break;
        }
        case "expert_step": {
          const decision = expertDecision(env.getGame(), env.getAgent(), {
            opponentIds: env.getOpponentIds(),
          });
          const stepped = env.step(decision.action);
          result = { ...wireResult(stepped), ...decision };
          break;
        }
        case "snapshot":
          result = env.snapshot();
          break;
        case "map_frame":
          result = env.mapFrame(msg.maxDim ?? 500);
          break;
        case "quit":
          process.stdout.write(
            JSON.stringify({ id: msg.id, ok: true, result: "bye" }) + "\n",
          );
          process.exit(0);
          break;
        default:
          throw new Error(`unknown cmd ${msg.cmd}`);
      }
      process.stdout.write(
        JSON.stringify({ id: msg.id, ok: true, result }) + "\n",
      );
    } catch (e) {
      process.stdout.write(
        JSON.stringify({
          id: msg.id,
          ok: false,
          error: e instanceof Error ? e.message : String(e),
        }) + "\n",
      );
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
