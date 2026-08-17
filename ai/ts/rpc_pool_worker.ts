/**
 * Worker-thread GameEnv slot for rpc_pool.ts.
 * Receives {reqId, cmd, ...} and replies with {reqId, ok, result?, error?, frame?}.
 */
import { parentPort, workerData } from "worker_threads";
import { Difficulty, isDifficulty } from "../../src/core/game/Game";
import { GameEnv } from "./env";
import { FactorizedAction, noopAction } from "./types";
import { packObsMaskFrame, slimStepResult } from "./wire_codec";

type SlotConfig = {
  mapName: string;
  stride: number;
  difficulty: Difficulty;
  nations: number;
  bots: number;
  maxTicks: number;
};

const slot: number = workerData.slot;
let cfg: SlotConfig = { ...workerData.cfg };
let env = new GameEnv({
  mapName: cfg.mapName,
  macroStride: cfg.stride,
  difficulty: cfg.difficulty,
  nations: cfg.nations,
  bots: cfg.bots,
  maxTicks: cfg.maxTicks,
});

function remakeEnv(): void {
  env = new GameEnv({
    mapName: cfg.mapName,
    macroStride: cfg.stride,
    difficulty: cfg.difficulty,
    nations: cfg.nations,
    bots: cfg.bots,
    maxTicks: cfg.maxTicks,
  });
}

function withBinary(result: {
  obs: import("./types").Observation;
  mask: import("./types").ActionMask;
  info: unknown;
  timing?: { simMs?: number; obsMs?: number; maskMs?: number; serializeMs?: number };
}): { slim: ReturnType<typeof slimStepResult>; frame: ArrayBuffer } {
  const t0 = performance.now();
  const frame = packObsMaskFrame(result.obs, result.mask);
  const serializeMs = performance.now() - t0;
  const timing = { ...(result.timing ?? {}), serializeMs };
  return { slim: slimStepResult({ ...result, timing }), frame };
}

async function handle(msg: {
  reqId: number;
  cmd: string;
  seed?: number;
  action?: FactorizedAction;
  map?: string;
  nations?: number;
  bots?: number;
  difficulty?: string;
  stride?: number;
  maxTicks?: number;
}): Promise<{ result?: unknown; frame?: ArrayBuffer }> {
  switch (msg.cmd) {
    case "ping":
      return { result: { pong: true, slot } };
    case "configure": {
      let changed = false;
      if (msg.map !== undefined && msg.map !== cfg.mapName) {
        cfg.mapName = msg.map;
        changed = true;
      }
      if (msg.nations !== undefined && msg.nations !== cfg.nations) {
        cfg.nations = msg.nations;
        changed = true;
      }
      if (msg.bots !== undefined && msg.bots !== cfg.bots) {
        cfg.bots = msg.bots;
        changed = true;
      }
      if (msg.stride !== undefined && msg.stride !== cfg.stride) {
        cfg.stride = msg.stride;
        changed = true;
      }
      if (msg.maxTicks !== undefined && msg.maxTicks !== cfg.maxTicks) {
        cfg.maxTicks = msg.maxTicks;
        changed = true;
      }
      if (msg.difficulty !== undefined && isDifficulty(msg.difficulty)) {
        if (msg.difficulty !== cfg.difficulty) {
          cfg.difficulty = msg.difficulty;
          changed = true;
        }
      }
      if (changed) remakeEnv();
      return {
        result: {
          configured: true,
          slot,
          map: cfg.mapName,
          nations: cfg.nations,
          bots: cfg.bots,
          difficulty: cfg.difficulty,
        },
      };
    }
    case "reset": {
      if (msg.difficulty && isDifficulty(msg.difficulty)) {
        if (msg.difficulty !== cfg.difficulty) {
          cfg.difficulty = msg.difficulty;
          remakeEnv();
        }
      }
      const reset = await env.reset(msg.seed);
      const packed = withBinary(reset);
      return { result: packed.slim, frame: packed.frame };
    }
    case "step": {
      const stepped = env.step(msg.action ?? noopAction());
      const packed = withBinary(stepped);
      return { result: packed.slim, frame: packed.frame };
    }
    case "quit":
      return { result: "bye" };
    default:
      throw new Error(`unknown worker cmd ${msg.cmd}`);
  }
}

if (!parentPort) {
  throw new Error("rpc_pool_worker must run as a worker thread");
}

parentPort.on("message", (msg) => {
  void (async () => {
    try {
      const { result, frame } = await handle(msg);
      if (frame) {
        parentPort!.postMessage(
          { reqId: msg.reqId, ok: true, result, frame },
          [frame],
        );
      } else {
        parentPort!.postMessage({ reqId: msg.reqId, ok: true, result });
      }
    } catch (e) {
      parentPort!.postMessage({
        reqId: msg.reqId,
        ok: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  })();
});
