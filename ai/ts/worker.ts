/**
 * Worker-thread RPC: one GameEnv per worker, JSON messages on parentPort.
 *
 * Protocol (parent → worker):
 *   { id, cmd: "reset", seed? }
 *   { id, cmd: "step", action: FactorizedAction }
 *   { id, cmd: "ping" }
 *
 * Response:
 *   { id, ok: true, result } | { id, ok: false, error }
 */
import { parentPort, workerData } from "worker_threads";
import { GameEnv } from "./env";
import { FactorizedAction, noopAction } from "./types";

const env = new GameEnv({
  mapName: workerData?.mapName ?? "plains",
  mapsDir: workerData?.mapsDir,
  difficulty: workerData?.difficulty,
  macroStride: workerData?.macroStride ?? 20,
  seed: workerData?.seed ?? 1,
});

type Req = {
  id: number;
  cmd: "reset" | "step" | "ping" | "snapshot";
  seed?: number;
  action?: FactorizedAction;
};

parentPort!.on("message", async (msg: Req) => {
  try {
    let result: unknown;
    switch (msg.cmd) {
      case "ping":
        result = { pong: true };
        break;
      case "reset":
        result = await env.reset(msg.seed);
        break;
      case "step":
        result = env.step(msg.action ?? noopAction());
        break;
      case "snapshot":
        result = env.snapshot();
        break;
      default:
        throw new Error(`unknown cmd`);
    }
    parentPort!.postMessage({ id: msg.id, ok: true, result });
  } catch (e) {
    parentPort!.postMessage({
      id: msg.id,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    });
  }
});
