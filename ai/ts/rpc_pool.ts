/**
 * One Node process, N worker_threads each owning a GameEnv.
 * JSON control on stdin/stdout; optional binary obs/mask frames on OPENFRONT_OBS_FD.
 *
 * Usage:
 *   npx tsx ai/ts/rpc_pool.ts --slots 4 --map onion --nations 3 --bots 2
 *
 * Commands (JSON line):
 *   ping, configure, reset, step, reset_batch, step_batch, quit
 * Slot field optional for single-slot cmds (default 0). Batches use parallel workers.
 */
import fs from "fs";
import path from "path";
import { createInterface } from "readline";
import { fileURLToPath } from "url";
import { Worker } from "worker_threads";
import { Difficulty, isDifficulty } from "../../src/core/game/Game";
import { frameToJsonObsMask } from "./wire_codec";

const TS_DIR = path.dirname(fileURLToPath(import.meta.url));

type EnvConfig = {
  mapName: string;
  stride: number;
  difficulty: Difficulty;
  nations: number;
  bots: number;
  maxTicks: number;
};

function parseArgs(): { slots: number; cfg: EnvConfig } {
  const args = process.argv.slice(2);
  let slots = 1;
  let mapName = "onion";
  let stride = 5;
  let difficulty: Difficulty = Difficulty.Easy;
  let nations = 3;
  let bots = 2;
  let maxTicks = 30_000;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--slots") slots = parseInt(args[++i], 10);
    if (args[i] === "--map") mapName = args[++i];
    if (args[i] === "--stride") stride = parseInt(args[++i], 10);
    if (args[i] === "--nations") nations = parseInt(args[++i], 10);
    if (args[i] === "--bots") bots = parseInt(args[++i], 10);
    if (args[i] === "--max-ticks") maxTicks = parseInt(args[++i], 10);
    if (args[i] === "--difficulty") {
      const value = args[++i];
      if (!isDifficulty(value)) throw new Error(`invalid difficulty: ${value}`);
      difficulty = value;
    }
  }
  if (!Number.isFinite(slots) || slots < 1 || slots > 64) {
    throw new Error("--slots must be 1..64");
  }
  return {
    slots,
    cfg: { mapName, stride, difficulty, nations, bots, maxTicks },
  };
}

function openObsFd(): number | null {
  const raw = process.env.OPENFRONT_OBS_FD;
  if (!raw) return null;
  const fd = parseInt(raw, 10);
  if (!Number.isFinite(fd) || fd < 0) return null;
  return fd;
}

/** Prefer fd3; write JSON-slim first conceptually — caller writes JSON after.
 * Frame bytes are written here when binary fd is available.
 */
function materializeResult(
  obsFd: number | null,
  slim: unknown,
  frame: ArrayBuffer | undefined,
): { result: unknown; frame: ArrayBuffer | null } {
  if (frame && obsFd !== null) {
    return { result: slim, frame };
  }
  if (frame) {
    const { obs, mask } = frameToJsonObsMask(frame);
    const base =
      slim && typeof slim === "object" ? (slim as Record<string, unknown>) : {};
    return {
      result: { ...base, obs, mask, encoding: "f32b64" },
      frame: null,
    };
  }
  return { result: slim, frame: null };
}

function writeFrame(obsFd: number, frame: ArrayBuffer): void {
  const payload = Buffer.from(frame);
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32LE(payload.byteLength, 0);
  fs.writeSync(obsFd, header);
  fs.writeSync(obsFd, payload);
}

type Pending = {
  resolve: (v: {
    ok: boolean;
    result?: unknown;
    error?: string;
    frame?: ArrayBuffer;
  }) => void;
};

class SlotWorker {
  readonly slot: number;
  private worker: Worker;
  private nextReq = 1;
  private pending = new Map<number, Pending>();
  private cfg: EnvConfig;

  constructor(slot: number, cfg: EnvConfig) {
    this.slot = slot;
    this.cfg = { ...cfg };
    const workerPath = path.join(TS_DIR, "rpc_pool_worker_boot.mjs");
    this.worker = new Worker(workerPath, {
      workerData: { slot, cfg: this.cfg },
    });
    this.worker.on(
      "message",
      (msg: {
        reqId: number;
        ok: boolean;
        result?: unknown;
        error?: string;
        frame?: ArrayBuffer;
      }) => {
        const p = this.pending.get(msg.reqId);
        if (!p) return;
        this.pending.delete(msg.reqId);
        p.resolve(msg);
      },
    );
    this.worker.on("error", (err) => {
      this.failPending(err.message);
    });
    this.worker.on("exit", (code) => {
      this.failPending(`worker ${this.slot} exited ${code}`);
    });
  }

  private failPending(error: string): void {
    for (const [, p] of this.pending) {
      p.resolve({ ok: false, error });
    }
    this.pending.clear();
  }

  call(
    cmd: string,
    payload: Record<string, unknown> = {},
  ): Promise<{
    ok: boolean;
    result?: unknown;
    error?: string;
    frame?: ArrayBuffer;
  }> {
    const reqId = this.nextReq++;
    return new Promise((resolve) => {
      this.pending.set(reqId, { resolve });
      this.worker.postMessage({ reqId, cmd, ...payload });
    });
  }

  async terminate(): Promise<void> {
    await this.worker.terminate();
  }
}

async function main() {
  console.log = (...args: unknown[]) => console.error(...args);
  console.debug = () => {};
  console.info = (...args: unknown[]) => console.error(...args);
  console.warn = (...args: unknown[]) => console.error(...args);

  const { slots, cfg } = parseArgs();
  const obsFd = openObsFd();
  const workers = Array.from(
    { length: slots },
    (_, i) => new SlotWorker(i, cfg),
  );

  process.stdout.write(
    JSON.stringify({
      ready: true,
      pool: true,
      slots,
      binaryFd: obsFd !== null,
      map: cfg.mapName,
      stride: cfg.stride,
      difficulty: cfg.difficulty,
      nations: cfg.nations,
      bots: cfg.bots,
      maxTicks: cfg.maxTicks,
    }) + "\n",
  );

  const rl = createInterface({ input: process.stdin, terminal: false });

  const reply = (id: unknown, body: Record<string, unknown>) => {
    // The Python peer reads the JSON header before draining fd 3. Write it
    // synchronously before its large binary frame so pipe backpressure cannot
    // leave both sides waiting on each other.
    fs.writeSync(1, JSON.stringify({ id, ...body }) + "\n");
  };

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg: {
      id?: number;
      cmd: string;
      slot?: number;
      seed?: number;
      action?: unknown;
      actions?: unknown[];
      seeds?: number[];
      map?: string;
      nations?: number;
      bots?: number;
      difficulty?: string;
      stride?: number;
      maxTicks?: number;
    };
    try {
      msg = JSON.parse(line);
    } catch {
      reply(undefined, { ok: false, error: "bad json" });
      continue;
    }

    try {
      switch (msg.cmd) {
        case "ping": {
          const slot = msg.slot ?? 0;
          const r = await workers[slot].call("ping");
          if (!r.ok) throw new Error(r.error ?? "ping failed");
          reply(msg.id, { ok: true, result: r.result });
          break;
        }
        case "configure": {
          const targets =
            msg.slot === undefined ? workers : [workers[msg.slot ?? 0]];
          const payload = {
            map: msg.map,
            nations: msg.nations,
            bots: msg.bots,
            difficulty: msg.difficulty,
            stride: msg.stride,
            maxTicks: msg.maxTicks,
          };
          const results = await Promise.all(
            targets.map((w) => w.call("configure", payload)),
          );
          for (const r of results) {
            if (!r.ok) throw new Error(r.error ?? "configure failed");
          }
          reply(msg.id, {
            ok: true,
            result: {
              configured: true,
              slots: targets.map((w) => w.slot),
              map: msg.map ?? cfg.mapName,
              nations: msg.nations ?? cfg.nations,
              bots: msg.bots ?? cfg.bots,
              difficulty: msg.difficulty ?? cfg.difficulty,
            },
          });
          break;
        }
        case "reset": {
          const slot = msg.slot ?? 0;
          const r = await workers[slot].call("reset", {
            seed: msg.seed,
            difficulty: msg.difficulty,
          });
          if (!r.ok) throw new Error(r.error ?? "reset failed");
          const m = materializeResult(obsFd, r.result, r.frame);
          reply(msg.id, { ok: true, result: m.result });
          if (m.frame && obsFd !== null) writeFrame(obsFd, m.frame);
          break;
        }
        case "step": {
          const slot = msg.slot ?? 0;
          const r = await workers[slot].call("step", { action: msg.action });
          if (!r.ok) throw new Error(r.error ?? "step failed");
          const m = materializeResult(obsFd, r.result, r.frame);
          reply(msg.id, { ok: true, result: m.result });
          if (m.frame && obsFd !== null) writeFrame(obsFd, m.frame);
          break;
        }
        case "reset_batch": {
          const seeds = msg.seeds ?? [];
          if (seeds.length !== slots) {
            throw new Error(`reset_batch expects ${slots} seeds`);
          }
          const settled = await Promise.all(
            workers.map((w, i) =>
              w.call("reset", { seed: seeds[i], difficulty: msg.difficulty }),
            ),
          );
          const results = [];
          const frames: ArrayBuffer[] = [];
          for (let i = 0; i < settled.length; i++) {
            const r = settled[i];
            if (!r.ok) throw new Error(r.error ?? `reset slot ${i} failed`);
            const m = materializeResult(obsFd, r.result, r.frame);
            results.push(m.result);
            if (m.frame) frames.push(m.frame);
          }
          reply(msg.id, { ok: true, results });
          if (obsFd !== null) {
            for (const frame of frames) writeFrame(obsFd, frame);
          }
          break;
        }
        case "step_batch": {
          const actions = msg.actions ?? [];
          if (actions.length !== slots) {
            throw new Error(`step_batch expects ${slots} actions`);
          }
          const settled = await Promise.all(
            workers.map((w, i) => w.call("step", { action: actions[i] })),
          );
          const results = [];
          const frames: ArrayBuffer[] = [];
          for (let i = 0; i < settled.length; i++) {
            const r = settled[i];
            if (!r.ok) throw new Error(r.error ?? `step slot ${i} failed`);
            const m = materializeResult(obsFd, r.result, r.frame);
            results.push(m.result);
            if (m.frame) frames.push(m.frame);
          }
          reply(msg.id, { ok: true, results });
          if (obsFd !== null) {
            for (const frame of frames) writeFrame(obsFd, frame);
          }
          break;
        }
        case "quit": {
          reply(msg.id, { ok: true, result: "bye" });
          await Promise.all(workers.map((w) => w.terminate()));
          process.exit(0);
          break;
        }
        default:
          throw new Error(`unknown cmd ${msg.cmd}`);
      }
    } catch (e) {
      reply(msg.id, {
        ok: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
