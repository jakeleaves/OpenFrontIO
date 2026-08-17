/**
 * Child-process GameEnv bridge for the live visualizer.
 *
 * Keeps heavy reset/step/map work off the viz HTTP thread so GET /state
 * stays responsive while the sim is busy.
 */
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import type { Difficulty } from "../../src/core/game/Game";
import type { ObsJson } from "./policy";
import type { ActionMask, FactorizedAction, StepInfo } from "./types";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

export type VizMapFrame = {
  w: number;
  h: number;
  cells: string;
  deltas?: { i: number; c: number }[] | null;
  units: {
    x: number;
    y: number;
    kind: string;
    owner: "agent" | "nation";
    nationIndex?: number;
  }[];
  attacks: { agent: number; nation: number };
  fullW: number;
  fullH: number;
  stride: number;
  nationCount: number;
};

export type VizSnapshot = {
  tick: number;
  agent: {
    tiles: number;
    troops: number;
    gold: number;
    cities?: number;
    boats?: number;
  };
  enemies: {
    alive: number;
    tiles: number;
    troops: number;
    strongestTiles: number;
    strongestTroops: number;
  };
  placement: number;
  winner: string | null;
};

export type VizBundle = {
  obs: ObsJson | null;
  mask: ActionMask;
  info: StepInfo;
  snapshot: VizSnapshot;
  map: VizMapFrame | null;
  action?: FactorizedAction;
};

type Pending = {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
};

export type VizEnvRpcOpts = {
  mapName: string;
  stride: number;
  nations: number;
  bots: number;
  difficulty?: Difficulty;
  maxTicks?: number;
};

export class VizEnvRpc {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private buf = "";
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private ready: Promise<void> | null = null;

  constructor(private opts: VizEnvRpcOpts) {}

  async start(): Promise<void> {
    if (this.proc) return;
    const args = [
      "tsx",
      "ai/ts/rpc_server.ts",
      "--map",
      this.opts.mapName,
      "--stride",
      String(this.opts.stride),
      "--nations",
      String(this.opts.nations),
      "--bots",
      String(this.opts.bots),
      "--max-ticks",
      String(this.opts.maxTicks ?? 30_000),
    ];
    if (this.opts.difficulty) {
      args.push("--difficulty", this.opts.difficulty);
    }

    this.proc = spawn("npx", args, {
      cwd: ROOT,
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });

    this.ready = new Promise<void>((resolve, reject) => {
      const onData = (chunk: Buffer) => {
        this.buf += chunk.toString();
        let idx: number;
        while ((idx = this.buf.indexOf("\n")) >= 0) {
          const line = this.buf.slice(0, idx);
          this.buf = this.buf.slice(idx + 1);
          if (!line.trim()) continue;
          let msg: {
            ready?: boolean;
            id?: number;
            ok?: boolean;
            result?: unknown;
            error?: string;
          };
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          if (msg.ready) {
            resolve();
            continue;
          }
          if (msg.id === undefined) continue;
          const p = this.pending.get(msg.id);
          if (!p) continue;
          this.pending.delete(msg.id);
          if (msg.ok) p.resolve(msg.result);
          else p.reject(new Error(msg.error ?? "rpc error"));
        }
      };
      this.proc!.stdout.on("data", onData);
      this.proc!.stderr.on("data", (d) => {
        const text = String(d);
        if (/error/i.test(text)) console.error(text.trimEnd());
      });
      this.proc!.on("exit", (code) => {
        const err = new Error(`viz env rpc exited ${code}`);
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
        reject(err);
      });
    });

    await this.ready;
  }

  private call(cmd: string, extra: Record<string, unknown> = {}) {
    if (!this.proc?.stdin.writable) {
      return Promise.reject(new Error("viz env rpc not started"));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc!.stdin.write(JSON.stringify({ id, cmd, ...extra }) + "\n");
    });
  }

  async vizReset(seed: number, difficulty: Difficulty, maxDim = 500) {
    return this.call("viz_reset", {
      seed,
      difficulty,
      maxDim,
    }) as Promise<VizBundle>;
  }

  async vizStep(opts: {
    action?: FactorizedAction;
    useExpert?: boolean;
    includeMap?: boolean;
    /** When false, skip obs encode + ~524KB JSON (heuristic display path). */
    includeObs?: boolean;
    /** Use a cached geometry-only boat mask for responsive policy playback. */
    liteMask?: boolean;
    maxDim?: number;
  }) {
    return this.call("viz_step", {
      action: opts.action,
      useExpert: Boolean(opts.useExpert),
      includeMap: Boolean(opts.includeMap),
      includeObs: opts.includeObs !== false,
      liteMask: Boolean(opts.liteMask),
      maxDim: opts.maxDim ?? 256,
    }) as Promise<VizBundle>;
  }

  async close(): Promise<void> {
    if (!this.proc) return;
    try {
      await this.call("quit");
    } catch {
      /* ignore */
    }
    this.proc.kill();
    this.proc = null;
  }
}
