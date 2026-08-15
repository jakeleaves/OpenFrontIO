/**
 * Pool of GameEnv processes via ai/ts/rpc_server.ts (one child per env).
 */
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import { FactorizedAction, noopAction } from "./types";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

type Pending = {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
};

class RpcChild {
  private proc: ChildProcessWithoutNullStreams;
  private buf = "";
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private ready: Promise<void>;

  constructor(
    private mapName: string,
    private stride: number,
  ) {
    this.proc = spawn(
      "npx",
      ["tsx", "ai/ts/rpc_server.ts", "--map", mapName, "--stride", String(stride)],
      { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] },
    );
    this.ready = new Promise((resolve, reject) => {
      const onData = (chunk: Buffer) => {
        this.buf += chunk.toString();
        let idx;
        while ((idx = this.buf.indexOf("\n")) >= 0) {
          const line = this.buf.slice(0, idx);
          this.buf = this.buf.slice(idx + 1);
          if (!line.trim()) continue;
          let msg: any;
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          if (msg.ready) {
            resolve();
            continue;
          }
          const p = this.pending.get(msg.id);
          if (!p) continue;
          this.pending.delete(msg.id);
          if (msg.ok) p.resolve(msg.result);
          else p.reject(new Error(msg.error ?? "rpc error"));
        }
      };
      this.proc.stdout.on("data", onData);
      this.proc.stderr.on("data", (d) => {
        // ignore debug noise
        if (String(d).includes("Error")) console.error(String(d));
      });
      this.proc.on("exit", (code) => {
        reject(new Error(`rpc exited ${code}`));
      });
    });
  }

  async waitReady() {
    await this.ready;
  }

  call(cmd: string, extra: Record<string, unknown> = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(JSON.stringify({ id, cmd, ...extra }) + "\n");
    });
  }

  async close() {
    try {
      await this.call("quit");
    } catch {
      /* ignore */
    }
    this.proc.kill();
  }
}

export class EnvWorkerPool {
  private children: RpcChild[] = [];
  private rr = 0;

  constructor(
    private n: number,
    private opts: { mapName?: string; macroStride?: number } = {},
  ) {}

  async start() {
    for (let i = 0; i < this.n; i++) {
      const c = new RpcChild(
        this.opts.mapName ?? "plains",
        this.opts.macroStride ?? 20,
      );
      await c.waitReady();
      this.children.push(c);
    }
  }

  async reset(workerIdx?: number, seed?: number) {
    const i = workerIdx ?? this.rr++ % this.n;
    return this.children[i].call("reset", { seed }) as Promise<{
      obs: unknown;
      mask: unknown;
      info: unknown;
    }>;
  }

  async step(workerIdx: number, action: FactorizedAction = noopAction()) {
    return this.children[workerIdx].call("step", { action }) as Promise<{
      obs: unknown;
      mask: unknown;
      info: unknown;
    }>;
  }

  size() {
    return this.n;
  }

  async close() {
    await Promise.all(this.children.map((c) => c.close()));
    this.children = [];
  }
}
