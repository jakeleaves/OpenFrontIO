/**
 * Stdin/stdout JSON-RPC gateway for Python trainers.
 * One line in → one line out. Commands: reset, step, ping, quit.
 *
 * Usage: npx tsx ai/ts/rpc_server.ts --map plains --stride 20
 */
import { createInterface } from "readline";
import { GameEnv } from "./env";
import { FactorizedAction, noopAction } from "./types";

function parseArgs() {
  const args = process.argv.slice(2);
  let mapName = "plains";
  let stride = 20;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--map") mapName = args[++i];
    if (args[i] === "--stride") stride = parseInt(args[++i], 10);
  }
  return { mapName, stride };
}

async function main() {
  // Keep stdout JSON-only for the Python RPC client.
  console.log = (...args: unknown[]) => console.error(...args);
  console.debug = () => {};
  console.info = (...args: unknown[]) => console.error(...args);
  console.warn = (...args: unknown[]) => console.error(...args);

  const { mapName, stride } = parseArgs();
  const env = new GameEnv({ mapName, macroStride: stride });
  const rl = createInterface({ input: process.stdin, terminal: false });

  // Ready signal for Python
  process.stdout.write(JSON.stringify({ ready: true, map: mapName }) + "\n");

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg: {
      id?: number;
      cmd: string;
      seed?: number;
      action?: FactorizedAction;
    };
    try {
      msg = JSON.parse(line);
    } catch (e) {
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
          result = await env.reset(msg.seed);
          break;
        case "step":
          result = env.step(msg.action ?? noopAction());
          break;
        case "snapshot":
          result = env.snapshot();
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
