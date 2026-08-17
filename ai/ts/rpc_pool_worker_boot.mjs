/**
 * Worker bootstrap using tsx's tsImport (no --import/--loader on the Worker).
 */
import { pathToFileURL } from "node:url";
import { tsImport } from "tsx/esm/api";

await tsImport("./rpc_pool_worker.ts", import.meta.url);
