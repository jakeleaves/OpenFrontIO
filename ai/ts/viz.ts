/**
 * Live UI for GameEnv: Agent vs all World nations (Solo FFA).
 *
 *   npx tsx ai/ts/viz.ts --port 9102
 *   npx tsx ai/ts/viz.ts --policy python --infer http://127.0.0.1:9101 --stride 10
 *
 * Open http://localhost:9102
 */
import fs from "fs";
import http from "http";
import path from "path";
import { Difficulty, isDifficulty } from "../../src/core/game/Game";
import { obsToJson } from "./obs";
import {
  ACTION_NAMES,
  describeAction,
  heuristicPolicy,
  PolicyHealth,
  PolicyProvider,
  pythonCkptPolicy,
} from "./policy";
import { ActionMask, FactorizedAction, noopAction, StepInfo } from "./types";
import { VizEnvRpc, VizMapFrame, VizSnapshot } from "./viz_env_rpc";

function argValue(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const PORT = parseInt(argValue("--port", "9100"), 10);
const STRIDE = parseInt(argValue("--stride", "5"), 10);
const MAP_NAME = argValue("--map", "world");
const INITIAL_STEP_MS = parseInt(argValue("--step-ms", "150"), 10);
const NATIONS = parseInt(argValue("--nations", "12"), 10);
const BOTS = parseInt(argValue("--bots", "12"), 10);
/** Rebuild map cells every N macro-steps (1 = every step). Stats still update every step. */
const MAP_EVERY = Math.max(1, parseInt(argValue("--map-every", "8"), 10) || 8);
const POLICY_NAME = argValue("--policy", "heuristic"); // heuristic | python
const INFER_URL = argValue("--infer", "http://127.0.0.1:9101");
const INFER_TIMEOUT_MS = Math.max(
  100,
  parseInt(argValue("--infer-timeout-ms", "5000"), 10) || 5_000,
);
const METRICS_PATH = path.resolve(
  argValue("--metrics", "ai/fixtures/checkpoints/training_metrics.jsonl"),
);

function makePolicy(): PolicyProvider {
  if (POLICY_NAME === "python") {
    return pythonCkptPolicy(INFER_URL, INFER_TIMEOUT_MS);
  }
  return heuristicPolicy();
}

const policy = makePolicy();
const envRpc = new VizEnvRpc({
  mapName: MAP_NAME,
  stride: STRIDE,
  nations: NATIONS,
  bots: BOTS,
  maxTicks: 30_000,
});

type Frame = {
  tick: number;
  done: boolean;
  winner: string | null;
  reward: number;
  cumulativeReward: number;
  action: string;
  target: string;
  troopFrac: number;
  troopRatio: number;
  troopCap: number;
  growthEfficiency: number;
  optimalGrowthRatio: number;
  legalActions: string[];
  policyName: string;
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
  attacks: { agent: number; nation: number };
  units: {
    x: number;
    y: number;
    kind: string;
    owner: "agent" | "nation";
    nationIndex?: number;
  }[];
  w: number;
  h: number;
  cells: string;
  seed: number;
  episode: number;
  stride: number;
  difficulty: Difficulty;
  fullW?: number;
  fullH?: number;
  mapStride?: number;
  nationCount?: number;
  inference?: PolicyHealth;
  stepDurationMs: number | null;
  stepError: string | null;
  simTick: number;
  achievedStepsPerSec: number;
};

let seed = 1;
let episode = 1;
let paused = false;
let stepMs = Number.isFinite(INITIAL_STEP_MS)
  ? Math.max(20, INITIAL_STEP_MS)
  : 80;
let lastActionLabel = "SPAWN";
let lastTarget = "unowned";
let lastTroopFrac = 0.1;
let lastLegal: string[] = [];
let cumulativeReward = 0;
let resetting = false;
let stepping = false;
let frame: Frame;
let history: {
  tick: number;
  agent: number;
  enemies: number;
  reward: number;
}[] = [];
let lastStepAt = 0;
let lastStepDurationMs: number | null = null;
let lastStepError: string | null = null;
/** Monotonic sim frame counter for SSE clients. */
let simTick = 0;
let stepsCompleted = 0;
let stepsWindowStart = Date.now();
let achievedStepsPerSec = 0;
type SseClient = { res: http.ServerResponse; id: number };
const sseClients = new Map<number, SseClient>();
let nextSseId = 1;
let lastObs = obsToJson({
  global: new Float32Array(1),
  local: new Float32Array(1),
  vector: new Float32Array(1),
});
let lastMask: ActionMask = {
  actionType: [],
  targetPlayer: [],
  cell: [],
  troopFrac: [],
  buildType: [],
};

function legalActionNames(mask: ActionMask): string[] {
  return (mask.actionType ?? [])
    .map((v, i) => (v ? (ACTION_NAMES[i] ?? String(i)) : null))
    .filter((x): x is string => Boolean(x));
}

type TrainingMetric = {
  ppo_step: number;
  loss: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  difficulty?: string;
  curriculum?: boolean;
  rollout?: {
    wins?: number;
    losses?: number;
    win_rate?: number;
    episodes_finished?: number;
  };
  eval: {
    mean_tiles: number;
    mean_troop_ratio: number;
    mean_return: number;
    win_rate?: number;
    wins?: number;
    losses?: number;
    mean_ticks?: number;
  } | null;
};

let liveWins = 0;
let liveLosses = 0;
let liveDifficulty: Difficulty = Difficulty.Easy;
/** Pre-serialized GET /state body — HTTP never waits on step/mapFrame/metrics. */
let lastStateJson = "{}";
let mapStepsSinceRefresh = 0;
let cachedMap: VizMapFrame | null = null;
let cachedMetrics: TrainingMetric[] = [];
let cachedMetricsAt = 0;
const METRICS_TTL_MS = 2000;

function trainingMetrics(): TrainingMetric[] {
  const now = Date.now();
  if (now - cachedMetricsAt < METRICS_TTL_MS && cachedMetrics.length) {
    return cachedMetrics;
  }
  try {
    cachedMetrics = fs
      .readFileSync(METRICS_PATH, "utf8")
      .trim()
      .split("\n")
      .filter(Boolean)
      .slice(-60)
      .flatMap((line) => {
        try {
          return [JSON.parse(line) as TrainingMetric];
        } catch {
          return [];
        }
      });
    cachedMetricsAt = now;
    return cachedMetrics;
  } catch {
    return cachedMetrics;
  }
}

function publishStateCache() {
  lastStateJson = JSON.stringify({
    frame,
    history,
    metrics: trainingMetrics(),
    episodes: {
      wins: liveWins,
      losses: liveLosses,
      total: liveWins + liveLosses,
      winRate:
        liveWins + liveLosses > 0 ? liveWins / (liveWins + liveLosses) : 0,
    },
    paused,
    stepMs,
    policy: policy.name,
    simTick,
    achievedStepsPerSec,
  });
  const payload = `id: ${simTick}\ndata: ${lastStateJson}\n\n`;
  for (const client of sseClients.values()) {
    try {
      client.res.write(payload);
    } catch {
      sseClients.delete(client.id);
    }
  }
}

function packFrame(
  info: StepInfo,
  snap: VizSnapshot,
  mapUpdate: VizMapFrame | null,
): Frame {
  if (mapUpdate) {
    cachedMap = mapUpdate;
    mapStepsSinceRefresh = 1;
  } else {
    mapStepsSinceRefresh += 1;
  }
  const map = cachedMap;
  if (!map) {
    throw new Error("viz map cache empty");
  }
  return {
    tick: info.tick,
    done: info.done,
    winner: info.winner,
    reward: info.reward,
    cumulativeReward,
    action: lastActionLabel,
    target: lastTarget,
    troopFrac: lastTroopFrac,
    troopRatio: info.troopRatio,
    troopCap: info.troopCap,
    growthEfficiency: info.growthEfficiency,
    optimalGrowthRatio: info.optimalGrowthRatio,
    legalActions: lastLegal,
    policyName: policy.name,
    agent: snap.agent,
    enemies: snap.enemies,
    placement: snap.placement,
    attacks: map.attacks,
    units: map.units,
    w: map.w,
    h: map.h,
    cells: map.cells,
    seed,
    episode,
    stride: STRIDE,
    difficulty: liveDifficulty,
    fullW: map.fullW,
    fullH: map.fullH,
    mapStride: map.stride,
    nationCount: map.nationCount,
    inference: policy.health?.(),
    stepDurationMs: lastStepDurationMs,
    stepError: lastStepError,
    simTick,
    achievedStepsPerSec,
  };
}

function shouldIncludeMap(): boolean {
  return cachedMap === null || mapStepsSinceRefresh >= MAP_EVERY;
}

async function resetEnv(nextSeed?: number) {
  seed = nextSeed ?? seed + 1;
  const metrics = trainingMetrics();
  const latestDifficulty = metrics.at(-1)?.difficulty;
  liveDifficulty = isDifficulty(latestDifficulty)
    ? latestDifficulty
    : Difficulty.Easy;
  const bundle = await envRpc.vizReset(seed, liveDifficulty, 256);
  if (bundle.obs) lastObs = bundle.obs;
  lastMask = bundle.mask;
  lastActionLabel = "SPAWN";
  lastTarget = "unowned";
  lastTroopFrac = 0.1;
  lastLegal = legalActionNames(bundle.mask);
  cumulativeReward = 0;
  history = [];
  lastStepDurationMs = null;
  lastStepError = null;
  cachedMap = null;
  mapStepsSinceRefresh = 0;
  frame = packFrame(bundle.info, bundle.snapshot, bundle.map);
  history.push({
    tick: bundle.info.tick,
    agent: bundle.info.agentTiles,
    enemies: bundle.info.enemyTilesTotal,
    reward: 0,
  });
  simTick += 1;
  publishStateCache();
  if (policy.reset) await policy.reset();
}

async function stepOnce() {
  if (paused || resetting || stepping) return;
  if (frame.done) {
    if (frame.winner === "agent") liveWins += 1;
    else liveLosses += 1;
    episode += 1;
    resetting = true;
    try {
      await resetEnv();
    } finally {
      resetting = false;
    }
    return;
  }

  stepping = true;
  const startedAt = performance.now();
  try {
    lastLegal = legalActionNames(lastMask);
    const includeMap = shouldIncludeMap();
    let action: FactorizedAction;
    let bundle;

    if (POLICY_NAME === "heuristic") {
      // Expert runs inside rpc_server — no policy obs round-trip needed.
      bundle = await envRpc.vizStep({
        useExpert: true,
        includeMap,
        includeObs: false,
        maxDim: 256,
      });
      action = bundle.action ?? noopAction();
    } else {
      action = await policy.act({
        tick: frame.tick,
        gold: frame.agent.gold,
        troopRatio: frame.troopRatio,
        obs: lastObs,
        mask: lastMask,
      });
      bundle = await envRpc.vizStep({
        action,
        includeMap,
        includeObs: true,
        // Use cached coastal geometry for boat legality during playback.
        // Selected BOAT actions use a budgeted in-cell decode (no map-wide search).
        liteMask: true,
        maxDim: 256,
      });
    }

    const desc = describeAction(action);
    lastActionLabel = desc.action;
    lastTarget = desc.target;
    lastTroopFrac = desc.troopFrac;
    if (bundle.obs) lastObs = bundle.obs;
    if (bundle.mask?.actionType?.length) {
      lastMask = bundle.mask;
      lastLegal = legalActionNames(bundle.mask);
    }

    cumulativeReward += bundle.info.reward;
    lastStepDurationMs = performance.now() - startedAt;
    lastStepError = null;
    simTick += 1;
    stepsCompleted += 1;
    const elapsed = (Date.now() - stepsWindowStart) / 1000;
    if (elapsed >= 1) {
      achievedStepsPerSec = stepsCompleted / elapsed;
      stepsCompleted = 0;
      stepsWindowStart = Date.now();
    }
    frame = packFrame(bundle.info, bundle.snapshot, bundle.map);
    if (bundle.info.tick % 10 === 0 || bundle.info.done) {
      history.push({
        tick: bundle.info.tick,
        agent: bundle.info.agentTiles,
        enemies: bundle.info.enemyTilesTotal,
        reward: Number(cumulativeReward.toFixed(1)),
      });
      if (history.length > 240) history.shift();
    }
    publishStateCache();
  } catch (error) {
    lastStepDurationMs = performance.now() - startedAt;
    lastStepError = error instanceof Error ? error.message : String(error);
    publishStateCache();
    console.error("Viz step failed:", error);
  } finally {
    stepping = false;
  }
}

const PAGE = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>OpenFront AI — ${MAP_NAME} Solo FFA</title>
  <style>
    :root {
      --bg: #0e1218;
      --panel: #161c26;
      --line: #2a3444;
      --text: #e8eef4;
      --muted: #8b9bb0;
      --agent: #4aa3ff;
      --nation: #e85d4c;
      --land: #c9b896;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; height: 100%; overflow: hidden;
      background: var(--bg); color: var(--text);
      font: 12px/1.3 ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .wrap {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      height: 100%;
      min-height: 0;
    }
    .stage {
      display: flex; flex-direction: column; min-width: 0; min-height: 0;
      padding: 10px 12px 10px 14px;
    }
    h1 { font-size: 14px; font-weight: 600; margin: 0 0 2px; letter-spacing: 0.02em; }
    .sub { color: var(--muted); margin: 0 0 8px; font-size: 11px; }
    .map-wrap {
      flex: 1; min-height: 0;
      display: flex; align-items: center; justify-content: center;
      background: var(--panel); border: 1px solid var(--line);
      overflow: hidden;
    }
    .training {
      flex: 0 0 118px;
      margin-top: 8px;
      padding: 8px 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      display: grid;
      grid-template-columns: 240px 150px minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
    }
    .training-title { color: var(--text); font-size: 11px; font-weight: 600; }
    .training-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 14px; margin-top: 5px; }
    .metric { color: var(--muted); font-size: 10px; }
    .metric b { color: var(--text); font-weight: 600; margin-left: 4px; }
    .metric-chart { width: 100%; height: 72px; }
    .chart-label { color: var(--muted); font-size: 10px; margin-bottom: 2px; }
    canvas {
      image-rendering: pixelated;
      width: 100%;
      height: 100%;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: #0b1016;
    }
    .side {
      border-left: 1px solid var(--line);
      padding: 10px 12px;
      overflow: hidden;
      display: flex; flex-direction: column; gap: 6px;
      min-height: 0;
    }
    .row { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
    .k { color: var(--muted); flex-shrink: 0; }
    .v { font-variant-numeric: tabular-nums; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px; }
    .bar { height: 6px; background: #243044; margin-top: 4px; display: flex; flex-shrink: 0; }
    .bar > i { display: block; height: 100%; }
    .controls { display: flex; gap: 6px; flex-wrap: wrap; flex-shrink: 0; }
    button {
      background: #243044; color: var(--text); border: 1px solid var(--line);
      padding: 4px 8px; cursor: pointer; font: inherit;
    }
    button:hover { background: #2c3a50; }
    button.primary { background: #1e3a5f; border-color: #3d6ea3; }
    .pill {
      display: inline-block; padding: 0 6px; border: 1px solid var(--line);
      color: var(--muted); font-size: 10px; letter-spacing: 0.04em;
    }
    .pill.on { color: #86efac; border-color: #3f6d4e; }
    .swatch { display: inline-block; width: 8px; height: 8px; margin-right: 5px; vertical-align: middle; }
    .legend { display: flex; flex-direction: column; gap: 2px; color: var(--muted); font-size: 11px; flex-shrink: 0; }
    .spark { width: 100%; height: 48px; flex-shrink: 0; }
    hr { border: 0; border-top: 1px solid var(--line); margin: 2px 0; flex-shrink: 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10px; }
    .block { flex-shrink: 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="stage">
      <h1>OpenFront Dominant Agent</h1>
      <p class="sub" id="liveMatch">Live exhibition · ${MAP_NAME} Solo FFA · offline GameEnv</p>
      <div class="map-wrap"><canvas id="map" width="100" height="100"></canvas></div>
      <section class="training">
        <div>
          <div class="training-title">Training dashboard</div>
          <div class="training-grid">
            <span class="metric">PPO step <b id="ppoStep">—</b></span>
            <span class="metric">Difficulty <b id="difficulty">—</b></span>
            <span class="metric">Eval win rate <b id="evalWinRate">—</b></span>
            <span class="metric">Live W/L <b id="liveWL">0/0</b></span>
            <span class="metric">Entropy <b id="entropy">—</b></span>
            <span class="metric">Eval tiles <b id="evalTiles">—</b></span>
          </div>
        </div>
        <div>
          <div class="chart-label">Eval win rate</div>
          <svg class="metric-chart" id="winChart" viewBox="0 0 150 72" preserveAspectRatio="none"></svg>
        </div>
        <div>
          <div class="chart-label">PPO entropy (exploration)</div>
          <svg class="metric-chart" id="entropyChart" viewBox="0 0 360 72" preserveAspectRatio="none"></svg>
        </div>
      </section>
    </div>
    <aside class="side">
      <div class="row"><span class="k">policy</span><span class="v mono" id="policy">—</span></div>
      <div class="row"><span class="k">episode</span><span class="v" id="episode">1</span></div>
      <div class="row"><span class="k">live W/L</span><span class="v" id="sideWL">0 / 0</span></div>
      <div class="row"><span class="k">seed</span><span class="v" id="seed">1</span></div>
      <div class="row"><span class="k">tick / stride</span><span class="v" id="tick">0</span></div>
      <div class="row"><span class="k">status</span><span class="v"><span class="pill on" id="status">LIVE</span></span></div>
      <div class="row"><span class="k">step / infer</span><span class="v mono" id="timing">—</span></div>
      <div class="row"><span class="k">sim sps</span><span class="v mono" id="sps">—</span></div>
      <div class="row"><span class="k">action</span><span class="v" id="action">—</span></div>
      <div class="row"><span class="k">troop frac</span><span class="v" id="frac">—</span></div>
      <div class="row"><span class="k">troop ratio</span><span class="v" id="ratio">—</span></div>
      <div class="row"><span class="k">legal</span><span class="v mono" id="legal">—</span></div>
      <hr />
      <div class="block">
        <div class="row"><span class="k" style="color:var(--agent)">Agent</span><span class="v" id="agentTiles">0</span></div>
        <div class="row"><span class="k">troops</span><span class="v" id="agentTroops">0</span></div>
        <div class="row"><span class="k">gold</span><span class="v" id="agentGold">0</span></div>
        <div class="row"><span class="k">cities / boats</span><span class="v" id="agentEco">0 / 0</span></div>
        <div class="row"><span class="k">attacks</span><span class="v" id="agentAtk">0</span></div>
        <div class="row"><span class="k">placement</span><span class="v" id="placement">1</span></div>
      </div>
      <div class="block">
        <div class="row"><span class="k" style="color:var(--nation)">FFA nations</span><span class="v" id="nationAlive">0 alive</span></div>
        <div class="row"><span class="k">enemy tiles</span><span class="v" id="nationTiles">0</span></div>
        <div class="row"><span class="k">enemy troops</span><span class="v" id="nationTroops">0</span></div>
        <div class="row"><span class="k">strongest</span><span class="v" id="nationStrong">0</span></div>
        <div class="row"><span class="k">attacks</span><span class="v" id="nationAtk">0</span></div>
      </div>
      <div class="block">
        <div class="k">territory share (agent vs all enemies)</div>
        <div class="bar"><i id="barA" style="background:var(--agent);width:50%"></i><i id="barN" style="background:var(--nation);width:50%"></i></div>
      </div>
      <div class="row"><span class="k">step reward</span><span class="v" id="reward">0</span></div>
      <div class="row"><span class="k">return</span><span class="v" id="ret">0</span></div>
      <svg class="spark" id="spark" viewBox="0 0 280 48" preserveAspectRatio="none"></svg>
      <div class="controls">
        <button class="primary" id="pause">Pause</button>
        <button id="reset">Reset</button>
        <button data-spd="40">Faster</button>
        <button data-spd="80">1×</button>
        <button data-spd="160">Slower</button>
      </div>
      <div class="legend">
        <div><span class="swatch" style="background:var(--agent)"></span>Agent</div>
        <div><span class="swatch" style="background:linear-gradient(90deg,#e85d4c,#e8c84c,#4ce88a,#4c9ee8,#c44ce8)"></span>Nations (unique colors)</div>
        <div><span class="swatch" style="background:var(--land)"></span>Unowned land</div>
        <div><span class="swatch" style="background:#1b3a52"></span>Water</div>
      </div>
    </aside>
  </div>
  <script>
    // Fixed terrain + agent. Nations use golden-ratio hues so neighbors differ.
    const BASE = { 0: [27,58,82], 1: [201,184,150], 2: [74,163,255] };
    const nationPalette = [];
    function hslToRgb(h, s, l) {
      const a = s * Math.min(l, 1 - l);
      const f = (n) => {
        const k = (n + h * 12) % 12;
        return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      };
      return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
    }
    function nationColor(i) {
      if (nationPalette[i]) return nationPalette[i];
      // Golden-ratio hue walk + alternating lightness for adjacent contrast.
      const h = (i * 0.618033988749895) % 1;
      const l = i % 2 === 0 ? 0.52 : 0.40;
      const s = 0.62 + (i % 5) * 0.05;
      nationPalette[i] = hslToRgb(h, Math.min(0.85, s), l);
      return nationPalette[i];
    }
    function cellColor(code) {
      if (code <= 2) return BASE[code] || BASE[1];
      return nationColor(code - 3);
    }
    function rgbCss(rgb) {
      return "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
    }
    const canvas = document.getElementById("map");
    const ctx = canvas.getContext("2d");
    let img = null;
    let cellsBuf = null;
    let prevFrame = null;
    let nextFrame = null;
    let blendT0 = 0;
    let blendMs = 80;
    let lastSimTick = -1;
    function fmt(n) { return Math.round(n).toLocaleString(); }
    function applyDeltas(base, deltas, w, h) {
      if (!deltas || !deltas.length) return base;
      const arr = base.split("");
      for (const d of deltas) {
        if (d.i >= 0 && d.i < arr.length) arr[d.i] = String.fromCharCode(d.c);
      }
      return arr.join("");
    }
    function paintCells(f, alpha) {
      if (!f || !f.cells) return;
      if (!img || canvas.width !== f.w || canvas.height !== f.h) {
        canvas.width = f.w; canvas.height = f.h;
        img = ctx.createImageData(f.w, f.h);
        cellsBuf = f.cells;
      }
      if (f.deltas && cellsBuf && cellsBuf.length === f.cells.length) {
        cellsBuf = applyDeltas(cellsBuf, f.deltas, f.w, f.h);
      } else {
        cellsBuf = f.cells;
      }
      const d = img.data;
      const coded = f.nationCount != null;
      const a = alpha == null ? 1 : alpha;
      for (let i = 0; i < cellsBuf.length; i++) {
        const code = coded
          ? cellsBuf.charCodeAt(i)
          : (Number(cellsBuf[i]) || 1);
        const c = cellColor(code);
        const o = i * 4;
        // Soft blend toward new colors keeps 60fps feel during sim jitter.
        d[o] = Math.round(d[o] * (1 - a) + c[0] * a);
        d[o+1] = Math.round(d[o+1] * (1 - a) + c[1] * a);
        d[o+2] = Math.round(d[o+2] * (1 - a) + c[2] * a);
        d[o+3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      for (const u of f.units || []) {
        if (u.owner === "agent") {
          ctx.fillStyle = "#dbeafe";
        } else if (u.nationIndex != null) {
          ctx.fillStyle = rgbCss(nationColor(u.nationIndex));
        } else {
          ctx.fillStyle = "#fee2e2";
        }
        ctx.fillRect(u.x - 1, u.y - 1, 3, 3);
      }
    }
    function drawHud(f) {
      if (!f) return;
      const enemies = f.enemies || { alive: 0, tiles: 0, troops: 0, strongestTiles: 0, strongestTroops: 0 };
      const land = Math.max(1, f.agent.tiles + (enemies.tiles || 0));
      document.getElementById("policy").textContent = f.policyName;
      document.getElementById("episode").textContent = f.episode;
      document.getElementById("seed").textContent = f.seed;
      document.getElementById("tick").textContent = f.tick + " / " + f.stride;
      const infer = f.inference;
      const inferMs = infer?.lastInferenceMs == null
        ? "—"
        : Math.round(infer.lastInferenceMs) + "ms";
      const stepMsVal = f.stepDurationMs == null
        ? "—"
        : Math.round(f.stepDurationMs) + "ms";
      document.getElementById("timing").textContent = stepMsVal + " / " + inferMs;
      document.getElementById("sps").textContent =
        (f.achievedStepsPerSec != null ? f.achievedStepsPerSec.toFixed(2) : "—") +
        " · tick#" + (f.simTick ?? 0);
      document.getElementById("liveMatch").textContent =
        "${MAP_NAME} Solo FFA · " + f.difficulty + " · " + (enemies.alive ?? 0) +
        " nations alive · map " + (f.fullW || f.w) + "×" + (f.fullH || f.h) +
        (f.mapStride > 1 ? " ↓" + f.mapStride : "");
      document.getElementById("action").textContent = f.action + " → " + f.target;
      document.getElementById("frac").textContent = (100 * f.troopFrac).toFixed(0) + "%";
      document.getElementById("ratio").textContent =
        (100 * f.troopRatio).toFixed(0) + "% of cap" +
        " · growth " + (100 * (f.growthEfficiency ?? 0)).toFixed(0) + "%" +
        " (peak @" + (100 * (f.optimalGrowthRatio ?? 0)).toFixed(0) + "%)";
      document.getElementById("legal").textContent = (f.legalActions || []).join(", ");
      document.getElementById("agentTiles").textContent = fmt(f.agent.tiles) + " tiles";
      document.getElementById("agentTroops").textContent = fmt(f.agent.troops);
      document.getElementById("agentGold").textContent = fmt(f.agent.gold);
      document.getElementById("agentEco").textContent =
        (f.agent.cities ?? 0) + " / " + (f.agent.boats ?? 0);
      document.getElementById("agentAtk").textContent = f.attacks.agent;
      document.getElementById("placement").textContent = "#" + (f.placement ?? 1);
      document.getElementById("nationAlive").textContent = (enemies.alive ?? 0) + " alive";
      document.getElementById("nationTiles").textContent = fmt(enemies.tiles || 0) + " tiles";
      document.getElementById("nationTroops").textContent = fmt(enemies.troops || 0);
      document.getElementById("nationStrong").textContent =
        fmt(enemies.strongestTiles || 0) + " tiles / " + fmt(enemies.strongestTroops || 0) + " tr";
      document.getElementById("nationAtk").textContent = f.attacks.nation;
      document.getElementById("reward").textContent = (f.reward >= 0 ? "+" : "") + f.reward.toFixed(1);
      document.getElementById("ret").textContent = f.cumulativeReward.toFixed(1);
      document.getElementById("barA").style.width = (100 * f.agent.tiles / land).toFixed(1) + "%";
      document.getElementById("barN").style.width = (100 * (enemies.tiles || 0) / land).toFixed(1) + "%";
      const st = document.getElementById("status");
      if (f.stepError) {
        st.textContent = "STEP ERROR";
        st.className = "pill";
      } else if (infer?.consecutiveFailures) {
        st.textContent = "INFER FALLBACK";
        st.className = "pill";
      } else if (f.done) {
        st.textContent = f.winner === "agent" ? "AGENT WIN" : f.winner ? "NATION WIN" : "DONE";
        st.className = "pill";
      } else {
        st.textContent = "LIVE";
        st.className = "pill on";
      }
    }
    function onState(j) {
      if (!j || !j.frame) return;
      const f = j.frame;
      if ((f.simTick ?? 0) !== lastSimTick) {
        prevFrame = nextFrame;
        nextFrame = f;
        lastSimTick = f.simTick ?? 0;
        blendT0 = performance.now();
        blendMs = Math.max(20, j.stepMs || 80);
      }
      spark(j.history);
      training(j.metrics, j.episodes);
      drawHud(f);
    }
    function rafLoop() {
      const now = performance.now();
      const t = Math.min(1, (now - blendT0) / blendMs);
      const f = nextFrame || prevFrame;
      if (f) paintCells(f, prevFrame ? Math.max(0.35, t) : 1);
      requestAnimationFrame(rafLoop);
    }
    requestAnimationFrame(rafLoop);
    function spark(hist) {
      const svg = document.getElementById("spark");
      if (!hist || hist.length < 2) { svg.innerHTML = ""; return; }
      const maxT = Math.max(1, ...hist.map(h => Math.max(h.agent, h.enemies ?? h.nation ?? 0)));
      const n = hist.length - 1;
      const line = (key) => hist.map((h,i) => {
        const x = (i / n) * 280;
        const y = 44 - ((h[key] ?? h.enemies ?? 0) / maxT) * 40;
        return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ");
      svg.innerHTML =
        '<path d="' + line("agent") + '" fill="none" stroke="#4aa3ff" stroke-width="1.5"/>' +
        '<path d="' + line("enemies") + '" fill="none" stroke="#e85d4c" stroke-width="1.5"/>';
    }
    function lineChart(el, values, color, width, height) {
      if (!values || values.length < 2) { el.innerHTML = ""; return; }
      const lo = Math.min(...values), hi = Math.max(...values);
      const span = Math.max(0.000001, hi - lo);
      const d = values.map((v, i) => {
        const x = (i / (values.length - 1)) * width;
        const y = height - 4 - ((v - lo) / span) * (height - 8);
        return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ");
      el.innerHTML = '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.5"/>';
    }
    function training(metrics, episodes) {
      document.getElementById("liveWL").textContent = (episodes?.wins ?? 0) + "/" + (episodes?.losses ?? 0);
      document.getElementById("sideWL").textContent = (episodes?.wins ?? 0) + " / " + (episodes?.losses ?? 0);
      if (!metrics || !metrics.length) return;
      const last = metrics[metrics.length - 1];
      const evals = metrics.filter(m => m.eval);
      document.getElementById("ppoStep").textContent = last.ppo_step;
      document.getElementById("difficulty").textContent =
        (last.train_difficulty || last.difficulty || "—") +
        (last.difficulty && last.train_difficulty && last.difficulty !== last.train_difficulty
          ? " → " + last.difficulty
          : "");
      document.getElementById("entropy").textContent = Number(last.entropy).toFixed(2);
      const ev = evals.length ? evals[evals.length - 1].eval : null;
      document.getElementById("evalTiles").textContent = ev
        ? fmt(ev.mean_tiles) + (ev.mean_nation_tiles != null ? "/" + fmt(ev.mean_nation_tiles) : "")
        : "pending";
      document.getElementById("evalWinRate").textContent = ev && ev.win_rate != null
        ? ((100 * ev.win_rate).toFixed(0) + "% " + (ev.difficulty ? "[" + ev.difficulty + "] " : "")
          + "(" + (ev.wins ?? 0) + "/" + ((ev.wins ?? 0) + (ev.losses ?? 0)) + ")")
        : "pending";
      lineChart(document.getElementById("entropyChart"), metrics.map(m => m.entropy), "#a78bfa", 360, 72);
      const liveTier = evals.filter(m =>
        (m.eval?.difficulty || m.train_difficulty || m.difficulty) ===
        (last.difficulty || last.train_difficulty)
      );
      lineChart(
        document.getElementById("winChart"),
        (liveTier.length ? liveTier : evals).map(m =>
          (m.eval && m.eval.win_rate != null ? m.eval.win_rate : 0)
        ),
        "#86efac",
        150,
        72,
      );
    }
    async function tick() {
      const r = await fetch("/state");
      const j = await r.json();
      onState(j);
    }
    // Prefer SSE push; fall back to polling if EventSource fails.
    try {
      const es = new EventSource("/events");
      es.onmessage = (ev) => {
        try { onState(JSON.parse(ev.data)); } catch (e) { console.error(e); }
      };
      es.onerror = () => { /* keep open; browser reconnects */ };
    } catch (e) {
      setInterval(tick, 80);
    }
    tick();
    document.getElementById("pause").onclick = async () => {
      const r = await fetch("/toggle", { method: "POST" });
      const j = await r.json();
      document.getElementById("pause").textContent = j.paused ? "Play" : "Pause";
    };
    document.getElementById("reset").onclick = () => fetch("/reset", { method: "POST" });
    document.querySelectorAll("button[data-spd]").forEach(b => {
      b.onclick = () => fetch("/speed?ms=" + b.dataset.spd, { method: "POST" });
    });
  </script>
</body>
</html>
`;

function send(
  res: http.ServerResponse,
  code: number,
  body: string,
  type: string,
) {
  res.writeHead(code, {
    "Content-Type": type,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", `http://127.0.0.1:${PORT}`);
  if (url.pathname === "/" && req.method === "GET") {
    send(res, 200, PAGE, "text/html; charset=utf-8");
    return;
  }
  if (url.pathname === "/state" && req.method === "GET") {
    // Serve last packed snapshot only — never rebuild on the request path.
    send(res, 200, lastStateJson, "application/json");
    return;
  }
  if (url.pathname === "/events" && req.method === "GET") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    const id = nextSseId++;
    sseClients.set(id, { res, id });
    res.write(`id: ${simTick}\ndata: ${lastStateJson}\n\n`);
    req.on("close", () => {
      sseClients.delete(id);
    });
    return;
  }
  if (url.pathname === "/health" && req.method === "GET") {
    send(
      res,
      200,
      JSON.stringify({
        ok: true,
        simTick,
        achievedStepsPerSec,
        stepMs,
        paused,
        sseClients: sseClients.size,
        lastStepDurationMs,
        lastStepError,
      }),
      "application/json",
    );
    return;
  }
  if (url.pathname === "/toggle" && req.method === "POST") {
    paused = !paused;
    publishStateCache();
    send(res, 200, JSON.stringify({ paused }), "application/json");
    return;
  }
  if (url.pathname === "/reset" && req.method === "POST") {
    episode += 1;
    resetting = true;
    void resetEnv()
      .then(() => {
        send(
          res,
          200,
          JSON.stringify({ ok: true, seed, episode }),
          "application/json",
        );
      })
      .finally(() => {
        resetting = false;
      });
    return;
  }
  if (url.pathname === "/speed" && req.method === "POST") {
    stepMs = Math.max(
      20,
      parseInt(url.searchParams.get("ms") ?? "80", 10) || 80,
    );
    publishStateCache();
    send(res, 200, JSON.stringify({ stepMs }), "application/json");
    return;
  }
  send(res, 404, "not found", "text/plain");
});

async function main() {
  console.debug = () => {};
  if (POLICY_NAME === "python") {
    try {
      const r = await fetch(`${INFER_URL.replace(/\/$/, "")}/health`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch (e) {
      console.error(
        `Python infer server not reachable at ${INFER_URL}. Start it with:\\n` +
          `  python ai/python/scripts/infer_server.py --ckpt ai/fixtures/checkpoints/policy.pt\\n`,
        e,
      );
      process.exit(1);
    }
  }
  const shutdown = async () => {
    try {
      await envRpc.close();
    } catch {
      /* ignore */
    }
  };
  process.once("SIGINT", () => {
    void shutdown().finally(() => process.exit(0));
  });
  process.once("SIGTERM", () => {
    void shutdown().finally(() => process.exit(0));
  });

  await envRpc.start();
  await resetEnv(1);
  lastStepAt = Date.now();
  setInterval(() => {
    if (paused || resetting || stepping) return;
    const now = Date.now();
    if (now - lastStepAt < stepMs) return;
    void stepOnce()
      .catch((e) => console.error(e))
      .finally(() => {
        lastStepAt = Date.now();
      });
  }, 20);
  server.listen(PORT, "127.0.0.1", () => {
    console.log(`OpenFront AI live UI → http://127.0.0.1:${PORT}`);
    console.log(
      `policy=${policy.name} map=${MAP_NAME} stride=${STRIDE} nations=${NATIONS} bots=${BOTS} stepMs=${stepMs} mapEvery=${MAP_EVERY} inferTimeoutMs=${INFER_TIMEOUT_MS} env=rpc`,
    );
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
