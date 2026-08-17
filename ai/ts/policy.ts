/**
 * Pluggable policies for GameEnv playback (heuristic / Python checkpoint RPC).
 */
import type { Game, Player } from "../../src/core/game/Game";
import { expertAction } from "./expert";
import {
  ActionMask,
  ActionType,
  FactorizedAction,
  TROOP_FRACS,
  noopAction,
} from "./types";

export type ObsJson = {
  global: number[];
  local: number[];
  vector: number[];
  shapes?: unknown;
};

export type PolicyContext = {
  tick: number;
  gold: number;
  troopRatio: number;
  obs: ObsJson;
  mask: ActionMask;
  game?: Game;
  agent?: Player;
};

export type PolicyProvider = {
  name: string;
  reset?: () => Promise<void>;
  act: (ctx: PolicyContext) => Promise<FactorizedAction>;
  health?: () => PolicyHealth;
};

export type PolicyHealth = {
  lastInferenceMs: number | null;
  consecutiveFailures: number;
  lastError: string | null;
  retryAt: number | null;
};

export function heuristicPolicy(): PolicyProvider {
  return {
    name: "heuristic",
    async act(ctx) {
      if (ctx.game && ctx.agent) {
        return expertAction(ctx.game, ctx.agent);
      }
      // Fallback without live game handles
      if (ctx.gold > 250_000 && ctx.tick % 80 < 10) {
        return {
          actionType: ActionType.BUILD,
          targetPlayer: 0,
          cellX: 8,
          cellY: 8,
          troopFrac: 0,
          buildType: 0,
        };
      }
      const attackNation =
        ctx.tick > 250 && Math.floor(ctx.tick / 50) % 3 === 2;
      return {
        actionType: ActionType.ATTACK,
        targetPlayer: attackNation ? 1 : 0,
        cellX: 16,
        cellY: 8,
        troopFrac: attackNation ? 49 : 34, // ~50% / ~35%
        buildType: 0,
      };
    },
  };
}

export function pythonCkptPolicy(
  baseUrl: string,
  timeoutMs: number = 5_000,
): PolicyProvider {
  const url = baseUrl.replace(/\/$/, "");
  const health: PolicyHealth = {
    lastInferenceMs: null,
    consecutiveFailures: 0,
    lastError: null,
    retryAt: null,
  };

  async function fetchWithTimeout(
    endpoint: string,
    init: RequestInit,
  ): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(endpoint, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }
  }

  function recordFailure(error: unknown) {
    health.consecutiveFailures += 1;
    health.retryAt = Date.now() + timeoutMs;
    health.lastError =
      error instanceof Error && error.name === "AbortError"
        ? `inference timed out after ${timeoutMs}ms`
        : error instanceof Error
          ? error.message
          : String(error);
  }

  return {
    name: `python:${url}`,
    async reset() {
      await fetchWithTimeout(`${url}/reset`, {
        method: "POST",
        body: "{}",
      }).catch(() => {});
    },
    async act(ctx) {
      if (health.retryAt !== null && Date.now() < health.retryAt) {
        return noopAction();
      }

      const startedAt = performance.now();
      try {
        const res = await fetchWithTimeout(`${url}/act`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            obs: ctx.obs,
            mask: ctx.mask,
            deterministic: false,
          }),
        });
        if (!res.ok) {
          throw new Error(`infer HTTP ${res.status}`);
        }
        const data = (await res.json()) as {
          ok: boolean;
          action?: FactorizedAction;
          error?: string;
        };
        if (!data.ok || !data.action) {
          throw new Error(data.error ?? "infer failed");
        }
        health.lastInferenceMs = performance.now() - startedAt;
        health.consecutiveFailures = 0;
        health.lastError = null;
        health.retryAt = null;
        return data.action;
      } catch (error) {
        health.lastInferenceMs = performance.now() - startedAt;
        recordFailure(error);
        return noopAction();
      }
    },
    health: () => ({ ...health }),
  };
}

export const ACTION_NAMES = [
  "NOOP",
  "SPAWN",
  "ATTACK",
  "BOAT",
  "CANCEL_ATTACK",
  "CANCEL_BOAT",
  "BUILD",
  "UPGRADE",
  "MOVE_WARSHIP",
  "DELETE",
];

export function describeAction(a: FactorizedAction): {
  action: string;
  target: string;
  troopFrac: number;
} {
  const action = ACTION_NAMES[a.actionType] ?? String(a.actionType);
  const target =
    a.actionType === ActionType.BUILD
      ? `build#${a.buildType}`
      : a.targetPlayer === 0
        ? "unowned land"
        : `nation#${a.targetPlayer - 1}`;
  return {
    action,
    target,
    troopFrac: TROOP_FRACS[a.troopFrac] ?? 0,
  };
}

export { noopAction };
