/**
 * Binary obs + packed action-mask frames for the Python side-channel (fd 3).
 *
 * Layout (little-endian), after a separate u32 length prefix written by the RPC host:
 *   Float32 global[GLOBAL_C*H*W]
 *   Float32 local[LOCAL_C*H*W]
 *   Float32 vector[VECTOR_DIM]
 *   packed bits: actionType, targetPlayer, cell, troopFrac, buildType
 */
import {
  ActionMask,
  GLOBAL_C,
  GLOBAL_H,
  GLOBAL_W,
  LOCAL_C,
  LOCAL_H,
  LOCAL_W,
  NUM_ACTION_TYPES,
  NUM_BUILD_TYPES,
  NUM_TARGET_PLAYERS,
  NUM_TROOP_FRACS,
  Observation,
  COARSE_H,
  COARSE_W,
  VECTOR_DIM,
} from "./types";

export const NUM_CELL = COARSE_H * COARSE_W;

export const FLOAT_BYTES =
  (GLOBAL_C * GLOBAL_H * GLOBAL_W +
    LOCAL_C * LOCAL_H * LOCAL_W +
    VECTOR_DIM) *
  4;

function maskBytes(n: number): number {
  return Math.ceil(n / 8);
}

export const MASK_BYTES =
  maskBytes(NUM_ACTION_TYPES) +
  maskBytes(NUM_TARGET_PLAYERS) +
  maskBytes(NUM_CELL) +
  maskBytes(NUM_TROOP_FRACS) +
  maskBytes(NUM_BUILD_TYPES);

export const FRAME_BYTES = FLOAT_BYTES + MASK_BYTES;

function packBits(dst: Uint8Array, offset: number, bits: boolean[]): number {
  const nbytes = maskBytes(bits.length);
  dst.fill(0, offset, offset + nbytes);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) {
      dst[offset + (i >> 3)] |= 1 << (i & 7);
    }
  }
  return offset + nbytes;
}

/** Pack one observation + mask into a transferable ArrayBuffer. */
export function packObsMaskFrame(obs: Observation, mask: ActionMask): ArrayBuffer {
  const buf = new ArrayBuffer(FRAME_BYTES);
  const f32 = new Float32Array(buf, 0, FLOAT_BYTES / 4);
  let fi = 0;
  f32.set(obs.global, fi);
  fi += obs.global.length;
  f32.set(obs.local, fi);
  fi += obs.local.length;
  f32.set(obs.vector, fi);

  const u8 = new Uint8Array(buf);
  let o = FLOAT_BYTES;
  o = packBits(u8, o, mask.actionType);
  o = packBits(u8, o, mask.targetPlayer);
  o = packBits(u8, o, mask.cell);
  o = packBits(u8, o, mask.troopFrac);
  o = packBits(u8, o, mask.buildType);
  if (o !== FRAME_BYTES) {
    throw new Error(`wire frame size mismatch: wrote ${o} expected ${FRAME_BYTES}`);
  }
  return buf;
}

/** JSON-only payload when binary side-channel is active. */
export function slimStepResult(result: {
  obs: Observation;
  mask: ActionMask;
  info: unknown;
  timing?: unknown;
}): { info: unknown; timing?: unknown; encoding: "fd3" } {
  return {
    info: result.info,
    timing: result.timing,
    encoding: "fd3",
  };
}

function unpackBits(src: Uint8Array, offset: number, n: number): boolean[] {
  const out = new Array<boolean>(n);
  for (let i = 0; i < n; i++) {
    out[i] = (src[offset + (i >> 3)] & (1 << (i & 7))) !== 0;
  }
  return out;
}

/**
 * Convert a packed frame into the legacy f32b64 JSON obs + bool mask arrays
 * (used when OPENFRONT_OBS_FD is not set).
 */
export function frameToJsonObsMask(frame: ArrayBuffer): {
  obs: {
    encoding: "f32b64";
    global: string;
    local: string;
    vector: string;
  };
  mask: ActionMask;
} {
  const u8 = new Uint8Array(frame);
  const gBytes = GLOBAL_C * GLOBAL_H * GLOBAL_W * 4;
  const lBytes = LOCAL_C * LOCAL_H * LOCAL_W * 4;
  const vBytes = VECTOR_DIM * 4;
  const global = Buffer.from(u8.subarray(0, gBytes)).toString("base64");
  const local = Buffer.from(u8.subarray(gBytes, gBytes + lBytes)).toString(
    "base64",
  );
  const vector = Buffer.from(
    u8.subarray(gBytes + lBytes, gBytes + lBytes + vBytes),
  ).toString("base64");
  let o = FLOAT_BYTES;
  const actionType = unpackBits(u8, o, NUM_ACTION_TYPES);
  o += Math.ceil(NUM_ACTION_TYPES / 8);
  const targetPlayer = unpackBits(u8, o, NUM_TARGET_PLAYERS);
  o += Math.ceil(NUM_TARGET_PLAYERS / 8);
  const cell = unpackBits(u8, o, NUM_CELL);
  o += Math.ceil(NUM_CELL / 8);
  const troopFrac = unpackBits(u8, o, NUM_TROOP_FRACS);
  o += Math.ceil(NUM_TROOP_FRACS / 8);
  const buildType = unpackBits(u8, o, NUM_BUILD_TYPES);
  return {
    obs: { encoding: "f32b64", global, local, vector },
    mask: { actionType, targetPlayer, cell, troopFrac, buildType },
  };
}
