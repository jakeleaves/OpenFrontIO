# OpenFront Dominant Agent (Nations 1v1)

AGPL-3.0 derivative of [OpenFrontIO](https://github.com/openfrontio/OpenFrontIO). All new code lives under `ai/`.

## Strategy

**Train against TypeScript `src/core`.** Rust under `sim/` / `ffi/` / `client/` is **frozen**.

## Layout

```
ai/
  ts/           # GameEnv, workers, RPC, local client — primary path
  python/       # policy + BC/APPO + ONNX export
  parity/       # Node oracle dumps
  fixtures/     # hashes + checkpoints
  sim|ffi|client/  # FROZEN
```

## Quick start

```bash
npm run inst

# Single env vs Impossible Nation
npx tsx ai/ts/smoke.ts

# Worker pool
npx tsx ai/ts/worker_pool_smoke.ts

# Offline demo client
npx tsx ai/ts/client.ts --demo --ticks 300

# Train (spawns TS RPC automatically)
cd ai && python python/scripts/train.py --bc-steps 2 --appo-steps 2 --batch-size 16

# Export ONNX (requires: pip install onnx)
python python/scripts/export_onnx.py \
  --ckpt fixtures/checkpoints/policy.pt \
  --out fixtures/checkpoints/policy.onnx

# Offline demo client (optional --onnx path noted for future ORT wiring)
npx tsx ai/ts/client.ts --demo --ticks 300 --onnx ai/fixtures/checkpoints/policy.onnx
```

## Env API

`GameEnv` (`ai/ts/env.ts`): `reset(seed)` → obs/mask/info; `step(factorizedAction)` advances `macroStride` ticks (default 20) with in-engine Impossible Nation.

Obs: global `(12,64,128)`, local `(8,64,64)`, vector `64`.  
Actions: factorized → real intents (`attack`, `boat`, `build_unit`, …) with legal masks.

## License

AGPL-3.0-only.
