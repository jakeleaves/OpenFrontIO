/**
 * Troop growth helpers mirroring Config.troopIncreaseRate.
 * Growth = (10 + troops^0.73 / 4) * (1 - troops/maxTroops).
 */

export function troopIncreaseAt(troops: number, maxTroops: number): number {
  const max = Math.max(1, maxTroops);
  const t = Math.max(0, Math.min(troops, max));
  const toAdd = (10 + Math.pow(t, 0.73) / 4) * (1 - t / max);
  return Math.max(0, toAdd);
}

/** Ratio in [0,1] that maximizes absolute troop regen for a given cap. */
export function optimalGrowthRatio(maxTroops: number): number {
  const max = Math.max(1, maxTroops);
  let bestR = 0.5;
  let best = -1;
  // Dense enough for shaping; peak is smooth in r.
  for (let i = 1; i < 100; i++) {
    const r = i / 100;
    const rate = troopIncreaseAt(r * max, max);
    if (rate > best) {
      best = rate;
      bestR = r;
    }
  }
  return bestR;
}

export function growthEfficiency(
  troops: number,
  maxTroops: number,
): { rate: number; peak: number; optimalRatio: number; efficiency: number } {
  const max = Math.max(1, maxTroops);
  const optimalRatio = optimalGrowthRatio(max);
  const rate = troopIncreaseAt(troops, max);
  const peak = troopIncreaseAt(optimalRatio * max, max);
  return {
    rate,
    peak,
    optimalRatio,
    efficiency: peak > 1e-6 ? rate / peak : 0,
  };
}
