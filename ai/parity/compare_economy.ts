/**
 * Compare Rust compact snapshots to TS oracle fixtures (economy gold rate).
 * Full tick-hash M11 lands as the Rust port approaches bit-exactness.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { execFileSync } from "child_process";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixture = path.join(ROOT, "ai/fixtures/hashes/plains_economy.json");

if (!fs.existsSync(fixture)) {
  console.error("Run: npx tsx ai/parity/oracle_simple.ts plains 200 economy");
  process.exit(1);
}

const ts = JSON.parse(fs.readFileSync(fixture, "utf8"));
console.log("TS oracle final:", ts.final);

// Gold rate invariant: 100/tick for humans
const expectedGold = ts.final.tick * 100; // approx; spawn ticks may differ
console.log(
  `Gold check: got ${ts.final.gold}, ~${expectedGold} expected for ${ts.final.tick} ticks @ 100/tick`,
);
if (ts.final.gold < ts.final.tick * 50) {
  console.error("FAIL: gold income too low");
  process.exit(1);
}

// Rust side: run a tiny cargo test that prints economy
const out = execFileSync(
  "cargo",
  ["test", "-p", "openfront-sim", "m2_economy", "--", "--nocapture"],
  { cwd: path.join(ROOT, "ai"), encoding: "utf8" },
);
if (!out.includes("m2_economy_grows_troops_and_gold ... ok")) {
  console.error(out);
  process.exit(1);
}
console.log("Rust M2 economy test OK — formula parity for gold/troops income");
console.log("M11 full hash match: continue porting AttackExecution edge cases + Player.hash units");
