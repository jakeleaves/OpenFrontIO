/**
 * Run the heuristic agent in the real local OpenFront Solo UI.
 *
 * Starts a visible Chrome window connected to localhost:9000. The game uses
 * the random-map selector, Impossible difficulty, and the requested bot count.
 *
 * Usage:
 *   npm run dev
 *   node ai/ts/play_solo_browser.mjs
 *   node ai/ts/play_solo_browser.mjs --bots 50 --interval 1500
 */
import { chromium } from "playwright";
import {
  gotoHome,
  openSoloModal,
} from "../../.claude/skills/run-openfront/driver.mjs";
import {
  attack,
  findSpawnTile,
  gameState,
  setAttackRatio,
  spawn,
  startSoloGame,
  waitForSpawnPhaseEnd,
} from "../../.claude/skills/run-openfront/game.mjs";

function arg(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : Number(process.argv[index + 1]) || fallback;
}

const bots = arg("--bots", 400);
const interval = arg("--interval", 1_500);
async function chooseAttackTarget(page) {
  return page.evaluate(async () => {
    const game = document.querySelector("build-menu")?.game;
    const me = game?.myPlayer();
    if (!game || !me || !me.isAlive()) return null;

    const { borderTiles } = await me.borderTiles();
    let enemy = null;
    for (const border of borderTiles) {
      for (const tile of game.neighbors(border)) {
        if (!game.isLand(tile)) continue;
        const x = game.x(tile);
        const y = game.y(tile);
        if (!game.hasOwner(tile)) return { x, y, kind: "expand" };
        if (enemy === null && game.owner(tile) !== me) {
          enemy = { x, y, kind: "attack" };
        }
      }
    }
    return enemy;
  });
}

async function main() {
  const browser = await chromium.launch({
    channel: "chrome",
    headless: false,
    args: ["--start-maximized"],
  });
  const context = await browser.newContext({
    viewport: { width: 1500, height: 1000 },
  });
  const page = await context.newPage();
  page.on("pageerror", (error) => console.warn("PAGEERROR:", error.message));

  await gotoHome(page);
  await openSoloModal(page);

  await page.evaluate((botCount) => {
    const modal = document.querySelector("single-player-modal");
    if (!modal) throw new Error("single-player modal is not available");
    modal.bots = botCount;
    modal.selectedDifficulty = "Impossible";
    modal.querySelector("game-config-settings")?.dispatchEvent(
      new CustomEvent("random-map-selected", {
        bubbles: true,
        composed: true,
      }),
    );
  }, bots);
  await page.waitForTimeout(300);
  await startSoloGame(page);

  console.log(`Solo game ready: random map, Impossible, ${bots} bots`);
  const spawnTile = await findSpawnTile(page);
  await spawn(page, spawnTile);
  await waitForSpawnPhaseEnd(page);
  await setAttackRatio(page, 0.5);
  console.log("Agent spawned:", await gameState(page));

  let running = true;
  process.on("SIGINT", () => {
    running = false;
  });
  while (running) {
    const state = await gameState(page);
    if (!state?.myPlayer?.isAlive) {
      console.log("Agent eliminated:", state);
      break;
    }

    const target = await chooseAttackTarget(page);
    if (target) {
      await attack(page, target.x, target.y);
      console.log(
        `tick ${state.ticks}: ${target.kind} at (${target.x}, ${target.y})`,
      );
    }
    await page.waitForTimeout(interval);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
