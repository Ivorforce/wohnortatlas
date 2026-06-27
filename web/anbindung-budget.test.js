// Unit test for the FLOOR-RELATIVE Anbindung budget + bliss anchor (2026-06). Extracts
// the self-contained `<anbindung-budget>` block from index.html (single source of truth)
// and checks the pure cores that computeTargetFloor / autoSetBudget / captureSlack wire to
// the DOM.  Run: `deno test --allow-read web/`
//
// Model: a target's reach FLOOR (best achievable minutes) anchors both the score's bliss
// point and the default budget, so a hard target (a whole city, optimal cell ~22 min) is
// not crushed and a single point (floor ≈ 0) stays at today's ~60/30.

const html = Deno.readTextFileSync(new URL("./index.html", import.meta.url));
const m = html.match(/\/\/ <anbindung-budget>[^\n]*\n([\s\S]*?)\/\/ <\/anbindung-budget>/);
if (!m) throw new Error("anbindung-budget block not found in index.html");
const { round5, reachFloorOf, blissFloorFor, budgetFor, slackFor } = new Function(
  m[1] + "\nreturn { round5, reachFloorOf, blissFloorFor, budgetFor, slackFor };")();

const near = (a, b, eps = 1e-9, msg = "") => {
  if (Math.abs(a - b) > eps) throw new Error(`${msg}: ${a} ≠ ${b}`);
};
const assert = (cond, msg) => { if (!cond) throw new Error(msg); };
const SENT = 120;

// --- reachFloorOf: best achievable minutes over a per-cell array ---------------------
Deno.test("reachFloorOf: min over reachable cells, ignoring null + the sentinel", () => {
  near(reachFloorOf([40, 22, null, 35], 4, SENT), 22, 1e-9, "min of the finite values");
  near(reachFloorOf([null, SENT, 150], 3, SENT), Infinity, 1e-9, "all unreachable -> ∞");
  near(reachFloorOf(null, 5, SENT), Infinity, 1e-9, "missing array -> ∞");
  near(reachFloorOf([SENT - 0.1, 200], 2, SENT), SENT - 0.1, 1e-9, "just below the sentinel counts; ≥ sentinel skipped");
});

// --- blissFloorFor: where the per-mode score saturates to 1 --------------------------
Deno.test("blissFloorFor: a point target keeps the comfort floor (unchanged behaviour)", () => {
  // floor ≈ 0–3 min sits below the comfort floor → max() leaves it untouched
  near(blissFloorFor(20, 3, 80), 20, 1e-9, "ÖPNV comfort 20 wins over a tiny floor");
  near(blissFloorFor(15, 0, 80), 15, 1e-9, "car comfort 15 wins over floor 0");
});

Deno.test("blissFloorFor: a whole-city target lifts the anchor to its floor (optimal cell -> 1)", () => {
  // ganz München ÖPNV floor ≈ 22 > comfort 20 → the optimal cell (t=22) scores 1, not mid
  near(blissFloorFor(20, 22, 80), 22, 1e-9, "floor above comfort becomes the anchor");
});

Deno.test("blissFloorFor: clamped below T so (T − floor) stays positive", () => {
  near(blissFloorFor(20, 100, 50), 49, 1e-9, "floor above the budget clamps to T-1");
});

Deno.test("blissFloorFor: an unreachable mode (∞ floor) falls back to the comfort floor", () => {
  near(blissFloorFor(15, Infinity, 80), 15, 1e-9, "∞ -> comfort (harmless; no cell reaches anyway)");
});

// --- budgetFor: default budget = floor + slack, snapped + clamped --------------------
Deno.test("budgetFor: a single point (floor ≈ 0) stays at today's ~60 / 30", () => {
  near(budgetFor(0, 60, 25, 180), 60, 1e-9, "default slack");
  near(budgetFor(0, 30, 25, 180), 30, 1e-9, "Schnelle Anbindung slack");
});

Deno.test("budgetFor: scales with the target's floor (München floor 22 -> 80 / 50)", () => {
  near(budgetFor(22, 60, 25, 180), 80, 1e-9, "default: round5(82)");
  near(budgetFor(22, 30, 25, 180), 50, 1e-9, "Schnelle: round5(52)");
});

Deno.test("budgetFor: snaps to 5 and clamps to the input's [min, max]", () => {
  near(budgetFor(11, 60, 25, 180), 70, 1e-9, "round5(71) -> 70");
  near(budgetFor(0, 10, 25, 180), 25, 1e-9, "below the input min clamps up");
  near(budgetFor(200, 60, 25, 180), 180, 1e-9, "above the input max clamps down");
});

// --- slackFor: the headroom a (manual / restored) budget implies ---------------------
Deno.test("slackFor: recovers headroom as budget − floor, clamped", () => {
  near(slackFor(80, 22, 10, 150), 58, 1e-9, "80 − 22");
  near(slackFor(20, 22, 10, 150), 10, 1e-9, "budget below floor clamps to the slack min");
  near(slackFor(500, 0, 10, 150), 150, 1e-9, "absurd budget clamps to the slack max");
});

// --- round-trip: switching targets (same floor) is stable ----------------------------
Deno.test("budgetFor∘slackFor: re-deriving the budget from the recovered slack is stable", () => {
  // a manual budget → captureSlack → autoSetBudget must reproduce the SAME budget, so
  // toggling away and back to a target never drifts.
  for (const [floor, budget] of [[22, 80], [0, 60], [12, 70], [22, 50]]) {
    const s = slackFor(budget, floor, 10, 150);
    near(budgetFor(floor, s, 25, 180), budget, 1e-9, `stable for floor ${floor}, budget ${budget}`);
  }
});

Deno.test("round5: snaps to the nearest 5", () => {
  near(round5(82), 80); near(round5(83), 85); near(round5(52), 50); near(round5(0), 0);
});
