// Unit test for the Anbindung math. Extracts the self-contained
// `<anbindung-math>` block from index.html (single source of truth) and checks the
// per-mode ramp (anbindungSub) and the best-wins + diversity aggregation
// (anbindungAggregate).  Run: `deno test --allow-read web/`
//
// Model: each enabled mode scores sc = clamp01((T − t·PEN)/(T − floor))^1.5
// (convex falloff, "wenn nötig" inflates the time by PEN=1.25). The modes
// combine by a tempered noisy-OR — best mode at full weight, every other mode
// multiplies in (1 − β·sc) — so the result is ∈ [max, 1] and MONOTONE in the
// mode set: enabling or upgrading a mode can never lower the Anbindung score.

const html = Deno.readTextFileSync(new URL("./index.html", import.meta.url));
const m = html.match(/\/\/ <anbindung-math>[^\n]*\n([\s\S]*?)\/\/ <\/anbindung-math>/);
if (!m) throw new Error("anbindung-math block not found in index.html");
const { anbindungSub, anbindungAggregate } = new Function(
  m[1] + "\nreturn { anbindungSub, anbindungAggregate };")();

const near = (a, b, eps = 1e-9, msg = "") => {
  if (Math.abs(a - b) > eps) throw new Error(`${msg}: ${a} ≠ ${b}`);
};
const assert = (cond, msg) => { if (!cond) throw new Error(msg); };

// --- anbindungSub: the per-mode ramp ------------------------------------------
Deno.test("anbindungSub: 1 at the floor, 0 at the budget, clamps below the floor", () => {
  near(anbindungSub(15, 15, 80, 1), 1, 1e-9, "t = floor -> 1");
  near(anbindungSub(80, 15, 80, 1), 0, 1e-9, "t = budget -> 0");
  near(anbindungSub(10, 15, 80, 1), 1, 1e-9, "below the floor clamps to 1");
});

Deno.test("anbindungSub: unreachable (null or ≥ sentinel) scores 0", () => {
  near(anbindungSub(null, 15, 80, 1), 0, 1e-9, "null");
  near(anbindungSub(120, 15, 80, 1), 0, 1e-9, "at the sentinel");
  near(anbindungSub(150, 15, 80, 1), 0, 1e-9, "beyond the sentinel");
});

Deno.test("anbindungSub: convex falloff — midpoint sits below the linear ramp", () => {
  // 47.5 min is halfway in [15, 80] -> linear 0.5, convex p=1.5 -> 0.5^1.5
  const s = anbindungSub(47.5, 15, 80, 1);
  near(s, Math.pow(0.5, 1.5), 1e-9, "convex value");
  assert(s < 0.5, "convexity pulls the mid-range below linear");
});

Deno.test("anbindungSub: 'wenn nötig' penalty never raises a mode's score", () => {
  for (const t of [20, 35, 50, 65]) {
    const gern = anbindungSub(t, 15, 80, 1.0);     // am liebsten
    const noet = anbindungSub(t, 15, 80, 1.25);    // wenn nötig
    assert(noet <= gern + 1e-12, `penalty must not raise the score (t=${t})`);
  }
});

// --- anbindungAggregate: best-wins + diversity, monotone ----------------------
Deno.test("anbindungAggregate: no enabled mode -> 0 (layer goes dark)", () => {
  near(anbindungAggregate([]), 0, 1e-9, "empty");
});

Deno.test("anbindungAggregate: a single mode returns its own score exactly", () => {
  for (const s of [0, 0.3, 0.7, 1]) near(anbindungAggregate([s]), s, 1e-9, `single ${s}`);
});

Deno.test("anbindungAggregate: result is never below the best mode (best-wins)", () => {
  for (const subs of [[0.9, 0.1], [0.2, 0.8, 0.5], [0.6, 0.6], [1, 0.3], [0, 0.4]]) {
    assert(anbindungAggregate(subs) >= Math.max(...subs) - 1e-12, `>= best for ${subs}`);
    assert(anbindungAggregate(subs) <= 1 + 1e-12, `stays bounded at 1 for ${subs}`);
  }
});

Deno.test("anbindungAggregate: MONOTONE — enabling a mode never lowers the score", () => {
  // the core guarantee: "Anbindung measures connectedness, not one commute"
  for (const base of [[0.7], [0.3, 0.6], [0.9]]) {
    for (const extra of [0, 0.1, 0.5, 0.9, 1]) {
      assert(anbindungAggregate([...base, extra]) >= anbindungAggregate(base) - 1e-12,
        `adding ${extra} to ${base} must not lower`);
    }
  }
});

Deno.test("anbindungAggregate: MONOTONE — upgrading a mode never lowers the score", () => {
  assert(anbindungAggregate([0.7, 0.5]) <= anbindungAggregate([0.7, 0.6]) + 1e-12,
    "raising a sub-score raises or holds");
  assert(anbindungAggregate([0.4, 0.4]) <= anbindungAggregate([0.4, 0.9]) + 1e-12,
    "upgrading the second mode to best raises or holds");
});

Deno.test("anbindungAggregate: two independent ways beat one (diversity bonus)", () => {
  assert(anbindungAggregate([0.5, 0.5]) > 0.5, "two mediocre modes beat one");
  assert(anbindungAggregate([0.9, 0.5]) > 0.9, "a second decent mode adds a bonus");
  // a weak fallback helps, but only marginally (best-wins dominates)
  const solo = anbindungAggregate([0.9]), withWeak = anbindungAggregate([0.9, 0.1]);
  assert(withWeak > solo, "weak fallback still helps");
  assert(withWeak - solo < 0.05, "but only a little");
});

Deno.test("anbindungAggregate: ties count both modes (max pulled exactly once)", () => {
  // 1 - (1-0.6)*(1-0.4*0.6) = 1 - 0.4*0.76 = 0.696
  near(anbindungAggregate([0.6, 0.6]), 1 - 0.4 * (1 - 0.4 * 0.6), 1e-9, "tie");
});
