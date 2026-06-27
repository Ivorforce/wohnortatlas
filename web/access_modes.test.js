// Unit test for the transport-mode access math. Extracts the self-contained
// `<access-mode-math>` block from index.html (single source of truth) and checks
// mode selection + the per-item access shift.  Run: `deno test --allow-read web/`
//
// Model: walking is the baseline; a faster willing mode is chosen only if it saves
// >= MODE_SWITCH (3) min over walking (you don't get the bike out for a trivial
// gain). proxScore is the distance utility; proxG adds the falloff exponent
// (γ=2 for "Direkt vor Ort", γ=1 for "Gut erreichbar").

const html = Deno.readTextFileSync(new URL("./index.html", import.meta.url));
const m = html.match(/\/\/ <access-mode-math>[^\n]*\n([\s\S]*?)\/\/ <\/access-mode-math>/);
if (!m) throw new Error("access-mode-math block not found in index.html");
const { proxScore, proxG, bestMode } = new Function(
  m[1] + "\nreturn { proxScore, proxG, bestMode };")();

const near = (a, b, eps = 1e-6, msg = "") => {
  if (Math.abs(a - b) > eps) throw new Error(`${msg}: ${a} ≠ ${b}`);
};
const isMode = (got, want, msg) => {
  if (got !== want) throw new Error(`${msg}: got ${got}, want ${want}`);
};

Deno.test("supermarket next door: bike saves ~nothing -> WALK (the bug)", () => {
  // dense-cell supermarket ~3.6-min walk; bike would be 3.58 — saves 0.02 min
  const r = bestMode(3.6, "foot", { foot: "gern", bike: "gern", car: null });
  isMode(r.mode, "foot", "near supermarket must be on foot");
  near(r.t, 3.6, 1e-9, "shows the walk time");
});

Deno.test("far supermarket (12-min walk): bike saves ~6 min -> BIKE", () => {
  const r = bestMode(12, "foot", { bike: "gern" });
  isMode(r.mode, "bike", "far supermarket switches to bike");
  near(r.t, 6.1, 1e-6, "bike time = 12·4.5/15 + 2.5");
});

Deno.test("doorstep (3-min walk): no faster mode worth it even with car", () => {
  const r = bestMode(3, "foot", { bike: "gern", car: "gern" });
  isMode(r.mode, "foot", "doorstep stays on foot");
  near(r.t, 3, 1e-9);
});

Deno.test("bike-ref amenity (hospital 20-min bike): bike clears the bar easily", () => {
  isMode(bestMode(20, "bike", { bike: "gern" }).mode, "bike", "far bike-ref -> bike");
  near(bestMode(20, "bike", { bike: "gern" }).t, 22.5, 1e-6, "20 + 2.5 overhead");
  // car-only profile reaches it by car (foot is 66.7 min)
  const c = bestMode(20, "bike", { car: "gern" });
  isMode(c.mode, "car", "car-only -> car");
  near(c.t, 16, 1e-6, "20·15/30 + 6");
});

Deno.test("want/can: 'wenn nötig' raises the bar, can flip a trip back to foot", () => {
  isMode(bestMode(9, "foot", { bike: "gern" }).mode, "bike", "gern: 9-min walk -> bike");
  isMode(bestMode(9, "foot", { bike: "koennen" }).mode, "foot", "koennen: same trip -> walk");
});

Deno.test("no mode selected falls back to walking", () => {
  isMode(bestMode(5, "foot", {}).mode, "foot", "all-off -> foot baseline");
  near(bestMode(5, "foot", {}).t, 5, 1e-9);
});

Deno.test("proxScore: linear 1 at the free radius -> 0 at tzero (foot)", () => {
  near(proxScore(3, 15, "foot"), 1, 1e-9, "3-min walk = free radius");
  near(proxScore(15, 15, "foot"), 0, 1e-9, "at tzero");
  near(proxScore(12, 15, "foot"), 0.25, 1e-9, "12-min walk = quarter credit (linear)");
});

Deno.test("proxG: Direkt-vor-Ort (γ=2) punishes mid distances; γ=1 is linear", () => {
  near(proxG(12, 15, "foot", 1), 0.25, 1e-9, "γ=1 == proxScore");
  near(proxG(12, 15, "foot", 2), 0.0625, 1e-9, "γ=2: 12-min walk ≈ 0.06, not 0.25");
  near(proxG(3, 15, "foot", 2), 1, 1e-9, "doorstep still full under γ=2");
});
