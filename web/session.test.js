// Round-trip test for the NAMED session codec (the browser-store format, distinct
// from the positional #link share blob). Extracts the self-contained
// `<session-codec>` block from index.html — sessionFromShare / shareFromSession —
// and evals it with MOCK config constants injected (the translators depend on the
// runtime, config-derived key/enum arrays W_KEYS/AC_KEYS/FRZ_KEYS/… which don't
// exist outside the browser).
//
// The guarantee this guards: shareFromSession(sessionFromShare(arr)) === arr, i.e.
// the named mirror can't silently drift from captureShareState's positional slot
// order — and that a stored session survives an enum/layer removal (a SHARE_V bump)
// by key, instead of being wiped.
// Run: `deno test --allow-read web/`

const html = Deno.readTextFileSync(new URL("./index.html", import.meta.url));
const m = html.match(/\/\/ <session-codec>[^\n]*\n([\s\S]*?)\/\/ <\/session-codec>/);
if (!m) throw new Error("session-codec block not found in index.html");

// A self-consistent mock build config (orders are what makes the positional form
// brittle; the named form must be immune to them).
const PRELUDE = `
  const SESSION_V = 1, SHARE_V = 8;
  const W_KEYS = ["miete", "anbindung", "natur", "ruhe"];
  const RENT_M = ["eur", "pay"];
  const STAGE_KEYS = ["gemischt", "familie", "senior"];
  const MODE_VALS = [null, "gern", "koennen", "nicht"];
  const AC_KEYS = ["nahversorgung", "kinder", "wasser"];
  const ACC_VALS = ["egal", "nice", "muss", "on"];
  const VIEW_VALS = ["__composite__", null, "__pop__", "__none__"];
  const TAB_VALS = ["factors", "detail", "vergleichen", "suche"];
  const FRZ_KEYS = ["kultur", "kino", "schwimmen", "klettern", "golf", "angebot"];
  const FRZ_VALS = ["egal", "on"];
`;
const { sessionFromShare, shareFromSession } = new Function(
  PRELUDE + m[1] + "\nreturn { sessionFromShare, shareFromSession };")();

const assert = (c, msg) => { if (!c) throw new Error(msg); };

// A representative positional captureShareState() array — VALID enum indices in
// every slot (an out-of-range index has no string to round-trip through). Slot
// order mirrors captureShareState; slot 0 is SHARE_V.
const A = [
  8,                                    // SHARE_V
  [12, 30, 0, 8],                       // weights (slider steps, one per W_KEYS)
  [0, 1200, 30, 75],                    // rent [modeIdx=eur, eur, pct, m2]
  60,                                   // anbindung budget
  800,                                  // density
  1,                                    // stage idx → "familie"
  0.5,                                  // compromise
  [1, 0, 3],                            // modes [bike=gern, car=null, oepnv=nicht]
  [2, 0, 1],                            // access [muss, egal, nice]  (egal omitted when named)
  [1, 0, 1, 0.25],                      // filters [transit, rent, pop, best]
  [2, 1, 0],                            // view [__pop__, normPop, showCands]
  1,                                    // tab → "detail"
  [11.57, 48.13, 9.5],                  // camera
  [[0, 11.6, 48.2], [1, 11.4, 48.1]],   // pins
  "augsburg",                           // commute target
  [1, 0, 1, 0, 0, 1],                   // freizeit [on, egal, on, egal, egal, on]
];

Deno.test("shareFromSession ∘ sessionFromShare is the identity (drift guard)", () => {
  assertEquals(shareFromSession(sessionFromShare(A)), A);
});

Deno.test("named form keys by name, omits at-default access/freizeit, tags version", () => {
  const o = sessionFromShare(A);
  assertEquals(o.v, 1, "carries SESSION_V");
  assertEquals(o.weights, { miete: 12, anbindung: 30, natur: 0, ruhe: 8 });
  assertEquals(o.access, { nahversorgung: "muss", wasser: "nice" }, "kinder=egal omitted");
  assertEquals(o.freizeit, { kultur: "on", schwimmen: "on", angebot: "on" }, "egal sources omitted");
  assertEquals(o.modes, { bike: "gern", car: null, oepnv: "nicht" });
  assertEquals(o.stage, "familie");
  assertEquals(o.view.key, "__pop__");
  assertEquals(o.tab, "detail");
});

Deno.test("survives a removed source / layer (the SHARE_V-bump scenario)", () => {
  // A session saved when "segeln" still existed: its key is simply dropped on
  // restore (not in current FRZ_KEYS), and an absent layer/source falls back to
  // its default index — no slot misalignment, no wipe.
  const o = sessionFromShare(A);
  o.freizeit = { kultur: "on", segeln: "on" };  // segeln no longer in the build
  delete o.weights.anbindung;                    // a weight the named form happens to lack
  const back = shareFromSession(o);
  assertEquals(back[15], [1, 0, 0, 0, 0, 0], "unknown 'segeln' dropped, only kultur kept");
  assertEquals(back[1], [12, 0, 0, 8], "missing weight → default 0");
});

Deno.test("a fully-empty named object decodes to all-defaults (no throw)", () => {
  const back = shareFromSession({ v: 1 });
  assertEquals(back[1], [0, 0, 0, 0], "weights default to 0");
  assertEquals(back[8], [0, 0, 0], "access all egal");
  assertEquals(back[15], [0, 0, 0, 0, 0, 0], "freizeit all egal");
  assertEquals(back[5], -1, "missing stage → -1 (applyShareState → 'gemischt')");
  assertEquals(back[11], -1, "missing tab → -1 (applyShareState → 'detail')");
  assertEquals(back[10][0], 0, "missing view key → __composite__");
});

// minimal deep-equal so the test stays dependency-free like share/presets.test.js
function assertEquals(a, b, msg) {
  if (JSON.stringify(a) !== JSON.stringify(b))
    throw new Error((msg ? msg + ": " : "") + `expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
}
