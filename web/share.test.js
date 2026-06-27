// Round-trip test for the shareable-URL codec. Extracts the self-contained
// `<share-codec>` block from index.html (the pure half: no DOM/map refs) and
// checks that packShare → decodeShare is lossless and that junk decodes to null.
// Run: `deno test --allow-read web/`
//
// The risky part of base64url is the alphabet swap (+/ → -_) and dropped `=`
// padding: get either wrong and shared links silently fail to restore. The
// capture/apply halves are browser-coupled and verified manually (see plan).

const html = Deno.readTextFileSync(new URL("./index.html", import.meta.url));
const m = html.match(/\/\/ <share-codec>[^\n]*\n([\s\S]*?)\/\/ <\/share-codec>/);
if (!m) throw new Error("share-codec block not found in index.html");
const { packShare, decodeShare, SHARE_V, decodeSession, SESSION_V } = new Function(
  m[1] + "\nreturn { packShare, decodeShare, SHARE_V, decodeSession, SESSION_V };")();

const assert = (c, msg) => { if (!c) throw new Error(msg); };

// a realistic "everything" state, incl. the values most likely to stress the
// encoder: null mode prefs, umlaut-free pins, negatives, floats, booleans.
// SHARE_V=2 is ONE positional array; slot 0 is the version. Weights as slider
// steps 0..60, enums as indices (order mirrors captureShareState).
const STATE = [
  SHARE_V,
  [24, 0, 60, 12, 30],       // weights
  [1, 1800, 30, 100],        // rent [modeIdx, eur, pct, m2]
  80, 600, 39, 0.5,          // anbindung budget, density, age, compromise
  [1, 0, 2],                 // modes [bike, car, oepnv]
  [2, 1, 0, 1, 2, 1, 1],     // amenity prefs
  [1, 0, 1, 0.125],          // filters [transit, rent, pop, f_best]
  [1, 1, 0],                 // view [viewIdx, norm_pop, show_cands]
  2,                         // tab
  [11.5712, 48.1374, 9.5],   // map camera
  [[0, 11.6, 48.2], [1, 11.41, 48.09]],  // pins
  "augsburg",                // commute target (null | "any" | "gross" | city id)
  [1, 0, 1, 0, 0, 1],        // freizeit source prefs (kultur, kino, schwimmen, klettern, golf, angebot)
];

Deno.test("packShare → decodeShare is lossless (deep equal)", () => {
  const round = decodeShare(packShare(STATE));
  assertEquals(round, STATE);
});

Deno.test("base64url payload is URL-safe and unpadded", () => {
  const blob = packShare(STATE);
  assert(!/[+/=]/.test(blob), `blob contains non-url-safe chars: ${blob}`);
});

Deno.test("junk / truncated / foreign fragments decode to null (atomic)", () => {
  assertEquals(decodeShare("not-base64-@@@"), null);
  assertEquals(decodeShare(packShare(STATE).slice(0, -5)), null, "truncated → null");
  assertEquals(decodeShare(packShare([999, 1])), null, "wrong SHARE_V → null");
  assertEquals(decodeShare(packShare({ v: SHARE_V })), null, "non-array payload → null");
  assertEquals(decodeShare(""), null);
});

// decodeSession is the PURE half of the named browser-store codec (JSON parse +
// SESSION_V gate). The translators that need runtime config live in <session-codec>
// and are exercised by session.test.js; here we just pin the version/junk gating.
Deno.test("decodeSession accepts the current SESSION_V, rejects everything else", () => {
  assertEquals(decodeSession(JSON.stringify({ v: SESSION_V, target: "augsburg" })),
    { v: SESSION_V, target: "augsburg" });
  assertEquals(decodeSession(JSON.stringify({ v: SESSION_V + 1 })), null, "foreign SESSION_V → null");
  assertEquals(decodeSession(JSON.stringify({ target: "x" })), null, "no version → null");
  assertEquals(decodeSession("not-json-@@@"), null, "junk → null");
  assertEquals(decodeSession(""), null);
  // An old positional share blob is NOT valid named JSON → decodeSession alone → null
  // (loadSession is what migrates it; that path is browser-coupled, tested manually).
  assertEquals(decodeSession(packShare(STATE)), null, "positional blob → null (migrated by loadSession)");
});

// minimal deep-equal so the test stays dependency-free like presets.test.js
function assertEquals(a, b, msg) {
  if (JSON.stringify(a) !== JSON.stringify(b))
    throw new Error((msg ? msg + ": " : "") + `expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
}
