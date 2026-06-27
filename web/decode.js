"use strict";
// Loads + decodes the binary payload (web/data.bin) and per-target reach chunks
// (web/reach/<id>.bin) for index.html. Both are gzip-compressed raw containers
// fetched over http(s) and inflated with DecompressionStream — the map is served,
// not opened from file://. Wire format documented in scripts/22_build_web.py — keep
// the two in sync. Nullable raw columns come back as plain arrays with null,
// matching the old JSON payload exactly; scores stay Uint8Array (0-100). (Percentile
// ranks are computed client-side per target — no "pct" columns are shipped anymore.)

// Define obj[name] as a getter that builds its value once on first access, then
// replaces itself with the plain value (so every later read is a normal property,
// no getter overhead). Used to defer per-column raw boxing in decodePayload off the
// load critical path.
function defineLazy(obj, name, build) {
  Object.defineProperty(obj, name, {
    configurable: true, enumerable: true,
    get() {
      const v = build();
      Object.defineProperty(obj, name, { value: v, writable: true, enumerable: true, configurable: true });
      return v;
    },
  });
}

async function gunzip(resp) {
  if (!resp.ok) throw new Error("fetch " + resp.url + ": " + resp.status);
  const stream = resp.body.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// The base payload is loaded in two steps by index.html's boot (fetch+gunzip, then
// decodePayload) so the basemap can paint between the inflate and the heavy decode —
// see the boot comment there. gunzip + decodePayload below are the reusable pieces.

// Fetch + inflate + decode one commute-target chunk into per-mode minute arrays.
async function loadTarget(url) {
  return decodeTarget(await gunzip(await fetch(url)));
}

function decodePayload(bytes) {
  const td = new TextDecoder();
  if (td.decode(bytes.subarray(0, 4)) !== "WHN1")
    throw new Error("data.bin: bad payload magic");
  const hlen = new DataView(bytes.buffer, bytes.byteOffset).getUint32(4, true);
  const head = JSON.parse(td.decode(bytes.subarray(8, 8 + hlen)));

  const TYPES = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array };
  const SENT = { u8: 255, u16: 65535, u32: 4294967295 };
  const align = o => (o + 3) & ~3;
  const cells = { scores: {}, raw: {} };
  let off = align(8 + hlen), h3hi, h3lo;
  for (const c of head.cols) {
    const T = TYPES[c.dtype];
    const a = new T(bytes.buffer, bytes.byteOffset + off, head.n);
    off = align(off + head.n * T.BYTES_PER_ELEMENT);
    if (c.kind === "h3hi") h3hi = a;
    else if (c.kind === "h3lo") h3lo = a;
    else if (c.kind === "label")
      cells.label = Array.from(a, v => head.labels[v]);
    else if (c.kind === "inhabited") cells.inhabited = a;
    else if (c.kind === "score") cells.scores[c.name] = a;
    else if (c.kind === "raw") {
      // u8 code -> value. Three header-driven encodings (see encode_raw in
      // 22_build_web.py): log {log,lo,q}, affine {lo,q}, or scaled-int {dec}.
      // Boxing 80+ columns × ~540k cells into null-able Arrays is ~1.3 s of the
      // boot, and many columns (the commute modes, the opt-in leisure sources)
      // aren't read until the user picks a target / ticks a source. So defer each
      // column's Array.from to first access via a self-replacing getter: only the
      // ~half the initial render touches pay in at load, the rest materialize
      // lazily, off the first-paint critical path. The typed-array VIEW `a` is
      // O(1) (backed by the inflated buffer, which the closures keep alive).
      const s = SENT[c.dtype];
      const dec = c.log ? (v => Math.expm1(c.lo + v * c.q))
        : c.q !== undefined ? (v => c.lo + v * c.q)
          : (v => v / 10 ** c.dec);
      defineLazy(cells.raw, c.name, () => Array.from(a, v => v === s ? null : dec(v)));
    }
  }
  // h3 string = hex of the 64-bit id without leading zeros (the high word
  // always has mode bits set, so plain toString(16) on it is exact)
  cells.h3 = new Array(head.n);
  for (let i = 0; i < head.n; i++)
    cells.h3[i] = h3hi[i].toString(16) + h3lo[i].toString(16).padStart(8, "0");
  return { cells, layers: head.layers || [], targets: head.targets || [],
           centers: head.centers || [], outline: head.outline || {},
           branches: head.branches || [] };
}

// Per-city commute-target chunk (from web/reach/<id>.bin): consecutive u8 minute
// arrays of length CELLS.n, 255 = no-reach, one per mode in `modes` order. Same cell
// order as the base payload. Returns the arrays keyed like CELLS.raw.* so the client
// can swap them straight into recomputeAnbindung. (Wire format: 22_build_web
// REACH_MODES — keep this `modes` list in sync.)
function decodeTarget(bytes) {
  // walk_min is LAST so a chunk written before foot existed (4 modes) still decodes:
  // infer the mode COUNT from the known cell count and map only the modes actually
  // present. The first four modes are unchanged; walk_min is simply absent (→ no foot
  // term in recomputeAnbindung) until the chunk is rebuilt with it. Keep this list in
  // the same order as 22_build_web REACH_MODES.
  const modes = ["transit_hbf_min", "transit_bike_min", "bike_hbf_min", "car_hbf_min", "walk_min"];
  const n = CELLS.h3.length;                       // cells per mode (base payload order)
  const nModes = Math.round(bytes.length / n);     // 4 (legacy) or 5 (with foot)
  const out = {};
  modes.slice(0, nModes).forEach((m, k) => {
    out[m] = Array.from(bytes.subarray(k * n, (k + 1) * n), v => v === 255 ? null : v);
  });
  return out;
}

// Field (Branche) target chunk (web/reach/branche-<key>.bin) — every field target (the
// cityness "Alle Branchen" baselines and the 9 job sectors) ships the SAME M/B/O wire format:
// 4 modes × (M minutes, B floor ×254, O ceiling ×254), per-mode grouped (04f/04h via
// wohnen/mbo). Returns { mbo:{transit:{m,b,o}, bike, walk, car} } for recomputeAnbindungMBO's
// budget interp. Keep modes in sync with 22 BRANCHE_O_MODES / 04h O_MODES.
async function loadBranche(url) { return decodeMBO(await gunzip(await fetch(url))); }
function decodeMBO(bytes) {
  const modes = ["transit", "bike", "walk", "car"];
  const n = CELLS.h3.length, mbo = {};
  modes.forEach((mode, k) => {
    const base = k * 3 * n;   // per-mode grouped: M, B, O
    mbo[mode] = {
      m: Array.from(bytes.subarray(base, base + n), v => v === 255 ? null : v),
      b: Array.from(bytes.subarray(base + n, base + 2 * n), v => v === 255 ? null : v / 254),
      o: Array.from(bytes.subarray(base + 2 * n, base + 3 * n), v => v === 255 ? null : v / 254),
    };
  });
  return { mbo };
}
