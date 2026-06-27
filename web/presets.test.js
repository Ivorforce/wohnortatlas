// Unit test for the onboarding weight recipe. Extracts the self-contained
// `<preset-recipe>` block from index.html (single source of truth) and checks the
// answer→weights/controls mapping.  Run: `deno test --allow-read web/`
//
// The HARD GATE is the gradient inversion (design.md "Correlation structure"):
// the default weights imply an urban optimum the density target can't flip, so a
// "Land" answer MUST make the anti-urban bundle (quiet+nature+green) out-weigh the
// amenity bundle (access+oepnv+anbindung). If this fails, the Ruhe-&-Natur profile is
// cosmetic and rural places never win — the whole point of the presets.

const html = Deno.readTextFileSync(new URL("./index.html", import.meta.url));
const m = html.match(/\/\/ <preset-recipe>[^\n]*\n([\s\S]*?)\/\/ <\/preset-recipe>/);
if (!m) throw new Error("preset-recipe block not found in index.html");
const { presetFromAnswers, seedModes, PRESET_BASE } = new Function(
  m[1] + "\nreturn { presetFromAnswers, seedModes, PRESET_BASE };")();

const URBAN = ["s_access", "s_oepnv", "s_anbindung"];
const ANTI = ["s_quiet", "s_nature", "s_green"];
const sum = (w, keys) => keys.reduce((a, k) => a + w[k], 0);
const assert = (c, msg) => { if (!c) throw new Error(msg); };

// every Q1 × Q3 combination, so the gate can't produce a non-inverting Land view
const HOUSEHOLDS = ["familie", "paar", "single", null];
const MODE_SETS = [
  undefined,                                   // seeded
  { bike: "gern", car: "gern", oepnv: "gern" },   // worst case for inversion
  { bike: "gern", car: null, oepnv: "nicht" },
];

Deno.test("Land inverts the urban gradient for every Q1×Q3 (the hard gate)", () => {
  for (const household of HOUSEHOLDS)
    for (const modes of MODE_SETS)
      for (const priorities of [[], ["natur"]]) {
        const { weights } = presetFromAnswers({ stadtland: "land", household, modes, priorities });
        assert(sum(weights, ANTI) > sum(weights, URBAN),
          `Land must invert: anti ${sum(weights, ANTI).toFixed(2)} ≤ urban `
          + `${sum(weights, URBAN).toFixed(2)} (household=${household}, modes=${JSON.stringify(modes)})`);
      }
});

Deno.test("Innenstadt keeps the amenity bundle on top", () => {
  const { weights } = presetFromAnswers({ stadtland: "innenstadt" });
  assert(sum(weights, URBAN) > sum(weights, ANTI), "Innenstadt should favour amenities");
});

Deno.test("Ortsbild is an inverted-U over density (peaks mid, dips at both ends)", () => {
  const ch = sl => presetFromAnswers({ stadtland: sl }).weights.s_character;
  assert(ch("kleinstadt") > ch("land") && ch("kleinstadt") > ch("innenstadt"),
    "Kleinstadt Ortsbild must exceed both extremes");
  assert(ch("stadtrand") > ch("land") && ch("stadtrand") > ch("innenstadt"),
    "Stadtrand Ortsbild must exceed both extremes");
});

Deno.test("Familie seeds the whole school arc; non-family all off", () => {
  const fam = presetFromAnswers({ household: "familie" }).controls.access;
  assert(fam.kita === "nice" && fam.grundschule === "nice", "Familie → KiTa+Grundschule Gut erreichbar");
  assert(fam.gymnasium === "on" && fam.realschule === "on" && fam.mittelschule === "on",
    "Familie → all weiterführende on (long-term: keep every track reachable)");
  const single = presetFromAnswers({ household: "single" }).controls.access;
  assert(single.kita === "egal" && single.grundschule === "egal"
    && single.gymnasium === "egal", "Single → school needs off");
});

Deno.test("Familie seeds the 'familie' life-stage; everyone else neutral 'gemischt'", () => {
  const fam = presetFromAnswers({ household: "familie" });
  assert(fam.controls.life_stage === "familie", "Familie → 'Junge Familie' (kid-dense signature, not just a young mean)");
  assert(fam.weights.s_age > 0.45 && fam.weights.s_age < 0.55, "Familie → s_age bumped back to ~0.5");
  const single = presetFromAnswers({ household: "single" });
  assert(single.controls.life_stage === "gemischt", "non-family → neutral Gemischt (default, no taste imposed)");
  assert(single.weights.s_age < 0.35, "non-family → s_age stays the low 0.3 base");
});

Deno.test("'Kita & Grundschule zu Fuß' priority tightens them to walkable (family only)", () => {
  const fuss = presetFromAnswers({ household: "familie", priorities: ["kinder_fuss"] }).controls.access;
  assert(fuss.kita === "muss" && fuss.grundschule === "muss", "family + chip → Direkt vor Ort");
  const noFuss = presetFromAnswers({ household: "familie", priorities: [] }).controls.access;
  assert(noFuss.kita === "nice" && noFuss.grundschule === "nice", "family without chip → Gut erreichbar");
  const single = presetFromAnswers({ household: "single", priorities: ["kinder_fuss"] }).controls.access;
  assert(single.kita === "egal", "non-family ignores the chip");
});

Deno.test("Nahversorgung stays a need; leisure sources land in controls.freizeit", () => {
  const none = presetFromAnswers({ priorities: [] });
  assert(none.controls.access.nahversorgung === "nice", "Nahversorgung stays a need (Gut erreichbar baseline)");
  // sources default OFF — the always-on baseline gives the default "how much to do" read
  assert(none.controls.freizeit.kultur === "egal", "Nachtleben & Kultur off by default (baseline carries it)");
  assert(none.controls.freizeit.schwimmen === "egal", "Schwimmen off unless asked");
  const swim = presetFromAnswers({ priorities: ["schwimmen"] });
  assert(swim.controls.freizeit.schwimmen === "on", "Schwimmen priority → source on");
  assert(swim.weights.s_freizeit > none.weights.s_freizeit, "Schwimmen priority bumps s_freizeit weight");
  const out = presetFromAnswers({ priorities: ["ausgehen"] });
  assert(out.controls.freizeit.kultur === "on", "Ausgehen priority → steep Kultur source on");
  assert(out.weights.s_freizeit > none.weights.s_freizeit, "Ausgehen priority bumps s_freizeit weight");
  // every Freizeit source has its own chip: ticking it turns ON exactly that source + bumps s_freizeit
  for (const [pri, key] of [["kino", "kino"], ["klettern", "klettern"], ["golf", "golf"]]) {
    const r = presetFromAnswers({ priorities: [pri] });
    assert(r.controls.freizeit[key] === "on", `${pri} priority → ${key} source on`);
    assert(r.controls.freizeit.schwimmen === "egal", `${pri} priority leaves other sources off`);
    assert(r.weights.s_freizeit > none.weights.s_freizeit, `${pri} priority bumps s_freizeit weight`);
  }
});

Deno.test("non-amenity priorities bump the right layer (Ruhe/Wetter/Internet)", () => {
  const base = presetFromAnswers({}).weights;
  const q = p => presetFromAnswers({ priorities: [p] }).weights;
  assert(q("ruhe").s_quiet > base.s_quiet, "Ruhe → s_quiet up");
  assert(q("wetter").s_climate > base.s_climate, "Gutes Wetter → s_climate up");
  assert(q("internet").s_broadband > base.s_broadband, "Schnelles Internet → s_broadband up");
  assert(q("ortsbild").s_character > base.s_character, "Bemerkenswertes Ortsbild → s_character up");
  assert(q("gruen").s_green > base.s_green, "Grünes Viertel → s_green up");
  // natur is erreichbare Natur only — it must NOT drive grün vor der Tür anymore
  assert(q("natur").s_green === base.s_green, "Natur must not bump s_green (that's gruen's job)");
});

Deno.test("Günstig/Wohnraum/Anbindung each tighten their ANCHOR (+ gentle weight bump)", () => {
  const baseP = presetFromAnswers({ household: "paar", rent_eur: 1800 });
  const base = baseP.weights, baseC = baseP.controls;
  const q = p => presetFromAnswers({ household: "paar", rent_eur: 1800, priorities: [p] });
  const ceil = c => c.rent_eur / c.rent_m2;  // €/m² affordability ceiling
  // Günstig → budget cut (% and €) is the MAIN move; m² drags down ~⅓ as far (frugal
  // accept a bit less space). Net: lower ceiling. Small s_rent bump.
  const g = q("guenstig");
  assert(g.controls.rent_pct < baseC.rent_pct && g.controls.rent_eur < baseC.rent_eur, "Günstig → budget down");
  assert(g.controls.rent_m2 < baseC.rent_m2, "Günstig → m² nudged down (coupling)");
  assert(ceil(g.controls) < ceil(baseC), "Günstig → €/m² ceiling down (budget cut dominates)");
  assert(g.weights.s_rent > base.s_rent, "Günstig → s_rent gently up");
  // Viel Wohnraum → more m² is the MAIN move; budget drags up ~⅓ as far (willing to
  // pay more). Net still lowers the ceiling, just less. Small s_rent bump.
  const vr = q("viel_raum");
  assert(vr.controls.rent_m2 > baseC.rent_m2, "Viel Wohnraum → rent_m2 up");
  assert(vr.controls.rent_eur > baseC.rent_eur, "Viel Wohnraum → budget nudged up (coupling)");
  assert(ceil(vr.controls) < ceil(baseC), "Viel Wohnraum → €/m² ceiling down (m² raise dominates)");
  assert(vr.weights.s_rent > base.s_rent, "Viel Wohnraum → s_rent gently up");
  // Picking BOTH still net-stacks toward cheap-per-m² (the compensations don't cancel).
  const both = presetFromAnswers({ household: "paar", rent_eur: 1800, priorities: ["guenstig", "viel_raum"] });
  assert(ceil(both.controls) < ceil(g.controls) && ceil(both.controls) < ceil(vr.controls),
    "Günstig + Viel Wohnraum → tighter ceiling than either alone");
  // Schnelle Anbindung → tighter time-budget (min), small s_anbindung bump
  const an = q("anbindung");
  assert(an.controls.anbindung_budget < baseC.anbindung_budget, "Anbindung → time-budget down");
  assert(an.weights.s_anbindung > base.s_anbindung, "Anbindung → s_anbindung gently up");
});

Deno.test("household drives needed m² and pay-% (Paar > Familie > Single)", () => {
  const c = h => presetFromAnswers({ household: h }).controls;
  assert(c("single").rent_m2 === 50 && c("paar").rent_m2 === 70 && c("familie").rent_m2 === 95,
    "m² ladder 50/70/95 (mover targets, not occupancy averages)");
  // default income level → multiple 1.0, so rent_pct = household share × 100
  assert(c("single").rent_pct < c("familie").rent_pct
    && c("familie").rent_pct < c("paar").rent_pct,
    "pay-% Single < Familie < Paar");
  assert(presetFromAnswers({}).controls.rent_pct === 34, "no household → Egal mix-weighted avg 34 %");
});

// Personas: realistic households → what the gate produces → does the implied €/m²
// AFFORDABILITY land where it should? The budget is a MAX; the *comfortable* target is
// ≈0.81× that (RENT_REL_MID). Anchors from rent.parquet (2026): kreis_pay (full-time
// gross €/mo) rural≈3450 / national-median≈3726 / Munich-city≈4850; actual rents
// (rent_cal €/m²) median 11.4, p95 16.9, Munich centre up to ~24.
Deno.test("personas land in sensible €/m² affordability ranges", () => {
  const PAY = { rural: 3450, median: 3726, munich: 4850 };
  const MID = 1 - 3.5 / 18;                                  // comfortable share of the ceiling
  const maxEm2 = (c, pay) => (c.rent_pct / 100) * pay / c.rent_m2;   // ceiling €/m² (the "max")
  const tgt = (c, pay) => MID * maxEm2(c, pay);                      // comfortable €/m²
  const persona = a => presetFromAnswers({ income_mode: "pay", ...a }).controls;

  // 1) Average single, local job. Targets a small flat; comfortable ≈ your ~30%-of-net
  //    reality. Affords median areas, NOT the priciest — the layer must discriminate.
  const single = persona({ household: "single", income_level: "schnitt" });
  assert(single.rent_m2 === 50, "single targets ~50 m²");
  assert(tgt(single, PAY.median) > 12 && tgt(single, PAY.median) < 16,
    `single@median comfortable ${tgt(single, PAY.median).toFixed(1)} €/m² (affords median, not p95+)`);
  assert(tgt(single, PAY.munich) > 15 && tgt(single, PAY.munich) < 21,
    `single@Munich-wage comfortable ${tgt(single, PAY.munich).toFixed(1)} €/m² (lower Munich, priced out of the ~23 centre)`);
  assert(maxEm2(single, PAY.munich) < 24.4, "even a Munich-wage single's MAX stays under the centre peak");

  // 2) Average family in a cheap rural area: should comfortably afford local rents (~7–10).
  const ruralFam = persona({ household: "familie", income_level: "schnitt" });
  assert(ruralFam.rent_m2 === 95, "family targets ~95 m²");
  assert(tgt(ruralFam, PAY.rural) > 8 && tgt(ruralFam, PAY.rural) < 12,
    `rural family comfortable ${tgt(ruralFam, PAY.rural).toFixed(1)} €/m² (affords rural)`);

  // 3) Below-average-income family at a Munich wage level: priced out → "live elsewhere".
  const poorFam = persona({ household: "familie", income_level: "unter" });
  assert(tgt(poorFam, PAY.munich) < 13,
    `below-avg family comfortable ${tgt(poorFam, PAY.munich).toFixed(1)} €/m² « Munich rents 17–24 → steered to cheaper regions`);

  // 4) Above-average dual-income couple who wants space: lots of m² (Viel Wohnraum ×
  //    the income space-bump), still affords most of Munich.
  const richCouple = persona({ household: "paar", income_level: "ueber", priorities: ["viel_raum"] });
  assert(richCouple.rent_m2 > 100, "Viel Wohnraum + above-median income → 100+ m²");
  assert(tgt(richCouple, PAY.munich) > 20 && tgt(richCouple, PAY.munich) < 30,
    `well-off couple comfortable ${tgt(richCouple, PAY.munich).toFixed(1)} €/m² (affords most of Munich)`);

  // 5) Festeinkommen (eur mode): household-aware default budget, ~17–20 €/m² over HH_M2 —
  //    no kreis_pay involved (you bring your own income). Family budget > single's, but
  //    a touch tighter per-m² (more space). None near the old flat-1800 = 36 €/m² single.
  const eurC = h => presetFromAnswers({ income_mode: "eur", household: h }).controls;
  const eurEm2 = c => c.rent_eur / c.rent_m2;
  const eSingle = eurC("single"), eFam = eurC("familie");
  assert(eSingle.rent_eur < 1800 && eFam.rent_eur > eSingle.rent_eur, "eur default: household-sized, all < old 1800");
  assert(eurEm2(eSingle) > 17 && eurEm2(eSingle) < 22, `eur single ${eurEm2(eSingle).toFixed(1)} €/m² ceiling`);
  assert(eurEm2(eFam) > 14 && eurEm2(eFam) < 20, `eur family ${eurEm2(eFam).toFixed(1)} €/m² ceiling`);
});

Deno.test("income nudges the space standard — mildly, pay-mode only", () => {
  const m2 = (lvl, mode = "pay") => presetFromAnswers({ household: "paar", income_mode: mode, income_level: lvl }).controls.rent_m2;
  assert(m2("unter") < m2("schnitt") && m2("schnitt") < m2("ueber"), "more income → more expected space");
  assert(m2("ueber") - m2("unter") < 0.3 * m2("schnitt"), "but mild (space-elasticity ~0.3, budget carries the rest)");
  assert(m2("ueber", "eur") === m2("schnitt", "eur"), "eur mode (bring-your-own income): level doesn't move space");
});

Deno.test("income level scales the collapsed pay-% (the gate's only job there)", () => {
  const p = lvl => presetFromAnswers({ household: "paar", income_level: lvl }).controls.rent_pct;
  assert(p("unter") < p("schnitt") && p("schnitt") < p("ueber"),
    "above-median earners get a higher budget %");
});

Deno.test("income level nudges s_rent weight; below > base > above", () => {
  const w = lvl => presetFromAnswers({ income_level: lvl }).weights.s_rent;
  assert(w("unter") > w("schnitt") && w("schnitt") > w("ueber"),
    "rent weight: unter > schnitt > ueber");
});

Deno.test("Festeinkommen raises broadband + ambient; Lokaler Beruf doesn't", () => {
  const eur = presetFromAnswers({ income_mode: "eur" }).weights;
  const pay = presetFromAnswers({ income_mode: "pay" }).weights;
  assert(eur.s_broadband > pay.s_broadband, "Festeinkommen raises broadband");
  assert(eur.s_quiet > pay.s_quiet && eur.s_green > pay.s_green, "Festeinkommen raises ambient");
  assert(presetFromAnswers({ income_mode: "eur" }).controls.rent_mode === "eur"
    && presetFromAnswers({}).controls.rent_mode === "pay", "rent_mode follows the tab; pay default");
});

Deno.test("Q2 'egal' damps the density weight and keeps a neutral target", () => {
  const c = presetFromAnswers({});  // no stadtland
  assert(c.weights.s_density < PRESET_BASE.s_density, "egal damps s_density weight");
  assert(c.controls.density_target === 2500, "egal → neutral Kleinstadt target (Einw./km² bebaut)");
});

Deno.test("an explicit Stadt/Land choice raises the density-preference weight (flat)", () => {
  for (const sl of ["land", "kleinstadt", "stadtrand", "innenstadt"])
    assert(presetFromAnswers({ stadtland: sl }).weights.s_density > PRESET_BASE.s_density,
      `${sl} → s_density weight above base`);
});

Deno.test("density target follows Stadt/Land", () => {
  const dt = sl => presetFromAnswers({ stadtland: sl }).controls.density_target;
  assert(dt("land") === 900 && dt("kleinstadt") === 2500
    && dt("stadtrand") === 5000 && dt("innenstadt") === 12000, "density target ladder (Einw./km² bebaut)");
});

Deno.test("Nahversorgung walkable (muss) for town/suburb/city; reachable (nice) for Land", () => {
  const nah = sl => presetFromAnswers({ stadtland: sl }).controls.access.nahversorgung;
  assert(nah("kleinstadt") === "muss" && nah("stadtrand") === "muss" && nah("innenstadt") === "muss",
    "urbaner choices → Direkt vor Ort (walkable food)");
  assert(nah("land") === "nice" && nah(null) === "nice",
    "Land / no choice → Gut erreichbar (driving to groceries accepted)");
});

Deno.test("all weights stay within the slider range [0,9]", () => {
  for (const household of HOUSEHOLDS)
    for (const stadtland of ["land", "kleinstadt", "stadtrand", "innenstadt", null])
      for (const income_mode of ["pay", "eur"])
        for (const income_level of ["unter", "schnitt", "ueber"]) {
          const { weights } = presetFromAnswers({ household, stadtland, income_mode, income_level,
            priorities: ["natur", "gruen", "ortsbild", "ausgehen", "sport", "schwimmen", "ruhe", "wetter", "internet"] });
          for (const k in weights)
            assert(weights[k] >= 0 && weights[k] <= 9, `${k}=${weights[k]} out of [0,9]`);
        }
});

// --- the two default-ON weighting pills (Lebensqualität bundle + Sicherheit) ---
// Deselecting "Lebensqualität" drops the opinion baseline + persona SMART guesses, so
// ONLY the user's explicit picks (Stadt/Land, modes, priority chips, income) + s_rent score.

Deno.test("Lebensqualität off, nothing else → only s_rent scores", () => {
  const bare = presetFromAnswers({ quality: false, sicherheit: false });
  assert(bare.weights.s_rent > 0, "rent always scores (core; income defaults to schnitt)");
  for (const k in bare.weights)
    if (k !== "s_rent") assert(bare.weights[k] === 0, `${k}=${bare.weights[k]} must be 0 with nothing picked`);
});

Deno.test("Lebensqualität off: household seeds NO weights — only Wohnfläche + Budget survive", () => {
  const fam = presetFromAnswers({ quality: false, household: "familie" });
  for (const k of ["s_quiet", "s_green", "s_access", "s_age"])
    assert(fam.weights[k] === 0, `familie smart-guess ${k} must be 0`);
  assert(fam.controls.life_stage === "gemischt", "life_stage neutralised (s_age match is a smart guess)");
  assert(fam.controls.access.kita === "egal" && fam.controls.access.gymnasium === "egal",
    "family school seeds dropped");
  assert(fam.controls.rent_m2 === 95, "but household m² target survives");
  assert(fam.controls.rent_pct > 0, "and household budget survives");
});

Deno.test("Lebensqualität off: explicit picks still score, untouched layers stay 0", () => {
  const gruen = presetFromAnswers({ quality: false, priorities: ["gruen"] });
  assert(gruen.weights.s_green > 0, "explicit Grünes Viertel still scores");
  assert(gruen.weights.s_quiet === 0, "an untouched layer stays 0");
  // explicit family kid-walk chip binds regardless of the bundle
  const fuss = presetFromAnswers({ quality: false, household: "familie", priorities: ["kinder_fuss"] });
  assert(fuss.controls.access.kita === "muss" && fuss.controls.access.grundschule === "muss",
    "explicit kinder_fuss binds even with the bundle off");
});

Deno.test("Lebensqualität off: explicit Stadt/Land still tilts the gradient", () => {
  const land = presetFromAnswers({ quality: false, stadtland: "land" });
  for (const k of ANTI) assert(land.weights[k] > 0, `land keeps ${k} (explicitly elevated)`);
  for (const k of URBAN) assert(land.weights[k] === 0, `land drops ${k} (down-mul → not my preference)`);
  assert(land.weights.s_density > 0, "explicit Stadt/Land keeps the density-preference weight");
  const city = presetFromAnswers({ quality: false, stadtland: "innenstadt" });
  for (const k of URBAN) assert(city.weights[k] > 0, `innenstadt keeps ${k}`);
  assert(city.controls.access.nahversorgung === "muss", "innenstadt keeps walkable food (explicit)");
});

Deno.test("Sicherheit pill gates the flood weight, independent of Lebensqualität", () => {
  assert(presetFromAnswers({ sicherheit: true }).weights.s_flood > 0, "sicherheit on → flood scores");
  assert(presetFromAnswers({ sicherheit: false }).weights.s_flood === 0, "sicherheit off → flood 0");
  assert(presetFromAnswers({ quality: false, sicherheit: true }).weights.s_flood > 0,
    "flood survives a deselected Lebensqualität bundle");
  assert(presetFromAnswers({ quality: true, sicherheit: false }).weights.s_flood === 0,
    "flood off even with Lebensqualität on");
});

Deno.test("'Miete egal' (rent_off) unweights rent AND drops its budget filter", () => {
  const off = presetFromAnswers({ rent_off: true });
  assert(off.weights.s_rent === 0, "rent_off → s_rent weight 0");
  assert(off.controls.rent_filter === false, "rent_off → 'über Budget ausblenden' off");
  const on = presetFromAnswers({});
  assert(on.weights.s_rent > 0 && on.controls.rent_filter === true, "default rates rent + keeps the filter");
  // overrides the income basis, and survives a deselected Lebensqualität bundle
  assert(presetFromAnswers({ income_mode: "eur", rent_off: true }).weights.s_rent === 0, "rent_off overrides eur basis");
  assert(presetFromAnswers({ quality: false, rent_off: true }).weights.s_rent === 0, "rent_off + bundle off → no rent");
});

Deno.test("the pills default ON — omitting them == quality:true, sicherheit:true", () => {
  const a = { household: "familie", stadtland: "kleinstadt", priorities: ["ruhe", "gruen"] };
  const def = presetFromAnswers(a), on = presetFromAnswers({ ...a, quality: true, sicherheit: true });
  for (const k in def.weights) assert(def.weights[k] === on.weights[k], `${k}: default must equal quality:true`);
});

Deno.test("seedModes: city→ÖPNV, country→car, edge resolves by household", () => {
  assert(seedModes(null, "innenstadt").oepnv === "gern", "Innenstadt → ÖPNV gern");
  assert(seedModes(null, "land").car === "gern", "Land → Auto gern");
  assert(seedModes("familie", "stadtrand").car === "gern", "Familie@Stadtrand → Auto gern");
  assert(seedModes("single", "stadtrand").oepnv === "gern", "Single@Stadtrand → ÖPNV gern");
});
