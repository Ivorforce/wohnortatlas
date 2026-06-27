SHELL := /bin/zsh
PY := poetry run python
JAVA_HOME_21 := $(shell brew --prefix openjdk@21 2>/dev/null)/libexec/openjdk.jdk/Contents/Home

RAW := data/raw
INT := data/interim
LAY := data/layers

# single source: wohnen/config.py (stdlib-only import; lazy `=` so the
# shell-out only happens when a recipe actually uses it)
BBOX_CSV = $(shell python3 -c "from wohnen.config import BBOX; print(','.join(map(str, BBOX)))")

.PHONY: all check-env download web publish clean-layers jobs freshness

all: web

check-env:
	@which osmium >/dev/null || (echo "missing: brew install osmium-tool" && exit 1)
	@test -x "$(JAVA_HOME_21)/bin/java" || (echo "missing: brew install openjdk@21" && exit 1)
	@"$(JAVA_HOME_21)/bin/java" -version 2>&1 | grep -q 'version "21' || (echo "JDK at $(JAVA_HOME_21) is not 21" && exit 1)
	@poetry run python -c "import geopandas, h3, osmium" || (echo "run: poetry install" && exit 1)
	@echo "env OK (JAVA_HOME=$(JAVA_HOME_21))"

download:
	$(PY) scripts/00_download.py

# ages of the time-sensitive raw inputs; GTFS expires ~7 days after download
freshness:
	@now=$$(date +%s); \
	for f in $(RAW)/gtfs-germany.zip $(RAW)/germany-latest.osm.pbf \
	         $(RAW)/breitband_gitterzellen.gpkg.zip; do \
		if [ -f "$$f" ]; then \
			age=$$(( (now - $$(stat -f %m "$$f")) / 86400 )); \
			printf "%4d d  %s\n" $$age "$$f"; \
			case "$$f" in *gtfs*) [ $$age -ge 6 ] && \
				echo "        ^ WARNING: GTFS likely expired (r5py degrades to walk/bike silently)";; esac; \
		else \
			echo "MISSING  $$f"; \
		fi; \
	done; true

# ---------- clip & base ----------
$(INT)/region.osm.pbf: $(RAW)/germany-latest.osm.pbf
	mkdir -p $(INT)
	osmium extract --overwrite --strategy complete_ways --bbox $(BBOX_CSV) -o $@ $<

$(INT)/region-filtered.osm.pbf: $(INT)/region.osm.pbf
	osmium tags-filter --overwrite -o $@ $< nwr/amenity nwr/leisure nwr/shop nwr/place nwr/highway nwr/railway nwr/healthcare nwr/natural nwr/tourism nwr/waterway nwr/historic

$(INT)/gtfs_region.zip: $(RAW)/gtfs-germany.zip
	$(PY) scripts/01_clip_gtfs.py

$(INT)/grid.parquet:
	$(PY) scripts/02_grid.py

$(INT)/pois.parquet $(INT)/roads.parquet $(INT)/green_areas.parquet $(INT)/no_access.parquet $(INT)/water_quality.parquet $(INT)/trees.parquet: $(INT)/region-filtered.osm.pbf
	$(PY) scripts/03_pois.py

# minimal POI subset for the expensive 04d_swim routing — extracted content-aware (03b),
# so regenerating pois.parquet only invalidates reach_spots.npz when these rows change.
# 03b + the SOURCES spec (wohnen/freizeit.py) are prereqs so editing the source CATEGORIES
# re-extracts (→ re-routes only if the points move); editing a MASS/τ is still a manual
# re-route (`rm reach_spots.npz`), as the masses are baked into the persisted gravity sums.
$(INT)/freizeit_spots.parquet: $(INT)/pois.parquet scripts/03b_freizeit_spots.py wohnen/freizeit.py
	$(PY) scripts/03b_freizeit_spots.py

# ---------- layers ----------
# configurable commute target (Lebensmittelpunkt): select center hexes, then
# reverse-route each → per-city reach chunks + the any-city/Großstadt aggregate maps.
$(LAY)/centers.parquet: $(INT)/grid.parquet $(LAY)/population.parquet
	$(PY) scripts/04b_centers.py

$(LAY)/entertainment.parquet: $(INT)/pois.parquet $(INT)/grid.parquet
	$(PY) scripts/05_entertainment.py

# national stop↔cell egress tables (walk + regular bike), precomputed ONCE — the
# origin-independent half of transit door-to-door. 04c min-pluses these against the
# per-center honest stop times instead of R5 rebuilding the egress linkage per batch
# (the old ~14 h cost). Depends only on the net + grid, so layer edits never re-route.
$(LAY)/egress.npz: $(INT)/region-filtered.osm.pbf $(INT)/gtfs_region.zip $(INT)/grid.parquet
	JAVA_HOME=$(JAVA_HOME_21) $(PY) scripts/04g_egress.py

# reverse-route centers → persist reach_centers.npz, the raw per-center reach (the ONLY
# expensive step; 04e/04f derive everything from it). npz is a named target so they depend.
# Transit modes decompose against egress.npz (honest stop times + cached egress).
$(LAY)/reach_centers.npz: $(LAY)/centers.parquet $(LAY)/egress.npz $(INT)/region-filtered.osm.pbf $(INT)/gtfs_region.zip
	JAVA_HOME=$(JAVA_HOME_21) $(PY) scripts/04c_reach.py

# derive ALL route-free commute targets from the per-center reach — no r5py, cheap, the
# tuning knob (per-city percentile / decay reference / tiers). Script IS a prereq, like
# 04e: editing it + `make web` re-derives with NO re-routing. Produces the per-city/part
# reach (catchment-weighted percentile) + the any/gross aggregate maps in one pass.
$(LAY)/reach_cityness.npz $(LAY)/reach_cities.npz $(LAY)/cities.parquet $(LAY)/reach_parts.npz $(LAY)/parts.parquet: $(LAY)/reach_centers.npz $(LAY)/centers.parquet $(LAY)/population.parquet scripts/04f_aggregate.py
	$(PY) scripts/04f_aggregate.py

# Branche (sector) Anbindung targets: GENESIS Unternehmensregister (52111-07-01-4) per
# Kreis × WZ-Abschnitt → jobs_kreis.parquet, then the SAME 04f aggregate argmax with O =
# sector jobs → reach_branche.npz (22 emits one lazy chunk per sector). Needs
# GENESIS_USER/GENESIS_PASS (free regionalstatistik.de account) on the FIRST pull, cached
# after; exits 0 / writes nothing if unset, so `make web` works without it. Script is a
# prereq like 04e/04f — tuning T_REF / JOBS_O_HALF_PCT = rerun (no routing). Optional:
# web/data.bin picks the npz up via $(wildcard) only if present. Run: `make jobs && make web`.
$(LAY)/reach_branche.npz: $(LAY)/reach_centers.npz $(LAY)/centers.parquet scripts/04h_jobs.py
	$(PY) scripts/04h_jobs.py
jobs: $(LAY)/reach_branche.npz

# reverse-route s_freizeit point sources (swim/kino/klettern/golf) → persist
# reach_spots.npz (raw gravity sums). ~1 h horizon. JDK21 + valid GTFS like 04c.
# 04d routes ONLY from freizeit_spots.parquet (the swim/kino/klettern/golf rows),
# NOT the full pois.parquet — so unrelated POI edits don't re-route (03b writes it content-
# aware, so it only changes when those specific spots move).
$(LAY)/reach_spots.npz: $(INT)/freizeit_spots.parquet $(INT)/region-filtered.osm.pbf $(INT)/gtfs_region.zip $(INT)/grid.parquet
	JAVA_HOME=$(JAVA_HOME_21) $(PY) scripts/04d_swim.py

# derive ALL s_freizeit surfaces from the routing caches — no r5py, cheap, the tuning
# knob. The script IS a prereq (unlike other layers): editing a τ/SAT_K constant and
# running `make web` then re-derives with no manual step (it can only reweight, not
# re-route). Re-runs when a route changes (npz) or the going-out mass (entertainment).
$(LAY)/freizeit.parquet: $(LAY)/reach_centers.npz $(LAY)/reach_spots.npz $(LAY)/entertainment.parquet scripts/04e_freizeit.py
	$(PY) scripts/04e_freizeit.py

$(LAY)/rent.parquet: $(INT)/grid.parquet $(LAY)/demographics.parquet
	$(PY) scripts/06_rent.py

# 07a isolates the headcount-grid products (population, catchment sums) that the routing
# reads; 07 enriches that file with the age/life-stage character fields. demographics.parquet
# stays a SUPERSET so non-routing consumers (rent/noise/nature/assemble) are unchanged.
# 07a is a prereq + the file is written content-aware, so editing the pop logic re-extracts
# but only re-routes (04b/04c) when the headcounts actually move.
$(LAY)/population.parquet: $(INT)/grid.parquet scripts/07a_population.py
	$(PY) scripts/07a_population.py

$(LAY)/demographics.parquet: $(INT)/grid.parquet $(LAY)/population.parquet
	$(PY) scripts/07_demographics.py

$(LAY)/noise.parquet: $(INT)/pois.parquet $(INT)/grid.parquet
	$(PY) scripts/08_noise.py

$(LAY)/greenness.parquet: $(INT)/grid.parquet $(INT)/green_areas.parquet $(INT)/no_access.parquet $(INT)/water_quality.parquet $(INT)/trees.parquet $(INT)/roads.parquet
	$(PY) scripts/09_greenness.py

$(LAY)/character.parquet: $(INT)/grid.parquet $(INT)/region-filtered.osm.pbf
	$(PY) scripts/10_denkmal.py

$(LAY)/broadband.parquet: $(INT)/grid.parquet
	$(PY) scripts/11_broadband.py

# coords for the address-only JedeSchule schools (no-coord Länder) from the official
# Land cadastre (OpenAddresses) — re-runs on a JedeSchule, OA-file, or geocode-logic change.
$(INT)/schools_geocoded.parquet: $(RAW)/jedeschule.csv scripts/03d_geocode.py wohnen/geocode.py
	$(PY) scripts/03d_geocode.py

# per-track school points: OSM locations authoritatively typed from JedeSchule (CC0),
# OSM name-typing as fallback. Cheap spatial match, no routing — re-runs on a pois,
# JedeSchule, geocode, or bucketing-logic change.
$(INT)/schools_points.parquet: $(INT)/pois.parquet $(RAW)/jedeschule.csv $(INT)/schools_geocoded.parquet scripts/03c_schools.py wohnen/schools.py
	$(PY) scripts/03c_schools.py

$(LAY)/schools.parquet $(LAY)/family.parquet: $(INT)/pois.parquet $(INT)/grid.parquet $(INT)/schools_points.parquet
	$(PY) scripts/12_schools_family.py

$(LAY)/nature.parquet: $(INT)/pois.parquet $(LAY)/greenness.parquet
	$(PY) scripts/13_nature.py

$(LAY)/oepnv.parquet: $(INT)/gtfs_region.zip $(INT)/grid.parquet
	$(PY) scripts/14_oepnv.py

$(LAY)/climate.parquet: $(INT)/grid.parquet
	$(PY) scripts/15_climate.py

$(LAY)/flood.parquet: $(INT)/grid.parquet
	$(PY) scripts/16_flood.py

$(LAY)/streets.parquet: $(INT)/region-filtered.osm.pbf $(INT)/grid.parquet
	$(PY) scripts/17_streets.py

$(LAY)/vacancy.parquet: $(INT)/grid.parquet $(RAW)/zensus_vacancy.zip scripts/18_vacancy.py
	$(PY) scripts/18_vacancy.py

LAYER_FILES := $(LAY)/entertainment.parquet $(LAY)/rent.parquet \
	$(LAY)/demographics.parquet $(LAY)/noise.parquet $(LAY)/greenness.parquet \
	$(LAY)/character.parquet $(LAY)/broadband.parquet $(LAY)/schools.parquet \
	$(LAY)/family.parquet $(LAY)/climate.parquet $(LAY)/flood.parquet \
	$(LAY)/streets.parquet $(LAY)/vacancy.parquet

# assemble tolerates missing layers; don't force all as hard deps. aggregates +
# freizeit IS forced — 20_assemble folds all reach_* s_freizeit surfaces into scores,
# so it must run first. (Anbindung is no longer a scores column — the cityness + sector
# field targets ship as reach_cityness.npz / reach_branche.npz chunks, fed straight to 22.)
$(LAY)/scores.parquet: $(INT)/grid.parquet $(LAY)/freizeit.parquet
	$(PY) scripts/20_assemble.py

# 22 reads scores + cities/reach_cities/reach_cityness/reach_branche for the manifest & chunks
web/data.bin: $(LAY)/scores.parquet $(LAY)/cities.parquet $(LAY)/reach_cities.npz $(LAY)/reach_cityness.npz $(wildcard $(LAY)/reach_branche.npz)
	$(PY) scripts/22_build_web.py

web: web/data.bin

# Force-push the CURRENT built bundle in web/ to the DigitalOcean deploy repo as a single
# snapshot commit (scripts/publish.sh). Deliberately has NO build prereqs: publishing only
# ships what's already in web/ — it must never trigger the (expensive, GTFS-sensitive)
# routing pipeline. Rebuild the data on purpose with `make web` first when you've tuned
# something; the script errors if web/data.bin is missing.
publish:
	bash scripts/publish.sh

clean-layers:
	rm -f $(LAY)/*.parquet web/data.bin
