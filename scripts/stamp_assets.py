#!/usr/bin/env python3
"""Stamp content-hashed filenames into the deploy bundle (run by publish.sh).

Operates on the throwaway TMP deploy dir, NOT the source web/ — so the source tree
keeps plain names and the dev server (`python3 -m http.server`) is unaffected.

Why: data.bin / decode.js / the reach chunks share a binary format, so an old
index.html paired with a new asset (or vice versa) decodes to garbage. Hashing the
filenames makes the HTML name the exact versions it expects; those names don't exist
until their bytes are deployed, so a stale edge can never assemble a mixed set — and a
mid-deploy origin returns 404 (fail-safe, self-healed client-side) rather than serving
wrong bytes under a shared name. Vendor libs are hashed too (a bump is the same risk).

Hash = first 12 hex of sha256(bytes); content-derived, so a name only moves when the
bytes do. Aborts loudly if any expected reference is missing — a dangling ref would
404 the live site, so a failed stamp must fail the publish.
"""
import hashlib
import sys
from pathlib import Path

HASH_LEN = 12

# (relative path, exact ref string in index.html, replacement template using {new}).
# Each ref must occur exactly once; a mismatch aborts (a refactor that renames the
# reference must be reflected here, else we'd ship an unhashed or dangling asset).
SINGLE_ASSETS = [
    ("data.bin", '"data.bin"', '"{new}"'),
    ("decode.js", 'src="decode.js"', 'src="{new}"'),
    ("vendor/maplibre-gl.js", "vendor/maplibre-gl.js", "{new}"),
    ("vendor/maplibre-gl.css", "vendor/maplibre-gl.css", "{new}"),
    ("vendor/h3-js.umd.js", "vendor/h3-js.umd.js", "{new}"),
    ("vendor/deck.gl.min.js", "vendor/deck.gl.min.js", "{new}"),
]

REACH_MARKER = '// PUBLISH:reach-dir'


def _hashed(rel: str, h: str) -> str:
    """data.bin -> data.<h>.bin ; vendor/deck.gl.min.js -> vendor/deck.gl.min.<h>.js"""
    p = Path(rel)
    return str(p.with_name(f"{p.stem}.{h}{p.suffix}"))


def main() -> None:
    root = Path(sys.argv[1])
    html_path = root / "index.html"
    html = html_path.read_text()
    renames: list[tuple[Path, Path]] = []

    for rel, ref_old, ref_tmpl in SINGLE_ASSETS:
        src = root / rel
        if not src.exists():
            sys.exit(f"stamp_assets: missing {rel}")
        new_rel = _hashed(rel, hashlib.sha256(src.read_bytes()).hexdigest()[:HASH_LEN])
        n = html.count(ref_old)
        if n != 1:
            sys.exit(f"stamp_assets: expected exactly 1x {ref_old!r} in index.html, found {n}")
        html = html.replace(ref_old, ref_tmpl.format(new=new_rel))
        renames.append((src, root / new_rel))

    # reach/ is a dir of lazy per-city chunks; hash all of them into one build token,
    # move them under reach/<token>/, and stamp the runtime-built path's REACH_DIR const.
    reach = root / "reach"
    if not reach.is_dir():
        sys.exit("stamp_assets: missing reach/")
    digest = hashlib.sha256()
    for f in sorted(p for p in reach.rglob("*") if p.is_file()):
        digest.update(f.relative_to(reach).as_posix().encode())
        digest.update(f.read_bytes())
    rh = digest.hexdigest()[:HASH_LEN]

    marker_line = next((ln for ln in html.splitlines() if REACH_MARKER in ln), None)
    if marker_line is None:
        sys.exit(f"stamp_assets: {REACH_MARKER!r} not found in index.html")
    html = html.replace(marker_line, f'const REACH_DIR = "reach/{rh}";  {REACH_MARKER} (stamped)')

    # Move chunks into reach/<token>/ (via a sibling temp so the target can live inside reach/).
    staging = root / f".reach-{rh}"
    reach.rename(staging)
    (root / "reach").mkdir()
    staging.rename(root / "reach" / rh)

    for src, dst in renames:
        src.rename(dst)
    html_path.write_text(html)

    print(f"stamp_assets: hashed data/decode + {len(SINGLE_ASSETS) - 2} vendor + reach/{rh}")


if __name__ == "__main__":
    main()
