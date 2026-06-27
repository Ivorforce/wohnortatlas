#!/usr/bin/env python3
"""Build the production favicon bundle from the chosen candidate (D: cluster + W).

Emits into web/:
  favicon.svg          self-contained vector (W outlined to a path — no font dep)
  favicon.ico          16/32/48 multi-res for legacy/Google
  apple-touch-icon.png 180, opaque white bg (iOS home screen)
  maskable-512.png     512, opaque, safe-zone padded (Android adaptive/PWA)
  site.webmanifest     PWA manifest referencing the above

Needs: rsvg-convert (librsvg, system), fontTools + Pillow (build-only). Run when
the mark changes — not a pipeline stage:

    /tmp/favvenv/bin/python scripts/make_favicon.py   # venv with fonttools+pillow
"""
import json
import math
import subprocess
import tempfile
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

GOLD, GREEN, TEAL, INK = "#F4C300", "#5EC962", "#21918C", "#15233A"
EDGE = ('stroke="#0c1b2e" stroke-opacity="0.16" stroke-width="1.4" '
        'stroke-linejoin="round"')
WFONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
POINTY = (-90, -30, 30, 90, 150, 210)


def hexpoly(cx, cy, r, fill, edge=True):
    pts = " ".join(f"{cx + r*math.cos(math.radians(a)):.2f},"
                   f"{cy + r*math.sin(math.radians(a)):.2f}" for a in POINTY)
    return f'<polygon points="{pts}" fill="{fill}" {EDGE if edge else ""}/>'


def w_path(cx, cy, cap):
    """Arial-Bold 'W' outlined to an SVG path, centred at (cx,cy), cap-height `cap`."""
    f = TTFont(WFONT)
    gname = f.getBestCmap()[ord("W")]
    gs = f.getGlyphSet()
    pen = SVGPathPen(gs)
    gs[gname].draw(pen)
    d = pen.getCommands()
    g = f["glyf"][gname]
    gxc, gyc, gh = (g.xMin + g.xMax) / 2, (g.yMin + g.yMax) / 2, g.yMax - g.yMin
    s = cap / gh
    return (f'<g fill="{INK}" transform="translate({cx},{cy}) '
            f'scale({s:.5f},{-s:.5f}) translate({-gxc:.2f},{-gyc:.2f})">'
            f'<path d="{d}"/></g>')


# candidate D geometry: back hexes, white gap ring, gold hero, outlined W
svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
       + hexpoly(34, 37, 20, GREEN)
       + hexpoly(66, 37, 20, TEAL)
       + hexpoly(50, 56, 31.5, "#fff", edge=False)
       + hexpoly(50, 56, 29, GOLD)
       + w_path(50, 56.5, 24)
       + "</svg>")
(WEB / "favicon.svg").write_text(svg)


def rasterize(px):
    """favicon.svg -> transparent RGBA PIL image at px*px via rsvg-convert."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        subprocess.run(["rsvg-convert", "-w", str(px), "-h", str(px),
                        str(WEB / "favicon.svg"), "-o", tmp.name], check=True)
        return Image.open(tmp.name).convert("RGBA")


# .ico — multi-res from a clean 256 render
rasterize(256).save(WEB / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])

# opaque PNGs: autocrop the cluster, then centre on white at a chosen fill ratio
big = rasterize(512)
icon = big.crop(big.getbbox())


def compose(px, fill):
    c = Image.new("RGBA", (px, px), (255, 255, 255, 255))
    sc = fill * px / max(icon.size)
    r = icon.resize((round(icon.width * sc), round(icon.height * sc)),
                    Image.LANCZOS)
    c.alpha_composite(r, ((px - r.width) // 2, (px - r.height) // 2))
    return c.convert("RGB")


compose(180, 0.84).save(WEB / "apple-touch-icon.png")   # iOS
compose(512, 0.66).save(WEB / "maskable-512.png")        # Android safe zone

manifest = {
    "name": "Deutscher Wohnortatlas",
    "short_name": "Wohnortatlas",
    "icons": [
        {"src": "favicon.svg", "type": "image/svg+xml", "sizes": "any"},
        {"src": "apple-touch-icon.png", "type": "image/png", "sizes": "180x180"},
        {"src": "maskable-512.png", "type": "image/png", "sizes": "512x512",
         "purpose": "maskable"},
    ],
    "theme_color": "#f4f3f1",
    "background_color": "#f4f3f1",
    "display": "standalone",
    "start_url": "/",
}
(WEB / "site.webmanifest").write_text(json.dumps(manifest, indent=2,
                                                 ensure_ascii=False))

for name in ("favicon.svg", "favicon.ico", "apple-touch-icon.png",
             "maskable-512.png", "site.webmanifest"):
    print(f"  wrote web/{name}")
