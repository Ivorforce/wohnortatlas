#!/usr/bin/env python3
"""Build web/og-image.png — the 1200x630 OpenGraph share preview.

Composites a label-free screenshot of the score map (assets/og-source.png) into
a 1200x630 frame, darkens the lower band with a gradient scrim, and sets the
brand lockup ("Deutscher Wohnortatlas" + tagline) bottom-left.

This is a branding asset, NOT a pipeline stage — run it by hand when the source
screenshot or the wordmark changes:

    python3 scripts/make_og_image.py        # uses any python with Pillow

After (re)generating, publish.sh already ships og-image.png (OPTIONAL_FILES).
Capture the source with the sidebar + all place/candidate labels hidden so the
preview reads as the national product, not one named city.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "og-source.png"
OUT = ROOT / "web" / "og-image.png"

W, H = 1200, 630          # OpenGraph canonical size
S = 2                     # supersample for crisp text, downscaled at the end
W2, H2 = W * S, H * S

# Which horizontal band of the (width-fitted) source to keep. 0 = top, 1 = bottom.
# Biased low: the colourful radial town + its gradient sit in the lower-middle,
# the calm grey rural top is partly kept as breathing room behind the lockup.
BAND = 0.62

ACCENT = (192, 95, 46)    # site accent orange (#c05f2e)

# Font candidates: classic Helvetica (closest to the site's -apple-system stack),
# then Arial as a guaranteed-present fallback. (path, ttc_index).
BOLD = [("/System/Library/Fonts/Helvetica.ttc", 1),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0)]
REG = [("/System/Library/Fonts/Helvetica.ttc", 0),
       ("/System/Library/Fonts/Supplemental/Arial.ttf", 0)]


def load_font(cands, size):
    for path, idx in cands:
        try:
            return ImageFont.truetype(path, size=size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# --- map: width-fit the source, then crop the chosen vertical band ---------
src = Image.open(SRC).convert("RGB")
fit_h = round(src.height * W2 / src.width)
src = src.resize((W2, fit_h), Image.LANCZOS)
top = round((fit_h - H2) * BAND)
base = src.crop((0, top, W2, top + H2)).convert("RGBA")

# --- scrim: black, transparent up top, ramping dark toward the bottom ------
grad = Image.new("L", (1, H2), 0)
start = int(H2 * 0.40)            # gradient begins ~40% down
amax = 175                        # peak alpha at the very bottom (~0.69)
for y in range(start, H2):
    grad.putpixel((0, y), int(smoothstep((y - start) / (H2 - start)) * amax))
grad = grad.resize((W2, H2))
scrim = Image.new("RGBA", (W2, H2), (8, 12, 18, 0))
scrim.putalpha(grad)
base = Image.alpha_composite(base, scrim)

# --- lockup: wordmark, accent rule, tagline (bottom-left) ------------------
mark_f = load_font(BOLD, 104)
tag_f = load_font(REG, 60)
x = 60 * S
wordmark, tagline = "Deutscher Wohnortatlas", "Finde deine Gegend in Deutschland"

txt = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
d = ImageDraw.Draw(txt)
tag_box = d.textbbox((0, 0), tagline, font=tag_f)
mark_box = d.textbbox((0, 0), wordmark, font=mark_f)
tag_h, mark_h = tag_box[3] - tag_box[1], mark_box[3] - mark_box[1]

tag_y = H2 - 56 * S - tag_h           # tagline baseline near the bottom margin
rule_h, gap = 8 * S, 18 * S
rule_y = tag_y - tag_box[1] - gap - rule_h
mark_y = rule_y - gap - mark_h

d.text((x, mark_y - mark_box[1]), wordmark, font=mark_f, fill=(255, 255, 255, 255))
d.rectangle([x, rule_y, x + 112 * S, rule_y + rule_h], fill=ACCENT + (255,))
d.text((x, tag_y), tagline, font=tag_f, fill=(255, 255, 255, 235))

# soft shadow so text holds over any hex colour, then the crisp text on top
sd = txt.split()[3].point(lambda a: int(a * 0.55))
black = Image.new("RGBA", (W2, H2), (0, 0, 0, 255))
black.putalpha(sd.filter(ImageFilter.GaussianBlur(3 * S)))
base = Image.alpha_composite(base, black)
base = Image.alpha_composite(base, txt)

# --- downscale to final size and save --------------------------------------
out = base.convert("RGB").resize((W, H), Image.LANCZOS)
out.save(OUT, "PNG", optimize=True)
print(f"wrote {OUT.relative_to(ROOT)}  {out.size[0]}x{out.size[1]}  "
      f"font={mark_f.getname()}")
