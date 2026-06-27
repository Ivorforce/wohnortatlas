#!/usr/bin/env bash
# Publish the built static bundle to the DigitalOcean deploy repo.
#
# DO App Platform watches that repo's default branch and redeploys on push. We snapshot
# web/ into a throwaway temp dir, `git init` it fresh, make ONE commit, and force-push.
# So the remote branch is always exactly one commit (~a few MB): old blobs become
# unreachable and GitHub garbage-collects them server-side. Nothing accumulates locally
# (the temp dir is discarded), and history never grows — tune the variables and republish
# as often as you like.
#
# Prereq: web/data.bin + web/reach/ must be current. `make publish` enforces this by
# depending on web/data.bin; if you run this script directly, run `make web` first.
set -euo pipefail

# Override via env if the repo/branch ever moves. SSH (matches the existing origin remote).
REMOTE="${WOHNEN_DEPLOY_REMOTE:-git@github.com:Ivorforce/wohnortatlas-builds.git}"
BRANCH="${WOHNEN_DEPLOY_BRANCH:-main}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ROOT/web"

# The deployable subset: frontend + built data + vendored libs. NOT the *.test.js (dev-only).
# index.html references everything by relative path (incl. vendor/), so the repo root IS the
# site root (App Platform output dir = "/").
FILES=(index.html decode.js method.html impressum.html data.bin
       favicon.svg favicon.ico apple-touch-icon.png maskable-512.png site.webmanifest)

# Optional assets: shipped if present, skipped (with a warning) if not — so publish never
# breaks on a not-yet-created asset. og-image.png is the 1200×630 OpenGraph preview; the
# og:image meta tags are live in index.html and the PNG is checked in under web/, so this
# normally ships it. If the PNG is ever removed, link shares just render image-less.
OPTIONAL_FILES=(og-image.png)

for f in "${FILES[@]}"; do
  [ -e "$WEB/$f" ] || { echo "error: missing $WEB/$f — run 'make web' first" >&2; exit 1; }
done
[ -d "$WEB/reach" ] || { echo "error: missing $WEB/reach — run 'make web' first" >&2; exit 1; }
[ -d "$WEB/vendor" ] || { echo "error: missing $WEB/vendor — vendored libs (maplibre/deck.gl/h3-js)" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for f in "${FILES[@]}"; do cp "$WEB/$f" "$TMP/$f"; done
for f in "${OPTIONAL_FILES[@]}"; do
  if [ -e "$WEB/$f" ]; then cp "$WEB/$f" "$TMP/$f"; else echo "warn: optional $WEB/$f absent — skipping (no link-preview image until created)" >&2; fi
done
cp -R "$WEB/reach" "$TMP/reach"
cp -R "$WEB/vendor" "$TMP/vendor"

# Content-hash the format-coupled assets (data.bin/decode.js/vendor/reach) and rewrite
# their refs in index.html, so returning users can't mix old HTML with new assets after
# a deploy. Operates on the TMP copy only (source keeps plain names for the dev server).
# Fails loudly — a missed rewrite would 404 the live site — so a non-zero exit aborts here.
python3 "$ROOT/scripts/stamp_assets.py" "$TMP"

SRC_REV="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

git -C "$TMP" init -q -b "$BRANCH"
git -C "$TMP" add -A
git -C "$TMP" -c user.name='wohnen-publish' -c user.email='publish@localhost' \
  commit -qm "deploy (src $SRC_REV)"
git -C "$TMP" push -f "$REMOTE" "$BRANCH"

echo "published: data.bin $(du -h "$TMP"/data.*.bin | cut -f1) + $(find "$TMP/reach" -name '*.bin' | wc -l | tr -d ' ') reach chunks → $REMOTE ($BRANCH, src $SRC_REV)"
