"""Cached downloads and polite cached API GETs."""

import hashlib
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .config import CACHE, USER_AGENT

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_call: dict[str, float] = {}


def cached_download(
    url: str, dest: Path, desc: str | None = None, progress: bool = True
) -> Path:
    """Stream url to dest unless it already exists (atomic via .part).

    Resumable: a partial `.part` left by an interrupted run is continued via an
    HTTP Range request, so a dropped connection picks up where it left off
    instead of restarting from zero. Servers that ignore Range (respond 200,
    not 206) cleanly restart the file.

    progress=False suppresses the per-file tqdm bar — set it when fetching many
    small files concurrently (parallel bars would clobber the terminal)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with _session.get(url, stream=True, timeout=120, headers=headers) as r:
        if r.status_code == 416:  # range past EOF -> .part already complete
            part.rename(dest)
            return dest
        if pos and r.status_code != 206:  # server ignored Range -> start over
            pos = 0
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + pos
        with open(part, "ab" if pos else "wb") as f, tqdm(
            total=total, initial=pos, unit="B", unit_scale=True,
            desc=desc or dest.name, disable=not progress,
        ) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest


def cached_post_json(
    url: str,
    body: dict,
    cache_key: str | None = None,
    rate_limit_s: float = 1.0,
    rate_bucket: str = "default",
    verify: bool | str = True,
) -> dict | list:
    """POST JSON with disk cache (by url+body hash) and per-bucket rate limit.

    verify: passed to requests — path to a CA bundle for servers with an
    incomplete certificate chain (e.g. inkar.de omits its intermediate).
    """
    key = cache_key or hashlib.sha256(
        (url + json.dumps(body, sort_keys=True)).encode()
    ).hexdigest()[:24]
    cache_file = CACHE / rate_bucket / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    wait = _last_call.get(rate_bucket, 0) + rate_limit_s - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    r = _session.post(url, json=body, timeout=180, verify=verify)
    _last_call[rate_bucket] = time.monotonic()
    r.raise_for_status()
    data = r.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def cached_get_json(
    url: str,
    params: dict | None = None,
    cache_key: str | None = None,
    rate_limit_s: float = 1.0,
    rate_bucket: str = "default",
) -> dict | list:
    """GET JSON with disk cache (by url+params hash) and per-bucket rate limit."""
    key = cache_key or hashlib.sha256(
        (url + json.dumps(params or {}, sort_keys=True)).encode()
    ).hexdigest()[:24]
    cache_file = CACHE / rate_bucket / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    wait = _last_call.get(rate_bucket, 0) + rate_limit_s - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    r = _session.get(url, params=params, timeout=180)
    _last_call[rate_bucket] = time.monotonic()
    r.raise_for_status()
    data = r.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data
