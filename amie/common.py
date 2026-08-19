"""Shared config loading and HTTP helpers."""
import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DATA_DIR = ROOT / CONFIG["paths"]["data_dir"]
DATA_DIR.mkdir(exist_ok=True)

_session = requests.Session()
_session.headers.update({"User-Agent": "amie-research/0.1"})


def get_json(url: str, params: dict | None = None):
    """GET with retry/backoff. Returns parsed JSON or raises after retries."""
    cfg = CONFIG["ingest"]
    last_err = None
    for attempt in range(cfg["retry_attempts"]):
        try:
            r = _session.get(url, params=params, timeout=cfg["request_timeout_s"])
            if r.status_code == 429:
                time.sleep(cfg["retry_backoff_s"] * (attempt + 1) * 2)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(cfg["retry_backoff_s"] * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")
