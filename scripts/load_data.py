#!/usr/bin/env python3
"""Download and prepare the BIRD dev set for this assignment.

Produces:
- data/bird/<db_id>.sqlite          - sqlite DB per database, surfaced at the top
- data/bird/dev_databases/...       - raw extracted contents
- evals/eval_set.jsonl              - curated eval questions
- load_test/perf_pool.jsonl         - questions for the load test
"""

import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "bird"
EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
PERF_FILE = ROOT / "load_test" / "perf_pool.jsonl"

BIRD_DEV_URL = os.environ.get(
    "BIRD_DEV_URL",
    "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
)

# Empty = use every DB present in BIRD dev. Restrict here only if you
# specifically want a smaller corpus.
SCOPED_DBS: list[str] = []

N_EVAL = 30
N_PERF = 1500


def download_and_extract() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "dev.zip"
    if not zip_path.exists():
        print(f"Downloading {BIRD_DEV_URL} ...")
        urllib.request.urlretrieve(BIRD_DEV_URL, zip_path)
    if not any(DATA_DIR.rglob("dev.json")):
        print(f"Extracting outer archive into {DATA_DIR} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)

    # BIRD's outer zip contains an inner dev_databases.zip with the actual
    # *.sqlite files. Extract it if present and not already unpacked.
    inner_zip = next(DATA_DIR.rglob("dev_databases.zip"), None)
    if inner_zip is not None and not any(DATA_DIR.rglob("*.sqlite")):
        print(f"Extracting inner archive {inner_zip} ...")
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(inner_zip.parent)


def build_eval_files() -> None:
    dev_json_path = next(DATA_DIR.rglob("dev.json"), None)
    if dev_json_path is None:
        sys.exit("Could not find dev.json after extraction - check the archive layout.")

    rows = json.loads(dev_json_path.read_text())
    if SCOPED_DBS:
        rows = [r for r in rows if r.get("db_id") in SCOPED_DBS]
    print(f"Loaded {len(rows)} questions across {len({r['db_id'] for r in rows})} DBs.")

    rnd = random.Random(0)  # stable shuffle for reproducibility
    rnd.shuffle(rows)

    eval_rows = rows[:N_EVAL]
    perf_source = rows[N_EVAL:]

    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_FILE.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps({
                "question": r["question"],
                "db_id": r["db_id"],
                "gold_sql": r["SQL"],
            }) + "\n")
    print(f"Wrote {len(eval_rows)} eval questions to {EVAL_FILE}")

    # No cycling - we use only unique rows. If the BIRD dev pool is smaller
    # than N_PERF after eval extraction, we just take what's available.
    PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    perf_rows = perf_source[:N_PERF]
    with PERF_FILE.open("w") as f:
        for r in perf_rows:
            f.write(json.dumps({
                "question": r["question"],
                "db_id": r["db_id"],
            }) + "\n")
    print(f"Wrote {len(perf_rows)} perf questions to {PERF_FILE}")


def consolidate_sqlite() -> None:
    """Surface every db at data/bird/<db_id>.sqlite for easy loading."""
    seen: set[str] = set()
    for found in DATA_DIR.rglob("*.sqlite"):
        db_id = found.stem
        if db_id in seen:
            continue
        seen.add(db_id)
        dest = DATA_DIR / f"{db_id}.sqlite"
        if not dest.exists() or dest.resolve() != found.resolve():
            shutil.copy(found, dest)
    available = sorted(p.stem for p in DATA_DIR.glob("*.sqlite"))
    print(f"Sqlite DBs available: {available}")
    if not available:
        sys.exit("No sqlite DBs found - the inner dev_databases.zip may not have extracted correctly.")


if __name__ == "__main__":
    download_and_extract()
    build_eval_files()
    consolidate_sqlite()
    print("Done.")
