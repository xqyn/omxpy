# project omx: local Enrichr library access
# XQ - LUMC
# 2026-11-10
# Libraries live in ref/enrich_db/<organism>/<name>.json as {term: [genes]},
# downloaded with ref/enrich_db/db_download_gp.py. The organism folder comes
# from the name suffix: GO_Biological_Process_2023_Human -> human/
#
# Functions:
#   get_db_dir      - locate enrich_db
#   load_library    - read one or more libraries
#   list_enrich_db  - list / check what is on disk

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd

_ENV_DB_DIR = "OMXPY_ENRICH_DB"  # optional override of the enrich_db location
_SUFFIX_RE = re.compile(r"_(Human|Mouse|Yeast|Fly|Fish|Worm)$")


# --- paths
def get_db_dir(db_dir: str | Path | None = None) -> Path:
    """Locate enrich_db: argument, then $OMXPY_ENRICH_DB, then <repo>/ref/enrich_db."""
    if db_dir is not None:
        return Path(db_dir).expanduser()
    env = os.environ.get(_ENV_DB_DIR)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[3] / "ref" / "enrich_db"


def _db_path(name: str, db_dir: str | Path | None = None) -> Path:
    m = _SUFFIX_RE.search(name)
    if not m:
        raise ValueError(f"'{name}' must end with an organism suffix (_Human, _Mouse, ...).")
    return get_db_dir(db_dir) / m.group(1).lower() / f"{name}.json"


def _download_cmd(name: str) -> str:
    org = _SUFFIX_RE.search(name).group(1).lower()
    # GO libraries are named without the organism on Enrichr; KEGG_*_Human is the full name
    base = _SUFFIX_RE.sub("", name) if name.startswith("GO_") else name
    cmd = f"python ref/enrich_db/db_download_gp.py --database {base} --organism {org}"
    return cmd + (f" --name {name}" if base != name else "")


# --- load
@lru_cache(maxsize=32)
def _read_json(path: str, mtime: float) -> dict[str, list[str]]:
    # mtime is part of the cache key: a re-downloaded file is re-read
    with open(path) as fh:
        return json.load(fh)


def load_library(
    db: str | Iterable[str],
    db_dir: str | Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Read libraries from enrich_db.

    Args:
        db (str | Iterable[str]): Full library name(s) incl. organism suffix,
            e.g. "GO_Biological_Process_2023_Human".
        db_dir (str | Path | None): Override the enrich_db location.

    Returns:
        dict: library name -> {term: [gene symbols]}.

    Raises:
        FileNotFoundError: If any library is missing (all are listed, with
            their download command).
    """
    names = [db] if isinstance(db, str) else list(db)
    paths = {n: _db_path(n, db_dir) for n in names}

    missing = [n for n, p in paths.items() if not p.is_file()]
    if missing:
        lines = [f"{len(missing)} librar{'y' if len(missing) == 1 else 'ies'} not found in {get_db_dir(db_dir)}:"]
        lines += [f"  {n}\n    {_download_cmd(n)}" for n in missing]
        raise FileNotFoundError("\n".join(lines))

    return {n: _read_json(str(p), p.stat().st_mtime) for n, p in paths.items()}


# --- list / check
def list_enrich_db(
    db: str | Iterable[str] | None = None,
    db_dir: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """List the libraries on disk, or check that specific ones are available.

    Args:
        db (str | Iterable[str] | None): None lists everything in enrich_db.
            Otherwise each full name is checked and the download command is
            printed for the missing ones.
        db_dir (str | Path | None): Override the enrich_db location.
        verbose (bool): Print the table.

    Returns:
        pd.DataFrame: Listing: organism, name, size_kb. Check: name, available.

    Example:
        >>> list_enrich_db()
        >>> list_enrich_db(["GO_Biological_Process_2023_Human", "KEGG_2026_Human"])
    """
    if db is None:
        root = get_db_dir(db_dir)
        rows = [
            {"organism": p.parent.name, "name": p.stem, "size_kb": round(p.stat().st_size / 1e3, 1)}
            for p in sorted(root.glob("*/*.json"))
        ]
        out = pd.DataFrame(rows, columns=["organism", "name", "size_kb"])
        if verbose:
            print(f"{len(out)} libraries in {root}")
            if not out.empty:
                print(out.to_string(index=False))
        return out

    names = [db] if isinstance(db, str) else list(db)
    out = pd.DataFrame({
        "name": names,
        "available": [_db_path(n, db_dir).is_file() for n in names],
    })
    if verbose:
        print(out.to_string(index=False))
        for n in out.loc[~out["available"], "name"]:
            print(f"  missing: {n}\n    {_download_cmd(n)}")
    return out