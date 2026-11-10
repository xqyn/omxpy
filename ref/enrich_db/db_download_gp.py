#!/usr/bin/env python
"""Download one Enrichr library with gseapy and save it as JSON.

Run from anywhere; the file is saved next to this script:

    <this folder>/<organism>/<name>.json      e.g. ref/enrich_db/human/GO_Molecular_Function_2026_Human.json

Examples:
    python gp_download_db.py --list
    python gp_download_db.py --database GO_Molecular_Function_2026 --organism human \
        --name GO_Molecular_Function_2026_Human
    python gp_download_db.py --database KEGG_2021_Human --organism human

If --name is omitted it defaults to <database>_<Organism> (no doubled suffix
when the database name already ends in _<Organism>, e.g. KEGG_2021_Human).
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import gseapy as gp

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Download an Enrichr library to ref/enrich_db.")
    ap.add_argument("--database", help="Enrichr library name, e.g. GO_Molecular_Function_2026")
    ap.add_argument("--organism", default="human", help="human (default), mouse, yeast, fly, fish, worm")
    ap.add_argument("--name", help="output file name without extension (default: <database>_<Organism>)")
    ap.add_argument("--list", action="store_true", help="print the available library names and exit")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    ap.add_argument("--retries", type=int, default=3, help="download attempts (default 3)")
    args = ap.parse_args()

    org = args.organism.capitalize()  # gseapy wants 'Human', 'Mouse', ...

    try:
        available = sorted(gp.get_library_name(organism=org))
    except Exception as exc:
        sys.exit(f"ERROR: could not reach Enrichr to list libraries: {exc}")

    if args.list:
        print("\n".join(available))
        return

    if not args.database:
        ap.error("--database is required (or use --list)")

    if args.database not in available:
        close = difflib.get_close_matches(args.database, available, n=5, cutoff=0.5)
        sys.exit(f"ERROR: '{args.database}' is not an Enrichr library for {org}. Closest: {close}")

    default_name = args.database if args.database.endswith(f"_{org}") else f"{args.database}_{org}"
    name = Path(args.name or default_name).name  # no folders in the name
    if name.endswith(".json"):
        name = name[: -len(".json")]

    out_dir = BASE_DIR / args.organism.lower()
    out = out_dir / f"{name}.json"

    if out.exists() and not args.force:
        print(f"Already exists, skipping (use --force to overwrite): {out}")
        return

    lib, last_exc = None, None
    for attempt in range(1, args.retries + 1):
        try:
            lib = gp.get_library(name=args.database, organism=org)
            if not lib:
                raise ValueError("downloaded library is empty")
            break
        except Exception as exc:
            last_exc = exc
            print(f"  attempt {attempt}/{args.retries} failed: {exc}")
            if attempt < args.retries:
                time.sleep(2 * attempt)
    if not lib:
        sys.exit(f"ERROR: could not download {args.database}: {last_exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lib))
    tmp.replace(out)  # atomic: never leaves a half-written file

    print(f"Saved {len(lib)} terms ({out.stat().st_size / 1e3:.0f} KB) -> {out}")


if __name__ == "__main__":
    main()