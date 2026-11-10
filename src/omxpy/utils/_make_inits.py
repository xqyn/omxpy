# src/dev/make_inits.py
"""Regenerate ``__init__.py`` for every package under ``src/``.

For each package folder, every public top-level function and class in each
non-underscore ``.py`` file is re-exported, so ``from pkg import func`` works
without editing ``__init__.py`` by hand.

Rules:
    - Files starting with "_" (e.g. ``_libs.py``) are skipped: they are
      internal and not exported at package level.
    - Names starting with "_" are skipped.
    - Only top-level definitions count (no nested functions or methods).
    - Folders in SKIP_DIRS (this ``dev`` folder, caches) are ignored.
    - Each ``__init__.py`` is overwritten, so do not put manual code in it.

Usage:
    python src/utils/_make_inits.py             # rewrite all __init__.py files
    python src/utils/_make_inits.py --dry-run   # show what would be written

Example:
    Given ``src/omx/enrich.py`` with ``run_enrichment`` and ``plot_enrich``,
    and ``src/omx/plot.py`` with ``plot_volcano``, the generated
    ``src/omx/__init__.py`` is::

        from .enrich import run_enrichment, plot_enrich
        from .plot import plot_volcano

    After that, ``from omx import plot_enrich`` works.
"""
from __future__ import annotations

import argparse
import ast
import warnings
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).parent.parent
LOG_FILE = Path(__file__).parent / "log" / "make_inits.log"
SKIP_DIRS = {"dev", "__pycache__"}
MAX_LINE = 88  # wrap the import line in parentheses beyond this width


def public_names(py_file: Path) -> list[str]:
    """Return the public top-level function and class names of a module.

    Args:
        py_file (Path): Python source file to parse (it is not imported).

    Returns:
        list[str]: Names in order of definition. Empty if there are none or
            the file has a syntax error (a warning is issued).

    Example:
        >>> public_names(Path("src/omx/enrich.py"))
        ['run_enrichment', 'combine_enrich_data', 'enrich_analysis', ...]
    """
    try:
        tree = ast.parse(py_file.read_text())
    except SyntaxError as exc:
        warnings.warn(f"Skipping {py_file}: syntax error ({exc}).")
        return []

    return [
        node.name
        for node in tree.body  # top level only: no nested helpers or methods
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    ]


def import_line(module: str, names: list[str]) -> str:
    """Build the import statement for one module.

    Uses one line if it fits in MAX_LINE, otherwise a parenthesised block
    with one name per line.

    Args:
        module (str): Module stem, e.g. ``"enrich"``.
        names (list[str]): Names to import from it.

    Returns:
        str: The import statement.

    Example:
        >>> import_line("plot", ["plot_volcano"])
        'from .plot import plot_volcano'
    """
    line = f"from .{module} import {', '.join(names)}"
    if len(line) <= MAX_LINE:
        return line
    body = "".join(f"    {n},\n" for n in names)
    return f"from .{module} import (\n{body})"


def package_exports(pkg_dir: Path) -> list[str]:
    """Collect the import statements for one package folder.

    Args:
        pkg_dir (Path): Package folder, e.g. ``src/omx``.

    Returns:
        list[str]: One import statement per module that has public names.
    """
    exports = []
    for py_file in sorted(pkg_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        names = public_names(py_file)
        if names:
            exports.append(import_line(py_file.stem, names))
    return exports


def make_inits(src: Path = SRC, dry_run: bool = False) -> list[str]:
    """Write ``__init__.py`` for every package under ``src``.

    Args:
        src (Path): Folder that contains the package folders.
        dry_run (bool): If True, print what would be written and change nothing.

    Returns:
        list[str]: Log lines describing what was (or would be) written.

    Example:
        >>> make_inits(dry_run=True)   # preview
        >>> make_inits()               # write
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [f"\n[{timestamp}]" + (" (dry run)" if dry_run else "")]

    for pkg_dir in sorted(src.iterdir()):
        if not pkg_dir.is_dir() or pkg_dir.name in SKIP_DIRS or pkg_dir.name.startswith((".", "_")):
            continue

        exports = package_exports(pkg_dir)
        if not exports:
            continue

        init = pkg_dir / "__init__.py"
        if dry_run:
            print(f"would write {init}")
        else:
            init.write_text("\n".join(exports) + "\n")
            print(f"wrote {init}")

        log_lines.append(f"  {pkg_dir.name}/__init__.py")
        log_lines.extend(f"    {line}" for export in exports for line in export.splitlines())

    return log_lines


def write_log(log_lines: list[str], log_file: Path = LOG_FILE) -> None:
    """Append log lines to the log file (the folder is created if needed)."""
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, "a") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"logged to {log_file}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Regenerate __init__.py for all packages in src/.")
    parser.add_argument("--dry-run", action="store_true", help="show what would be written, change nothing")
    args = parser.parse_args()

    log_lines = make_inits(dry_run=args.dry_run)
    if not args.dry_run:
        write_log(log_lines)


if __name__ == "__main__":
    main()