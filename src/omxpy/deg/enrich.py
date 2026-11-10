# project omx: enrichment analysis (GO, KEGG, any Enrichr library)
# XQ - LUMC
# 2026-05-04  python adaptation from omx_GO.R
# 2026-11-10  generalised to any library; file lookup moved to _libs.py
#
# The functions here work on gene sets as plain dicts {term: [genes]}.
# Only enrich_analysis touches _libs, and only to turn library names into dicts.
#
# Functions:
#   run_enrichment       - local Fisher test, one gene list vs one library
#   combine_enrich_data  - merge up/down results, pick top-n
#   enrich_analysis      - main entry point: all libraries, up + down
#   plot_enrich_terms    - horizontal bar plot (up right / down left)
#   enrich_plot          - plot every table returned by enrich_analysis

from __future__ import annotations

import os
import re
import textwrap
import warnings
from collections.abc import Iterable
from pathlib import Path

import gseapy as gp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from ._libs import load_library

_XAXIS_OPTIONS = ("OddsRatio", "GeneRatio", "CombinedScore")
_SORT_OPTIONS = ("CombinedScore", "p.adjust", "OddsRatio", "GeneRatio")
_GO_ID_RE = re.compile(r"\s*\((GO:\d+)\)\s*$")  # Enrichr GO terms: "name (GO:0001234)"
_DOT_XAXIS = ("FoldEnrichment", "OddsRatio", "GeneRatio", "CombinedScore")

def _as_list(x) -> list[str] | None:
    """Gene collection (list, Index, Series, array...) -> de-duplicated list[str]."""
    if x is None:
        return None
    if isinstance(x, str):
        raise TypeError("Expected a list-like of gene symbols, got a single string.")
    return list(dict.fromkeys(str(g).strip() for g in x if pd.notna(g)))


# --- enrichment
def run_enrichment(
    genes: Iterable[str],
    lib: dict[str, list[str]],
    name: str = "",
    regulation: str = "Upregulated",
    background: Iterable[str] | None = None,
    pvalue_cutoff: float = 0.05,
) -> pd.DataFrame:
    """Run enrichment for one gene list against one library (local Fisher test).

    Args:
        genes: Gene symbols to test.
        lib (dict): Gene sets as {term: [genes]}.
        name (str): Library name, stored in the "Library" column.
        regulation (str): "Upregulated" or "Downregulated".
        background: Gene universe, e.g. ``adata.var_names``. Strongly
            recommended for DE results.
        pvalue_cutoff (float): Adjusted p-value threshold.

    Returns:
        pd.DataFrame: Description, GO_ID, p.adjust, Count, GeneRatio, OddsRatio,
            CombinedScore, Regulation, Library, gene_symbols (+ gseapy
            columns). Empty if nothing passes the cutoff or the test fails.
    """
    genes = _as_list(genes)
    if not genes:
        return pd.DataFrame()

    print(f"  {name} — {regulation} ...")

    try:
        bg = _as_list(background)
        if bg is not None:
            n_in, bg_set = len(genes), set(bg)
            genes = [g for g in genes if g in bg_set]
            print(f"  {len(genes)}/{n_in} query genes found in background ({len(bg)} genes).")
            if not genes:
                print(f"  No {regulation} genes overlap the background.")
                return pd.DataFrame()

        enr = gp.enrich(gene_list=genes, gene_sets=lib, background=bg, outdir=None, verbose=False)
        df = enr.results.copy()
    except Exception as exc:
        warnings.warn(f"Enrichment FAILED for {regulation}/{name}: {exc}")
        return pd.DataFrame()

    df = df[df["Adjusted P-value"] < pvalue_cutoff].copy()
    if df.empty:
        print(f"  No {regulation} terms found in {name}.")
        return pd.DataFrame()

    # --- GeneRatio / Count (gseapy "Overlap" = k/K)
    def _parse_ratio(s):
        try:
            a, b = s.split("/")
            return int(a) / int(b)
        except Exception:
            return np.nan

    df["GeneRatio"] = df["Overlap"].apply(_parse_ratio)
    df["Count"] = df["Overlap"].apply(
        lambda s: int(s.split("/")[0]) if isinstance(s, str) and "/" in s else 0
    )
    
    # --- FoldEnrichment = (k/n) / (K/N)
    k = df["Count"].astype(float)
    K = df["Overlap"].apply(lambda s: int(s.split("/")[1]) if isinstance(s, str) and "/" in s else np.nan)
    n = len(genes)  # query genes, already filtered to the background
    N = len(bg) if bg is not None else len({g for v in lib.values() for g in v})
    df["FoldEnrichment"] = (k / n) / (K / N)

    # --- OddsRatio / CombinedScore
    for src, dst in (("Odds Ratio", "OddsRatio"), ("Combined Score", "CombinedScore")):
        df[dst] = df[src].astype(float).replace([np.inf, -np.inf], np.nan) if src in df else np.nan

    if df[["GeneRatio", "OddsRatio", "CombinedScore"]].isna().all(axis=None):
        warnings.warn(f"No plottable metric found for {regulation}/{name}.")
        return pd.DataFrame()

    df["GO_ID"] = df["Term"].str.extract(_GO_ID_RE.pattern, expand=False)
    df["Regulation"] = regulation
    df["Library"] = name

    df = df.rename(columns={"Term": "Description", "Adjusted P-value": "p.adjust", "Genes": "gene_symbols"})
    return df.reset_index(drop=True)


def combine_enrich_data(
    up: pd.DataFrame | None = None,
    down: pd.DataFrame | None = None,
    name: str = "",
    top: int = 10,
    go_id: bool = True,
    sort_by: str = "CombinedScore",
) -> pd.DataFrame:
    """Merge the top terms of the up- and down-regulated results.

    Args:
        up: run_enrichment() output for UP genes.
        down: run_enrichment() output for DOWN genes.
        name (str): Library name, used in messages.
        top (int): Max terms per direction.
        go_id (bool): Keep the "(GO:xxxxxxx)" suffix in the term label
            (default True). False strips it; the GO_ID column is always kept.
            No effect for libraries without GO IDs (KEGG etc.).
        sort_by (str): "CombinedScore", "p.adjust", "OddsRatio" or
            "GeneRatio". p.adjust ascending, the rest descending. Falls back
            to p.adjust if the column is entirely NaN.

    Returns:
        pd.DataFrame: Combined table, empty if both inputs are empty.
    """
    up = pd.DataFrame() if up is None else up
    down = pd.DataFrame() if down is None else down

    if up.empty and down.empty:
        warnings.warn(f"No terms found in {name} for UP or DOWN.")
        print("  Consider increasing pvalue_cutoff.")
        return pd.DataFrame()

    def _top(df: pd.DataFrame) -> pd.DataFrame:
        key = sort_by
        if df[key].isna().all():
            warnings.warn(f"{name}: '{sort_by}' unavailable, ranking by p.adjust instead.")
            key = "p.adjust"
        return df.sort_values(key, ascending=(key == "p.adjust"), na_position="last").head(top)

    combined = pd.concat([_top(d) for d in (up, down) if not d.empty], ignore_index=True)
    combined = combined.dropna(subset=["Count"])

    if not go_id:
        combined["Description"] = combined["Description"].str.replace(_GO_ID_RE, "", regex=True)

    return combined.reset_index(drop=True)


def enrich_analysis(
    db: str | Iterable[str] | dict[str, dict[str, list[str]]],
    de_up: Iterable[str] | None = None,
    de_down: Iterable[str] | None = None,
    background: Iterable[str] | None = None,
    setting: str = "enrich_analysis",
    pvalue_cutoff: float = 0.05,
    top: int = 10,
    sort_by: str = "CombinedScore",
    go_id: bool = True,
    db_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run enrichment for UP and DOWN genes against one or more libraries.

    Pure computation, no plotting. Pass the result to enrich_plot().

    Args:
        db: Library name(s) from enrich_db, with organism suffix
            ("GO_Biological_Process_2023_Human", "KEGG_2026_Human"), or a
            dict {library name: {term: [genes]}} for custom gene sets.
            See list_enrich_db() for what is on disk.
        de_up: Upregulated gene symbols (list, Index, Series...).
        de_down: Downregulated gene symbols.
        background: Gene universe, e.g. ``adata.var_names``. None uses all
            genes of each library (not recommended for DE).
        setting (str): Label used in logs.
        pvalue_cutoff (float): Adjusted p-value threshold.
        top (int): Top N terms per direction per library.
        sort_by (str): "CombinedScore", "p.adjust", "OddsRatio", "GeneRatio".
        go_id (bool): Keep the GO ID in the term label (default True).
        db_dir (str | Path | None): Override the enrich_db location.

    Returns:
        dict[str, pd.DataFrame]: Library name -> combined table (empty
            DataFrame if no terms were found).

    Raises:
        ValueError: If neither de_up nor de_down is given, or sort_by is invalid.
        FileNotFoundError: If a library is not in enrich_db (all missing ones
            are listed before any test is run).

    Example:
        >>> tables = enrich_analysis(
        ...     db=["GO_Biological_Process_2023_Human", "KEGG_2026_Human"],
        ...     de_up=up, de_down=down, background=adata.var_names,
        ...     setting="cond_A",
        ... )
        >>> figs = enrich_plot(tables, setting="cond_A")
    """
    if de_up is None and de_down is None:
        raise ValueError("Provide at least one of de_up or de_down.")
    if sort_by not in _SORT_OPTIONS:
        raise ValueError(f"sort_by must be one of {_SORT_OPTIONS}.")

    up = _as_list(de_up) or []
    down = _as_list(de_down) or []
    bg = _as_list(background)
    libs = db if isinstance(db, dict) else load_library(db, db_dir)

    tables: dict[str, pd.DataFrame] = {}

    for name, lib in libs.items():
        print(f"\n[ Enrichment ] {setting}")
        print(f"  library      : {name}")
        print(f"  background   : {'custom (' + str(len(bg)) + ' genes)' if bg is not None else 'library default'}")
        print(f"  ▲ UP         : {len(up)} genes")
        print(f"  ▼ DOWN       : {len(down)} genes")
        print(f"  top N        : {top}")
        print(f"  sort_by      : {sort_by}")
        print(f"  pvalue cutoff: {pvalue_cutoff}")
        print("  " + "─" * 44)

        kw = dict(lib=lib, name=name, background=bg, pvalue_cutoff=pvalue_cutoff)
        combined = combine_enrich_data(
            up=run_enrichment(up, regulation="Upregulated", **kw),
            down=run_enrichment(down, regulation="Downregulated", **kw),
            name=name, top=top, go_id=go_id, sort_by=sort_by,
        )

        if combined.empty:
            print(f"  No valid terms for {name}.")
            tables[name] = pd.DataFrame()
            continue

        combined.attrs["pvalue_cutoff"] = pvalue_cutoff
        print(f"  terms found  : UP = {(combined['Regulation'] == 'Upregulated').sum()}"
              f"  |  DOWN = {(combined['Regulation'] == 'Downregulated').sum()}")
        tables[name] = combined
        print(f"  Done — {name}")
        print("  " + "─" * 44)

    return tables


# --- plotting
def plot_enrich_terms(
    enrich_combine: pd.DataFrame,
    setting: str = "enrich_analysis",
    name: str = "",
    xaxis: str = "OddsRatio",
    pvalue_cutoff: float | None = None,
    fig_dir: str | None = None,
    fig_width: float = 10,
    row_height: float = 0.45,
    min_rows: int = 6,
    wrap_width: int = 50,
    fig_format: tuple[str, ...] = ("png", "pdf"),
    show: bool = False,
    cap_quantile: float = 0.75,
    cap_threshold: float = 30.0,
) -> plt.Figure | None:
    """Horizontal diverging bar plot of enrichment results.

    UP terms extend rightward, DOWN terms leftward. Bars are coloured by
    adjusted p-value (light -> dark teal).

    Args:
        enrich_combine (pd.DataFrame): One table from enrich_analysis().
        setting (str): Title and filename prefix.
        name (str): Library name for title and filename. Default: the
            "Library" column of the table.
        xaxis (str): "OddsRatio" (default), "GeneRatio" or "CombinedScore".
        pvalue_cutoff (float | None): Colorbar label. None -> the value used
            in enrich_analysis() (0.05 if absent).
        fig_dir (str | None): Save directory; None skips saving.
        fig_width (float): Figure width in inches.
        row_height (float): Inches per bar. The figure height is computed from
            the number of bars, so bar thickness is the same in every plot.
        min_rows (int): Fewer bars than this are padded with empty space, so
            very small plots keep the same bar thickness and a usable colorbar.
        wrap_width (int): Character width for y-axis label wrapping.
        fig_format (tuple[str, ...]): File format(s), e.g. ("png", "pdf").
        show (bool): False (default) closes the figure after saving, so nothing
            is displayed in a notebook and memory is freed. True keeps it open.
            Has no effect when fig_dir is None (nothing saved, figure stays open).
        cap_quantile (float): Quantile of |xaxis| used as axis cap when one
            term dominates the scale.
        cap_threshold (float): Minimum value the quantile cap must reach
            before capping is applied.

    Returns:
        plt.Figure | None: The figure, or None if nothing is plottable.

    Raises:
        ValueError: If xaxis is invalid or cap_quantile not in (0, 1].
        KeyError: If required columns are missing.
    """
    if xaxis not in _XAXIS_OPTIONS:
        raise ValueError(f"xaxis must be one of {_XAXIS_OPTIONS}, got '{xaxis}'.")
    if not (0 < cap_quantile <= 1):
        raise ValueError(f"cap_quantile must be in (0, 1], got {cap_quantile}.")

    missing = {"Description", "p.adjust", "Count", "Regulation", xaxis} - set(enrich_combine.columns)
    if missing:
        raise KeyError(f"enrich_combine is missing columns: {missing}")

    if pvalue_cutoff is None:
        pvalue_cutoff = enrich_combine.attrs.get("pvalue_cutoff", 0.05)
    if not name and "Library" in enrich_combine.columns and len(enrich_combine):
        name = str(enrich_combine["Library"].iloc[0])

    df = enrich_combine.copy()
    n_nan = int(df[xaxis].isna().sum())
    if n_nan:
        warnings.warn(f"{name}: dropping {n_nan} term(s) with no '{xaxis}' value.")
        df = df.dropna(subset=[xaxis])
    if df.empty:
        warnings.warn(f"{name}: nothing to plot for xaxis='{xaxis}'.")
        return None

    # --- signed value: UP right (+), DOWN left (-)
    sign = np.where(df["Regulation"] == "Downregulated", -1.0, 1.0)
    df["plot_val"] = sign * df[xaxis].abs().to_numpy()
    df["label"] = df["Description"].apply(lambda s: "\n".join(textwrap.wrap(str(s), wrap_width)))

    # first row is drawn at the bottom: DOWN (most negative) bottom, UP top
    df = df.sort_values("plot_val", ascending=True, kind="stable").reset_index(drop=True)

    # --- axis cap (one huge bar should not squash the rest)
    abs_vals = df["plot_val"].abs()
    raw_max = abs_vals.max()
    quantile_cap = abs_vals.quantile(cap_quantile)

    apply_cap = (quantile_cap >= cap_threshold) and (raw_max > quantile_cap)
    axis_cap = quantile_cap if apply_cap else raw_max
    if axis_cap == 0:
        axis_cap = 1  # all-zero guard

    df["plot_val_capped"] = df["plot_val"].clip(lower=-axis_cap, upper=axis_cap)
    df["is_truncated"] = df["plot_val"].abs() > axis_cap

    # --- colour scale: -log10(p.adjust)
    log_p = -np.log10(df["p.adjust"].clip(lower=1e-300))
    vmin, vmax = log_p.min(), log_p.max()
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list("go_cmap", ["#8CD0D5", "#05474A"])
    colors = cmap(norm(log_p.values))

    # --- figure
    n_rows = max(len(df), min_rows)
    fig, ax = plt.subplots(figsize=(fig_width, (n_rows + 1) * row_height + 1.5))

    # integer y positions (not string labels): the same term can be in both
    # UP and DOWN without the two bars collapsing onto one row
    y = np.arange(len(df))
    bars = ax.barh(y, df["plot_val_capped"].to_numpy(), color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_ylim(-1.5, n_rows - 0.5)  # room for the UP/DOWN badges below the first bar

    for bar, (_, row) in zip(bars, df.iterrows()):
        x = bar.get_width()
        ax.annotate(
            str(int(row["Count"])),
            xy=(x, bar.get_y() + bar.get_height() / 2),
            xytext=(4 if x >= 0 else -4, 0), textcoords="offset points",
            va="center", ha="left" if x >= 0 else "right", fontsize=12,
        )
        if row["is_truncated"]:  # bar was clipped: show the real value inside it
            arrow = "\u25B6" if x >= 0 else "\u25C0"
            ax.annotate(
                f"{row[xaxis]:.0f} {arrow}",
                xy=(x, bar.get_y() + bar.get_height() / 2),
                xytext=(-6 if x >= 0 else 6, 0), textcoords="offset points",
                va="center", ha="right" if x >= 0 else "left",
                fontsize=7, color="white", fontweight="bold",
            )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # --- UP / DOWN badges
    badge_y = -0.9
    if (df["plot_val"] > 0).any():
        ax.annotate(
            "UP", xy=(axis_cap * 0.5, badge_y), fontsize=8, color="white",
            fontweight="bold", ha="center", annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="#08519C", ec="none"),
        )
    if (df["plot_val"] < 0).any():
        ax.annotate(
            "DOWN", xy=(-axis_cap * 0.5, badge_y), fontsize=8, color="white",
            fontweight="bold", ha="center", annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F48D79", ec="none"),
        )

    # --- axes
    xlabel = {"OddsRatio": "Odds Ratio", "GeneRatio": "Gene Ratio", "CombinedScore": "Combined Score"}[xaxis]
    ax.set_xlabel(xlabel + (" (capped)" if apply_cap else ""), fontsize=11)
    ax.set_ylabel("Terms", fontsize=11)
    ax.set_title(f"{setting} — {name.replace('_', ' ')}" if name else setting, fontsize=13, fontweight="bold")

    ax.set_xlim(-axis_cap * 1.15, axis_cap * 1.15)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}" if abs(v) >= 10 else f"{abs(v):.2f}")
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f"$-\\log_{{10}}$(adj p-value)\n(cutoff < {pvalue_cutoff})", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: r"$10^{" + f"{-v:.0f}" + r"}$")
    )

    fig.tight_layout()

    # --- save
    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        for ext in fig_format:
            fp = os.path.join(fig_dir, f"{setting}_{name or 'enrich'}_pv{pvalue_cutoff}_{xaxis}_enrich.{ext}")
            fig.savefig(fp, dpi=320, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {fp}")
        if not show:
            plt.close(fig)

    return fig


def plot_enrich(
    tables: dict[str, pd.DataFrame],
    setting: str = "enrich_analysis",
    xaxis: str = "OddsRatio",
    pvalue_cutoff: float | None = None,
    fig_dir: str | None = "figures/",
    fig_width: float = 10,
    row_height: float = 0.35,
    min_rows: int = 6,
    wrap_width: int = 50,
    fig_format: tuple[str, ...] = ("png", "pdf"),
    show: bool = False,
) -> dict[str, plt.Figure | None]:
    """Render plots for every table returned by enrich_analysis().

    Args:
        tables (dict[str, pd.DataFrame]): Library name -> combined table.
        setting (str): Title and filename prefix.
        xaxis (str): "OddsRatio" (default), "GeneRatio" or "CombinedScore".
        pvalue_cutoff (float | None): Colorbar label; None -> the value used
            in enrich_analysis().
        fig_dir (str | None): Output directory; None skips saving.
        fig_width (float): Figure width in inches.
        row_height (float): Inches per bar (same bar thickness in every plot).
        min_rows (int): Minimum number of rows reserved in the plot.
        wrap_width (int): Character width for y-axis label wrapping.
        fig_format (tuple[str, ...]): File format(s), e.g. ("png", "pdf").
        show (bool): False (default) saves and closes each figure without
            displaying it. True also displays it (notebook).

    Returns:
        dict[str, plt.Figure | None]: Library name -> Figure, or None if that
            library had no terms.

    Example:
        >>> figs = enrich_plot(tables, setting="cond_A", xaxis="CombinedScore",
        ...                    fig_dir="figures/alt_view/")  # re-plot, no re-test
    """
    if xaxis not in _XAXIS_OPTIONS:
        raise ValueError(f"xaxis must be one of {_XAXIS_OPTIONS}, got '{xaxis}'.")

    figures: dict[str, plt.Figure | None] = {}
    for name, table in tables.items():
        if table is None or table.empty:
            print(f"  No valid terms for {name}. Skipping plot.")
            figures[name] = None
            continue

        figures[name] = plot_enrich_terms(
            enrich_combine=table, setting=setting, name=name, xaxis=xaxis,
            pvalue_cutoff=pvalue_cutoff, fig_dir=fig_dir,
            fig_width=fig_width, row_height=row_height, min_rows=min_rows, wrap_width=wrap_width, fig_format=fig_format, show=show,
        )
    return figures


def plot_enrich_dot_terms(
    enrich_combine: pd.DataFrame,
    setting: str = "enrich_analysis",
    name: str = "",
    xaxis: str = "FoldEnrichment",
    pvalue_cutoff: float | None = None,
    fig_dir: str | None = None,
    fig_width: float = 9,
    row_height: float = 0.32,
    min_rows: int = 6,
    wrap_width: int = 50,
    fig_format: tuple[str, ...] = ("png", "pdf"),
    show: bool = False,
    size_range: tuple[float, float] = (30, 350),
    up_colors: tuple[str, str] = ("#C6DBEF", "#08519C"),
    down_colors: tuple[str, str] = ("#FCBBA1", "#CB181D"),
) -> plt.Figure | None:
    """Dot plot: x = xaxis, colour = -log10(p.adjust), size = Count.

    UP terms are blue, DOWN terms red; each has its own colorbar. Rows are
    grouped by direction (UP on top), sorted by x within each group.
    """
    if xaxis not in _DOT_XAXIS:
        raise ValueError(f"xaxis must be one of {_DOT_XAXIS}, got '{xaxis}'.")
    missing = {"Description", "p.adjust", "Count", "Regulation", xaxis} - set(enrich_combine.columns)
    if missing:
        raise KeyError(f"enrich_combine is missing columns: {missing}")

    if pvalue_cutoff is None:
        pvalue_cutoff = enrich_combine.attrs.get("pvalue_cutoff", 0.05)
    if not name and "Library" in enrich_combine.columns and len(enrich_combine):
        name = str(enrich_combine["Library"].iloc[0])

    df = enrich_combine.dropna(subset=[xaxis]).copy()
    if df.empty:
        warnings.warn(f"{name}: nothing to plot for xaxis='{xaxis}'.")
        return None

    df["label"] = df["Description"].apply(lambda s: "\n".join(textwrap.wrap(str(s), wrap_width)))
    df["log_p"] = -np.log10(df["p.adjust"].clip(lower=1e-300))

    # DOWN at the bottom, UP on top; within each group the largest x is on top
    df["_dir"] = (df["Regulation"] == "Upregulated").astype(int)
    df = df.sort_values(["_dir", xaxis], ascending=True, kind="stable").reset_index(drop=True)
    y = np.arange(len(df))
    n_rows = max(len(df), min_rows)

    # size: Count -> marker area (shared across UP and DOWN)
    cmin, cmax = df["Count"].min(), df["Count"].max()
    smin, smax = size_range

    def _size(c):
        c = np.asarray(c, dtype=float)
        return np.full_like(c, (smin + smax) / 2) if cmax == cmin else smin + (c - cmin) / (cmax - cmin) * (smax - smin)

    fig, ax = plt.subplots(figsize=(fig_width, (n_rows + 1) * row_height + 1.5))
    ax.set_axisbelow(True)
    ax.grid(color="#E5E5E5", linewidth=0.8)

    groups = {
        "Upregulated": (up_colors, "UP"),
        "Downregulated": (down_colors, "DOWN"),
    }
    cbar_slots = {"Upregulated": 0.60, "Downregulated": 0.28}  # y position (axes fraction)

    for reg, (cols, label) in groups.items():
        sub = df[df["Regulation"] == reg]
        if sub.empty:
            continue
        vmin, vmax = sub["log_p"].min(), sub["log_p"].max()
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mcolors.LinearSegmentedColormap.from_list(f"{reg}_cmap", list(cols))
        ax.scatter(
            sub[xaxis], y[sub.index], s=_size(sub["Count"]), c=sub["log_p"],
            cmap=cmap, norm=norm, edgecolors="none", zorder=3,
        )
        cax = ax.inset_axes([1.04, cbar_slots[reg], 0.03, 0.25])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label(f"{label}  $-\\log_{{10}}$(adj p)", fontsize=8)
        cb.ax.tick_params(labelsize=8)

    # separator between DOWN and UP groups
    n_down = int((df["_dir"] == 0).sum())
    if 0 < n_down < len(df):
        ax.axhline(n_down - 0.5, color="grey", linewidth=0.8, linestyle="--", zorder=1)

    if xaxis == "FoldEnrichment":
        ax.axvline(1, color="black", linewidth=0.8, linestyle=":", zorder=2)

    # size legend (# genes)
    ticks = np.unique(np.round(np.linspace(cmin, cmax, 3)).astype(int))
    handles = [
        ax.scatter([], [], s=float(_size(t)), color="black", label=str(t)) for t in ticks
    ]
    ax.legend(
        handles=handles, title="# genes", loc="upper left", bbox_to_anchor=(1.03, 0.2),
        frameon=False, labelspacing=1.2, fontsize=8, title_fontsize=9,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_ylim(-0.8, n_rows - 0.2)
    xmin, xmax = df[xaxis].min(), df[xaxis].max()
    pad = (xmax - xmin) * 0.08 or 0.1
    ax.set_xlim(xmin - pad, xmax + pad)

    xlabel = {"FoldEnrichment": "Fold Enrichment", "OddsRatio": "Odds Ratio",
              "GeneRatio": "Gene Ratio", "CombinedScore": "Combined Score"}[xaxis]
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f"{setting} — {name.replace('_', ' ')}" if name else setting, fontsize=13, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        for ext in fig_format:
            fp = os.path.join(fig_dir, f"{setting}_{name or 'enrich'}_pv{pvalue_cutoff}_{xaxis}_dot.{ext}")
            fig.savefig(fp, dpi=320, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {fp}")
        if not show:
            plt.close(fig)
    return fig


def plot_enrich_dot(
    tables: dict[str, pd.DataFrame],
    setting: str = "enrich_analysis",
    xaxis: str = "FoldEnrichment",
    fig_dir: str | None = "figures/",
    **kwargs,
) -> dict[str, plt.Figure | None]:
    """Dot plot for every table returned by enrich_analysis()."""
    figures = {}
    for name, table in tables.items():
        if table is None or table.empty:
            print(f"  No valid terms for {name}. Skipping plot.")
            figures[name] = None
            continue
        figures[name] = plot_enrich_dot_terms(
            table, setting=setting, name=name, xaxis=xaxis, fig_dir=fig_dir, **kwargs
        )
    return figures


# --- testing
def plot_pathway_genes(
    de_classified: dict[str, pd.DataFrame],
    genes: Iterable[str],
    title: str = "pathway",
    conds: list[str] | None = None,
    lfc_col: str = "logfoldchanges",
    padj_col: str = "pvals_adj",
    top: int = 30,
    fig_dir: str | None = None,
    fig_format: tuple[str, ...] = ("png",),
    row_height: float = 0.28,
    col_width: float = 1.0,
    size_range: tuple[float, float] = (20, 250),
    cmap: str = "RdBu_r",
    show: bool = False,
) -> plt.Figure | None:
    """Gene x condition dot plot for one pathway / gene set.

    colour = logFC (diverging, centred on 0), size = -log10(padj),
    black edge = classified UP/DOWN in that condition.
    Genes are ordered by number of conditions where DE, then max |logFC|.
    """
    genes = _as_list(genes)
    conds = conds or list(de_classified)

    lfc = pd.DataFrame(np.nan, index=genes, columns=conds)
    sig = pd.DataFrame(np.nan, index=genes, columns=conds)
    is_deg = pd.DataFrame(False, index=genes, columns=conds)

    for c in conds:
        d = de_classified[c].drop_duplicates("names").set_index("names")
        common = d.index.intersection(genes)
        lfc.loc[common, c] = d.loc[common, lfc_col].astype(float)
        sig.loc[common, c] = -np.log10(d.loc[common, padj_col].astype(float).clip(lower=1e-300))
        is_deg.loc[common, c] = d.loc[common, "deg"].isin(["UP", "DOWN"])

    keep = lfc.notna().any(axis=1)            # drop genes absent from every table
    lfc, sig, is_deg = lfc[keep], sig[keep], is_deg[keep]
    if lfc.empty:
        warnings.warn(f"{title}: none of the genes were found in the DE tables.")
        return None

    order = pd.DataFrame({"n": is_deg.sum(axis=1), "m": lfc.abs().max(axis=1)})
    order = order.sort_values(["n", "m"], ascending=False).head(top)
    lfc, sig, is_deg = lfc.loc[order.index], sig.loc[order.index], is_deg.loc[order.index]

    n_g, n_c = lfc.shape
    xs, ys = np.meshgrid(np.arange(n_c), np.arange(n_g))   # row 0 = top gene
    v = lfc.to_numpy().ravel()
    s = sig.to_numpy().ravel()
    deg_flat = is_deg.to_numpy().ravel()
    ok = ~np.isnan(v)

    smin, smax = size_range
    s_max = np.nanmax(s) or 1.0
    sizes = smin + (s / s_max) * (smax - smin)

    lim = np.nanmax(np.abs(v)) or 1.0
    fig, ax = plt.subplots(figsize=(n_c * col_width + 3, n_g * row_height + 2))
    ax.set_axisbelow(True)
    ax.grid(color="#E5E5E5", linewidth=0.8)
    sc = ax.scatter(
        xs.ravel()[ok], ys.ravel()[ok], s=sizes[ok], c=v[ok],
        cmap=cmap, vmin=-lim, vmax=lim,
        edgecolors=np.where(deg_flat[ok], "black", "none"), linewidths=1.0, zorder=3,
    )
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(lfc.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_g))
    ax.set_yticklabels(lfc.index, fontsize=9)
    ax.set_xlim(-0.6, n_c - 0.4)
    ax.set_ylim(n_g - 0.4, -0.6)              # inverted: top-ranked gene at the top
    ax.set_title(title, fontsize=13, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cb = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.02)
    cb.set_label("log fold change", fontsize=9)

    ticks = np.unique(np.round(np.linspace(0, s_max, 4)[1:]).astype(int))
    handles = [ax.scatter([], [], s=smin + (t / s_max) * (smax - smin), color="grey", label=str(t))
               for t in ticks]
    ax.legend(handles=handles, title="-log10(adj p)", loc="upper left",
              bbox_to_anchor=(1.18, 0.35), frameon=False, labelspacing=1.2, fontsize=8, title_fontsize=9)

    fig.tight_layout()
    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        safe = re.sub(r"[^\w\-]+", "_", title)
        for ext in fig_format:
            fp = os.path.join(fig_dir, f"{safe}_genes.{ext}")
            fig.savefig(fp, dpi=320, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {fp}")
        if not show:
            plt.close(fig)
    return fig