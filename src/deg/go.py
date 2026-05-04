# project omx: GO analysis
# XQ - LUMC
# 2026-05-04
# gene ontology analysis: 
# python adaptation from omx_GO.R
#
# Functions:
#   go_enrichment           - run enrichr for one ontology + direction
#   combine_go_data         - merge up/down results, pick top-n
#   plot_go_terms           - horizontal bar plot (up right / down left)
#   go_analysis             - main entry point: runs all 3 ontologies

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import gseapy as gp


# ── ontology mapping ──────────────────────────────────────────────────────────
ONTOLOGY = {
    "MF": "Molecular Function",
    "CC": "Cellular Component",
    "BP": "Biological Process",
}

_GENE_SETS = {
    "MF": "GO_Molecular_Function_2023",
    "CC": "GO_Cellular_Component_2023",
    "BP": "GO_Biological_Process_2023",
}

# valid choices for the x-axis metric
_XAXIS_OPTIONS = ("OddsRatio", "GeneRatio", "CombinedScore")

# # ── module info ───────────────────────────────────────────────────────────────
# print(
#     "\n[ omx GO analysis ]"
#     f"\n  ontologies  : { {k: v for k, v in ONTOLOGY.items()} }"
#     f"\n  gene sets   : { {k: v for k, v in _GENE_SETS.items()} }"
#     f"\n  xaxis opts  : {_XAXIS_OPTIONS}"
#     "\n  usage       : go_analysis(de_up=[], de_down=[], organism='human'|'mouse')"
#     "\n"
# )


# ── go_enrichment ─────────────────────────────────────────────────────────────
def go_enrichment(
    genes: list[str],
    regulation: str = "Upregulated",
    ont: str = "MF",
    organism: str = "human",
    background: list[str] | None = None,
    pvalue_cutoff: float = 0.05,
) -> pd.DataFrame:
    """Run GO enrichment for one gene list and one ontology via Enrichr.

    Detects available score columns automatically:
      - ``Overlap``       present → computes GeneRatio  (background=None)
      - ``Odds Ratio``    present → uses OddsRatio      (background supplied)
    Both are stored; the caller decides which to plot.

    Args:
        genes (list[str]): Gene symbols to test.
        regulation (str): "Upregulated" or "Downregulated".
        ont (str): One of "MF", "CC", "BP".
        organism (str): "human" or "mouse".
        background (list[str] | None): Background gene universe.
        pvalue_cutoff (float): Adjusted p-value threshold.

    Returns:
        pd.DataFrame: Columns guaranteed present on success:
            Description, p.adjust, Count, GeneRatio, OddsRatio,
            CombinedScore, Regulation, gene_symbols.
        Empty DataFrame if nothing passes the cutoff or enrichr fails.
    """
    if not genes:
        return pd.DataFrame()

    gene_set = _GENE_SETS[ont]
    print(f"  GO {regulation} — {ONTOLOGY[ont]} ...")

    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_set,
            background=background,
            organism=organism,
            outdir=None,
            verbose=False,
        )
        df = enr.results.copy()
    except Exception as exc:
        warnings.warn(f"enrichr failed for {regulation}/{ont}: {exc}")
        return pd.DataFrame()

    df = df[df["Adjusted P-value"] < pvalue_cutoff].copy()

    if df.empty:
        print(f"  No {regulation} GO terms found for {ONTOLOGY[ont]}.")
        return pd.DataFrame()

    # ── GeneRatio ─────────────────────────────────────────────────────────────
    if "Overlap" in df.columns:
        def _parse_ratio(s):
            try:
                a, b = s.split("/")
                return int(a) / int(b)
            except Exception:
                return np.nan

        df["GeneRatio"] = df["Overlap"].apply(_parse_ratio)
        df["Count"]     = df["Overlap"].apply(
            lambda s: int(s.split("/")[0]) if isinstance(s, str) and "/" in s else 0
        )
    else:
        df["GeneRatio"] = np.nan
        df["Count"]     = df["Genes"].apply(
            lambda s: len(s.split(";")) if isinstance(s, str) else 0
        )

    # ── OddsRatio ─────────────────────────────────────────────────────────────
    if "Odds Ratio" in df.columns:
        df["OddsRatio"] = df["Odds Ratio"].astype(float)
    else:
        df["OddsRatio"] = np.nan

    # ── CombinedScore ─────────────────────────────────────────────────────────
    if "Combined Score" in df.columns:
        df["CombinedScore"] = df["Combined Score"].astype(float)
    else:
        df["CombinedScore"] = np.nan

    # guard: need at least one plottable metric
    if df[["GeneRatio", "OddsRatio", "CombinedScore"]].isna().all(axis=None):
        warnings.warn(f"No plottable metric found for {regulation}/{ont}.")
        return pd.DataFrame()

    df["Regulation"] = regulation

    df = df.rename(columns={
        "Term":             "Description",
        "Adjusted P-value": "p.adjust",
        "Genes":            "gene_symbols",
    })

    return df.reset_index(drop=True)


# ── combine_go_data ───────────────────────────────────────────────────────────
def combine_go_data(
    go_up: pd.DataFrame | None = None,
    go_down: pd.DataFrame | None = None,
    ont: str = "MF",
    top: int = 10,
    go_id: bool = False,
    sort_by: str = "CombinedScore",
) -> pd.DataFrame:
    """Merge top GO results from up- and down-regulated sets.

    Args:
        go_up (pd.DataFrame | None): Output of go_enrichment() for UP genes.
        go_down (pd.DataFrame | None): Output of go_enrichment() for DOWN genes.
        ont (str): Ontology key used in warnings.
        top (int): Max terms per direction.
        go_id (bool): Append GO ID to description when available.
        sort_by (str): Column to rank terms before taking top-N.
            One of "CombinedScore" (default), "p.adjust", "OddsRatio",
            "GeneRatio". For p.adjust ranking is ascending; all others
            descending.

    Returns:
        pd.DataFrame: Combined dataframe, empty if both inputs are empty.
    """
    if go_up is None:
        go_up = pd.DataFrame()
    if go_down is None:
        go_down = pd.DataFrame()

    if go_up.empty and go_down.empty:
        warnings.warn(f"No GO terms found in {ONTOLOGY[ont]} for UP or DOWN.")
        print("  Consider increasing pvalue_cutoff.")
        return pd.DataFrame()

    ascending = sort_by == "p.adjust"

    def _top(df):
        if df.empty or sort_by not in df.columns:
            return df.head(top)
        return (
            df.sort_values(sort_by, ascending=ascending)
              .head(top)
        )

    parts = []
    if not go_up.empty:
        parts.append(_top(go_up))
    if not go_down.empty:
        parts.append(_top(go_down))

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=["Count"])

    if go_id and "ID" in combined.columns:
        combined["Description"] = (
            combined["Description"] + " (" + combined["ID"] + ")"
        )

    return combined.reset_index(drop=True)


# ── plot_go_terms ─────────────────────────────────────────────────────────────
def plot_go_terms(
    go_combine: pd.DataFrame,
    setting: str = "go_analysis",
    ont: str = "MF",
    xaxis: str = "OddsRatio",
    pvalue_cutoff: float = 0.05,
    figure_dir: str | None = None,
    figsize: tuple[int, int] = (10, 7),
    wrap_width: int = 50,
    fig_format: tuple[str, ...] = ("png", "pdf"),
) -> plt.Figure:
    """Horizontal diverging bar plot of GO enrichment results.

    UP terms extend rightward, DOWN terms extend leftward.
    Bars are coloured by adjusted p-value (light → dark teal).

    Args:
        go_combine (pd.DataFrame): Output of combine_go_data().
        setting (str): Title and filename prefix.
        ont (str): Ontology key — one of "MF", "CC", "BP".
        xaxis (str): Metric for bar length. One of:
            "OddsRatio" (default) — fold-enrichment style, unbounded.
            "GeneRatio"           — fraction of query genes in term, 0–1.
            "CombinedScore"       — Enrichr combined score (ln p × z-score).
        pvalue_cutoff (float): Shown in colorbar label.
        figure_dir (str | None): Save directory; None skips saving.
        figsize (tuple[int, int]): Figure size as (width, height) in inches.
            Defaults to (10, 7).
        wrap_width (int): Character width for y-axis label wrapping.
        fig_format (tuple[str, ...]): File format(s) to save, e.g. ("png", "pdf").

    Returns:
        plt.Figure: The rendered matplotlib figure.

    Raises:
        ValueError: If xaxis is not one of the accepted options.
        KeyError: If required columns are missing from go_combine.
    """
    if xaxis not in _XAXIS_OPTIONS:
        raise ValueError(
            f"xaxis must be one of {_XAXIS_OPTIONS}, got '{xaxis}'."
        )

    required = {"Description", "p.adjust", "Count", "Regulation", xaxis}
    missing  = required - set(go_combine.columns)
    if missing:
        raise KeyError(f"go_combine is missing columns: {missing}")

    df = go_combine.copy()

    # ── build signed plot value ───────────────────────────────────────────────
    # UP = positive (bars go right), DOWN = negative (bars go left)
    df["plot_val"] = df.apply(
        lambda r: -abs(r[xaxis]) if r["Regulation"] == "Downregulated"
                  else abs(r[xaxis]),
        axis=1,
    )

    df["label"] = df["Description"].apply(
        lambda s: "\n".join(textwrap.wrap(str(s), wrap_width))
    )

    # sort: DOWN terms at top (most negative first), UP terms at bottom
    df = df.sort_values("plot_val", ascending=True).reset_index(drop=True)

    # ── colour scale: -log10(p.adjust) ───────────────────────────────────────
    log_p = -np.log10(df["p.adjust"].clip(lower=1e-300))
    vmin, vmax = log_p.min(), log_p.max()
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap   = mcolors.LinearSegmentedColormap.from_list(
        "go_cmap", ["#8CD0D5", "#05474A"]
    )
    colors = cmap(norm(log_p.values))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(df["label"], df["plot_val"], color=colors, edgecolor="none")

    # gene count labels at bar tips
    x_range = df["plot_val"].abs().max()
    offset  = x_range * 0.01          # 1 % of axis range — scales with data

    for bar, (_, row) in zip(bars, df.iterrows()):
        x  = bar.get_width()
        ha = "left" if x >= 0 else "right"
        ax.text(
            x + (offset if x >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            str(int(row["Count"])),
            va="center", ha=ha, fontsize=8,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # UP / DOWN badges (only when both directions present)
    has_up   = (df["plot_val"] > 0).any()
    has_down = (df["plot_val"] < 0).any()
    if has_up and has_down:
        xmax    = df["plot_val"].max() * 1.25
        xmin    = df["plot_val"].min() * 1.25
        badge_y = -0.9
        ax.annotate(
            "UP", xy=(xmax * 0.5, badge_y), fontsize=8, color="white",
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#08519C", ec="none"),
        )
        ax.annotate(
            "DOWN", xy=(xmin * 0.5, badge_y), fontsize=8, color="white",
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F48D79", ec="none"),
        )

    # ── axis labels & formatting ──────────────────────────────────────────────
    xlabel_map = {
        "OddsRatio":     "Odds Ratio",
        "GeneRatio":     "Gene Ratio",
        "CombinedScore": "Combined Score",
    }
    ax.set_xlabel(xlabel_map[xaxis], fontsize=11)
    ax.set_ylabel("GO Terms", fontsize=11)
    ax.set_title(
        f"{setting} — {ONTOLOGY[ont]}", fontsize=13, fontweight="bold"
    )

    # x-axis ticks: absolute values, fewer decimals for large numbers
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{abs(x):.0f}" if abs(x) >= 10 else f"{abs(x):.2f}"
        )
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(
        f"$-\\log_{{10}}$(adj p-value)\n(cutoff < {pvalue_cutoff})", fontsize=8
    )
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: r"$10^{" + f"{-x:.0f}" + r"}$"
        )
    )

    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────────
    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
        for ext in fig_format:
            fp = os.path.join(
                figure_dir,
                f"{setting}_{ont}_pv{pvalue_cutoff}_{xaxis}_go_term.{ext}",
            )
            fig.savefig(fp, dpi=320, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {fp}")

    return fig


# ── go_analysis ───────────────────────────────────────────────────────────────
def go_analysis(
    setting: str = "go_analysis",
    ont: str | tuple[str, ...] = ("MF", "CC", "BP"),
    de_up: list[str] | None = None,
    de_down: list[str] | None = None,
    organism: str = "human",
    background: list[str] | None = None,
    pvalue_cutoff: float = 0.05,
    top: int = 10,
    sort_by: str = "CombinedScore",
    xaxis: str = "OddsRatio",
    go_id: bool = False,
    figure_dir: str | None = "figures/",
    figsize: tuple[int, int] = (10, 7),
    wrap_width: int = 50,
    fig_format: tuple[str, ...] = ("png", "pdf"),
) -> dict[str, plt.Figure | None]:
    """Run GO enrichment for UP and DOWN genes across one or more ontologies.

    Args:
        setting (str): Label used in titles and filenames.
        ont (str | tuple[str, ...]): Ontology key(s) from {"MF", "CC", "BP"}.
            Defaults to all three.
        de_up (list[str] | None): Upregulated gene symbols.
        de_down (list[str] | None): Downregulated gene symbols.
        organism (str): "human" or "mouse".
        background (list[str] | None): Background gene universe.
            When None, Enrichr uses its default background and returns
            Overlap/GeneRatio. When supplied, Enrichr returns Odds Ratio.
        pvalue_cutoff (float): Adjusted p-value threshold.
        top (int): Top N terms per direction per plot.
        sort_by (str): Ranking metric used to select top-N terms.
            One of "CombinedScore" (default), "p.adjust", "OddsRatio",
            "GeneRatio". p.adjust is ranked ascending; all others descending.
        xaxis (str): Metric to plot on the x-axis.
            One of "OddsRatio" (default), "GeneRatio", "CombinedScore".
        go_id (bool): Append GO ID to term description when available.
        figure_dir (str | None): Output directory for saved figures.
            None skips saving. Defaults to "figures/".
        figsize (tuple[int, int]): Figure size as (width, height) in inches.
            Defaults to (10, 7).
        wrap_width (int): Character width for y-axis label wrapping.
        fig_format (tuple[str, ...]): File format(s) to save,
            e.g. ("png", "pdf").

    Returns:
        dict[str, plt.Figure | None]: Ontology key mapped to Figure,
            or None if no terms were found for that ontology.

    Raises:
        ValueError: If neither de_up nor de_down is provided,
                    or if xaxis is not a valid option.
        TypeError: If de_up or de_down are not lists.

    Example:
        >>> figs = go_analysis(
        ...     setting="Fibroblast_vs_Epithelial",
        ...     de_up=up_genes,
        ...     de_down=down_genes,
        ...     organism="mouse",
        ...     background=all_genes,
        ...     xaxis="OddsRatio",
        ...     sort_by="CombinedScore",
        ...     figsize=(12, 8),
        ...     figure_dir="output/go/",
        ... )
    """
    if de_up is None and de_down is None:
        raise ValueError("Provide at least one of de_up or de_down.")
    if de_up is not None and not isinstance(de_up, (list, tuple)):
        raise TypeError("de_up must be a list of gene symbols.")
    if de_down is not None and not isinstance(de_down, (list, tuple)):
        raise TypeError("de_down must be a list of gene symbols.")
    if xaxis not in _XAXIS_OPTIONS:
        raise ValueError(f"xaxis must be one of {_XAXIS_OPTIONS}.")

    if isinstance(ont, str):
        ont = (ont,)

    figures: dict[str, plt.Figure | None] = {}

    for o in ont:
        if o not in ONTOLOGY:
            warnings.warn(f"Unknown ontology '{o}', skipping. Use MF, CC, or BP.")
            continue

        print(f"\n[ GO Analysis ] {setting} — {ONTOLOGY[o]}")
        print(f"  gene set    : {_GENE_SETS[o]}")
        print(f"  organism    : {organism}")
        print(f"  background  : {'custom (' + str(len(background)) + ' genes)' if background is not None else 'Enrichr default'}")
        print(f"  UP          : {len(de_up)   if de_up   else 0} genes")
        print(f"  DOWN        : {len(de_down) if de_down else 0} genes")
        print(f"  top N       : {top}  |  sort_by : {sort_by}")
        print(f"  xaxis       : {xaxis}  |  pvalue cutoff : {pvalue_cutoff}")
        print(f"  figsize     : {figsize}  |  fig_format : {fig_format}")
        print(f"  figure_dir  : {figure_dir if figure_dir else 'not saving'}")
        print("  " + "─" * 40)

        go_up = go_enrichment(
            genes=list(de_up) if de_up else [],
            regulation="Upregulated",
            ont=o,
            organism=organism,
            background=background,
            pvalue_cutoff=pvalue_cutoff,
        )
        go_down = go_enrichment(
            genes=list(de_down) if de_down else [],
            regulation="Downregulated",
            ont=o,
            organism=organism,
            background=background,
            pvalue_cutoff=pvalue_cutoff,
        )

        go_combine = combine_go_data(
            go_up=go_up,
            go_down=go_down,
            ont=o,
            top=top,
            go_id=go_id,
            sort_by=sort_by,
        )

        if go_combine.empty:
            print(f"  No valid GO terms for {ONTOLOGY[o]}. Skipping plot.")
            figures[o] = None
            continue

        fig = plot_go_terms(
            go_combine=go_combine,
            setting=setting,
            ont=o,
            xaxis=xaxis,
            pvalue_cutoff=pvalue_cutoff,
            figure_dir=figure_dir,
            figsize=figsize,
            wrap_width=wrap_width,
            fig_format=fig_format,
        )

        figures[o] = fig
        print(f"  Done — {ONTOLOGY[o]}")
        print("  " + "-" * 40)

    return figures