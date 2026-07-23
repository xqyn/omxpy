from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import math
import os
from typing import Optional, Union

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path

def _draw_panel(
    ax: Axes,
    udf: pd.DataFrame,
    col: str,
    cats: list,
    color_map: dict,
    dot_size: float,
    alpha: float,
    bg_alpha: Optional[float],
    background: bool,
    bg_color: str = "silver",
    label_cats: bool = False,
    label_fontsize: float = 6,
    label_fontweight: str = "bold",
    label_outline: bool = True,
    label_arrow: bool = False,
    label_offset: tuple = (0.5, 0.5),
) -> None:
    """Draw one UMAP panel: optional grey context layer + colored category scatter.

    Used internally by `plot_umap` for both the single-panel view (background=False)
    and each facet of the split grid (background=True). Not part of the public API.

    Args:
        ax: Matplotlib Axes to draw into.
        udf: DataFrame with columns ['u1', 'u2', col] — UMAP coords + category labels.
        col: Column in `udf` used for category masking/coloring.
        cats: Categories to plot on this axes (all categories for single panel,
            one category for a split facet).
        color_map: Mapping {category: color} for foreground points/labels.
        dot_size: Scatter marker size (matplotlib `s=`).
        alpha: Foreground marker opacity.
        bg_alpha: Background marker opacity. Required if `background=True`.
        background: If True, first plot all cells in `bg_color` for spatial context,
            then overlay only `cats` in color. If False, plot only `cats`.
        bg_color: Color for the background context layer. Ignored if background=False.
        label_cats: If True, annotate each category's centroid with its name.
        label_fontsize: Font size for centroid labels.
        label_fontweight: Font weight for centroid labels.
        label_outline: If True, add a white/black stroke outline behind labels
            for readability against variable backgrounds.
        label_arrow: If True, offset the label from the centroid and connect
            with a thin line (avoids label overlapping points). If False,
            place the label directly at the centroid.
        label_offset: (x, y) offset in UMAP coordinate units, used only when
            `label_arrow=True`.

    Returns:
        None. Mutates `ax` in place.
    """
    u1, u2 = udf["u1"].values, udf["u2"].values

    if background:
        ax.scatter(u1, u2, s=dot_size, c=bg_color, alpha=bg_alpha,
                   rasterized=True, linewidths=0)

    for cat in cats:
        mask = (udf[col] == cat).values
        ax.scatter(u1[mask], u2[mask], s=dot_size, alpha=alpha,
                   color=color_map[cat], label=cat, rasterized=True,
                   linewidths=0)

        if label_cats:
            cx, cy = udf.loc[mask, "u1"].median(), udf.loc[mask, "u2"].median()
            if label_arrow:
                txt = ax.annotate(
                    str(cat), xy=(cx, cy),
                    xytext=(cx + label_offset[0], cy + label_offset[1]),
                    fontsize=label_fontsize, fontweight=label_fontweight,
                    color=color_map[cat], ha="center", va="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", color=color_map[cat], lw=0.8),
                )
            else:
                txt = ax.text(cx, cy, str(cat), fontsize=label_fontsize,
                              fontweight=label_fontweight, ha="center",
                              va="center", color=color_map[cat], zorder=5)
            if label_outline:
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])


def plot_umap(
    adata,
    col: str,
    conditions: Optional[list] = None,
    color_map: Optional[dict] = None,
    palette: str = "tab20",
    obsm_key: str = "X_umap",
    split: bool = False,
    ax: Optional[Axes] = None,          # only used when split=False
    ncol: int = 3,
    blank_pos: Optional[int] = None,
    dot_size: float = 2.0,
    alpha: float = 0.75,
    bg_alpha: Optional[float] = None,   # split mode only; auto-scaled if None
    bg_color: str = "silver",           # split mode only
    text_size: float = 6,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    legend: bool = True,
    legend_markerscale: float = 5,
    legend_order: Optional[Union[list, str]] = None,
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    axis_show: bool = True,
    spines: bool = False,
    label_cats: bool = False,
    label_fontsize: Optional[float] = None,
    label_fontweight: str = "bold",
    label_outline: bool = True,
    label_arrow: bool = False,
    label_offset: tuple = (0.5, 0.5),
    panel_width: float = 2.88,
    panel_height: float = 2.75,
    figsize: Optional[tuple] = None,    # single-plot mode override
    dpi: int = 320,
    fig_dir: Optional[str] = None,
) -> Union[Axes, Figure]:
    """Plot UMAP colored by an obs column, as a single panel or a per-condition grid.

    Args:
        adata: AnnData object with `obsm[obsm_key]` (2D embedding) and
            `obs[col]` (categorical/label column).
        col: Column in `adata.obs` used for coloring and (if split=True) faceting.
        conditions: Ordered list of categories to plot. Defaults to
            `sorted(adata.obs[col].unique())`.
        color_map: Mapping {category: color}. Auto-generated from `palette` if None.
        palette: Seaborn palette name used to auto-generate `color_map`.
        obsm_key: Key in `adata.obsm` holding the 2D embedding (e.g. 'X_umap').
        split: If False, plot all categories overlaid on one axes. If True,
            plot one facet per category, each with a grey full-dataset
            background for context.
        ax: Existing Axes to draw into. Only honored when `split=False`;
            ignored when `split=True` (a new grid figure is always created).
        ncol: Number of columns in the split grid. Ignored if `split=False`.
        blank_pos: Zero-based slot index at which to insert an empty/hidden
            panel in the split grid (e.g. to align facets visually).
        dot_size: Scatter marker size (matplotlib `s=`).
        alpha: Foreground marker opacity.
        bg_alpha: Background layer opacity in split mode. If None, auto-scaled
            as `min(0.3, 5000 / n_cells)`.
        bg_color: Background layer color in split mode.
        text_size: Base font size for axis labels/legend/title fallback.
        title: Panel/figure title. Defaults to `col` if None. Only applied in
            single-panel mode.
        title_fontsize: Font size for titles. Falls back to `text_size` if None.
        legend: If True, draw a legend (single-panel mode only).
        legend_markerscale: Marker size multiplier in the legend.
        legend_order: 'auto' to sort categories numerically, an explicit list
            to set order, or None to use `conditions`/discovery order.
        xlabel: X-axis label text.
        ylabel: Y-axis label text.
        axis_show: If True, draw axis labels (xlabel/ylabel or fig.supxlabel/
            supylabel in split mode).
        spines: If True, show axes spines/frame; if False, hide them.
        label_cats: If True, annotate each category's centroid with its name.
        label_fontsize: Font size for centroid labels. Falls back to `text_size`.
        label_fontweight: Font weight for centroid labels.
        label_outline: If True, outline centroid labels for readability.
        label_arrow: If True, offset centroid labels and connect with a line.
        label_offset: (x, y) offset in UMAP units, used only if `label_arrow=True`.
        panel_width: Per-panel width in inches (split mode).
        panel_height: Per-panel height in inches (split mode).
        figsize: Figure size override for single-panel mode. Ignored if split=True.
        dpi: Figure resolution.
        fig_dir: If provided, saves the figure as PNG to `{fig_dir}/umap_{col}.png`.

    Returns:
        matplotlib.axes.Axes if split=False, matplotlib.figure.Figure if split=True.

    Raises:
        KeyError: If `obsm_key` is not in `adata.obsm` or `col` is not in `adata.obs`.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm")
    if col not in adata.obs:
        raise KeyError(f"'{col}' not found in adata.obs")

    udf = pd.DataFrame(adata.obsm[obsm_key][:, :2], columns=["u1", "u2"],
                       index=adata.obs.index)
    udf[col] = adata.obs[col].values

    cats = conditions or sorted(udf[col].unique())
    if color_map is None:
        colors = sns.color_palette(palette, len(cats))
        color_map = dict(zip(cats, colors))

    fs_label = label_fontsize if label_fontsize is not None else text_size

    # ---- single panel ----
    if not split:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (6, 5), dpi=dpi)

        # NOTE: previously nested under `if ax is None:` — bug fixed by
        # dedenting so this always runs, whether ax was passed in or created.
        _draw_panel(ax, udf, col, cats, color_map, dot_size, alpha, None,
                   background=False, bg_color=bg_color, label_cats=label_cats,
                   label_fontsize=fs_label, label_fontweight=label_fontweight,
                   label_outline=label_outline, label_arrow=label_arrow,
                   label_offset=label_offset)

        ax.set_xticks([]); ax.set_yticks([])
        if axis_show:
            ax.set_xlabel(xlabel, size=text_size)
            ax.set_ylabel(ylabel, size=text_size)
        ax.set_title(title or col, size=title_fontsize or text_size)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(spines)
        if legend:
            order = cats
            if legend_order == "auto":
                try:
                    order = sorted(cats, key=lambda x: float(x))
                except (ValueError, TypeError):
                    pass
            elif legend_order is not None:
                order = legend_order
            handles, labels = ax.get_legend_handles_labels()
            lh = dict(zip(labels, handles))
            ax.legend([lh[o] for o in order if o in lh],
                     [o for o in order if o in lh],
                     markerscale=legend_markerscale, fontsize=text_size,
                     bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
        fig = ax.figure
        result = ax

    # ---- split grid ----
    else:
        if bg_alpha is None:
            bg_alpha = float(min(0.3, 5_000 / max(len(udf), 1)))
        slots = list(cats)
        if blank_pos is not None:
            slots.insert(max(0, min(blank_pos, len(slots))), None)
        nrow = math.ceil(len(slots) / ncol)
        fig, axs = plt.subplots(nrow, ncol,
                                figsize=(panel_width * ncol, panel_height * nrow),
                                dpi=dpi)
        axs_flat = list(axs.flat) if hasattr(axs, "flat") else [axs]
        for i, a in enumerate(axs_flat):
            if i >= len(slots) or slots[i] is None:
                a.set_visible(False)
                continue
            cat = slots[i]
            _draw_panel(a, udf, col, [cat], color_map, dot_size, alpha, bg_alpha,
                       background=True, bg_color=bg_color,
                       label_cats=label_cats, label_fontsize=fs_label,
                       label_fontweight=label_fontweight,
                       label_outline=label_outline, label_arrow=label_arrow,
                       label_offset=label_offset)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(cat, fontsize=title_fontsize or text_size)
            a.set_aspect("equal")
            a.set_frame_on(spines)
        if axis_show:
            fig.supxlabel(xlabel, fontsize=text_size)
            fig.supylabel(ylabel, fontsize=text_size)
        plt.tight_layout()
        result = fig
        plt.close(fig)

    if fig_dir:
        save_path = Path(fig_dir)
        if save_path.suffix.lower() not in (".png", ".pdf", ".svg"):
            save_path = save_path.with_suffix(".png")
        #save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return result

def plot_umap_gene(
    adata,
    gene,
    # Layout
    figsize=(6, 5),
    point_size=10,
    alpha=0.8,
    # Colormap
    cmap="magma",
    vmin=None,
    vmax=None,
    vcenter=None,           # For diverging colormaps (e.g. 'RdBu_r')
    na_color="#d3d3d3",     # Color for cells with 0 / NaN expression
    na_in_legend=True,
    # Axes & title
    title=None,
    title_fontsize=13,
    xlabel="UMAP 1",
    ylabel="UMAP 2",
    label_fontsize=10,
    tick_fontsize=8,
    show_ticks=False,
    # Colorbar
    colorbar=True,
    colorbar_label=None,
    colorbar_shrink=0.5,
    colorbar_aspect=15,
    # Layer (raw, normalised, scaled, etc.)
    layer=None,             # None = adata.X; else adata.layers[layer]
    use_raw=True,           # if True, pull from adata.raw
    # Background
    bg_color="white",
    # Ordering
    sort_order=True,        # plot high-expression cells on top
    # Output
    save=None,              # e.g. "umap_Pou5f1.pdf"
    ax=None,
    frame=True,
):
    """
    Customisable UMAP scatter coloured by a single gene's expression.

    Parameters
    ----------
    adata   : AnnData object (must have adata.obsm['X_umap'])
    gene    : str, gene name to colour by
    layer   : str or None — which layer to use (None → adata.X)
    use_raw : bool — pull expression from adata.raw (overrides layer)
    ...     : all other kwargs documented above

    Returns
    -------
    fig, ax
    """

    # ── 1. Pull UMAP coordinates ──────────────────────────────────────────
    umap = adata.obsm["X_umap"]
    x, y = umap[:, 0], umap[:, 1]

    # ── 2. Pull expression values ─────────────────────────────────────────
    if use_raw and adata.raw is not None:
        idx = adata.raw.var_names.get_loc(gene)
        expr = np.asarray(adata.raw.X[:, idx].todense()).flatten()
    elif layer is not None:
        idx = adata.var_names.get_loc(gene)
        mat = adata.layers[layer]
        expr = np.asarray(mat[:, idx].todense() if hasattr(mat, "todense") else mat[:, idx]).flatten()
    else:
        idx = adata.var_names.get_loc(gene)
        mat = adata.X
        expr = np.asarray(mat[:, idx].todense() if hasattr(mat, "todense") else mat[:, idx]).flatten()

    # ── 3. Split zero / non-zero cells ───────────────────────────────────
    mask_expr = expr > 0
    x_na, y_na = x[~mask_expr], y[~mask_expr]
    x_ex, y_ex, c_ex = x[mask_expr], y[mask_expr], expr[mask_expr]

    # ── 4. Sort so brightest dots sit on top ─────────────────────────────
    if sort_order:
        order = np.argsort(c_ex)
        x_ex, y_ex, c_ex = x_ex[order], y_ex[order], c_ex[order]

    # ── 5. Colour-norm ────────────────────────────────────────────────────
    _vmin = vmin if vmin is not None else c_ex.min() if mask_expr.any() else 0
    _vmax = vmax if vmax is not None else c_ex.max() if mask_expr.any() else 1

    if vcenter is not None:
        norm = mcolors.TwoSlopeNorm(vmin=_vmin, vcenter=vcenter, vmax=_vmax)
    else:
        norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

    # ── 6. Figure / axes ──────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    else:
        fig = ax.get_figure()

    ax.set_facecolor(bg_color)

    # ── 7. Plot NA cells first (background) ──────────────────────────────
    ax.scatter(x_na, y_na, s=point_size, c=na_color, alpha=alpha,
               linewidths=0, rasterized=True)

    # ── 8. Plot expressing cells ──────────────────────────────────────────
    sc = ax.scatter(x_ex, y_ex, s=point_size, c=c_ex, cmap=cmap,
                    norm=norm, alpha=alpha, linewidths=0, rasterized=True)

    # ── 9. Colorbar ───────────────────────────────────────────────────────
    if colorbar and mask_expr.any():
        cb = fig.colorbar(sc, ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect)
        cb.set_label(colorbar_label or gene, fontsize=label_fontsize)
        cb.ax.tick_params(labelsize=tick_fontsize)

    # ── 10. Labels & cosmetics ────────────────────────────────────────────
    ax.set_title(title or gene, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_aspect('equal')
    ax.set_frame_on(frame)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if na_in_legend and x_na.size > 0:
        ax.legend(markerscale=1.5, fontsize=tick_fontsize,
                  frameon=False, loc="lower right")

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved → {save}")

    return fig, ax


def plot_violin(
    adata,
    keys: Union[str, List[str]],
    groupby: Optional[str] = None,
    # --- Layout ---
    figsize: tuple = (10, 5),
    dpi: int = 100,
    ncols: int = 4,
    # --- Colors ---
    palette: Optional[Union[str, List[str], dict]] = None,
    stripplot_color: str = "black",
    stripplot_alpha: float = 0.5,
    # --- Violin shape ---
    scale: str = "width",           # "width", "count", "area"
    inner: Optional[str] = "box",   # "box", "quartile", "point", "stick", None
    bw: float = 0.5,                # bandwidth / smoothing
    cut: float = 2.0,               # how far violin extends beyond data range
    # --- Strip/jitter ---
    stripplot: bool = True,
    jitter: Union[bool, float] = 0.4,
    size: float = 1.5,              # dot size
    # --- Axes & labels ---
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    rotation: float = 45,
    fontsize: int = 11,
    title_fontsize: int = 13,
    # --- Legend ---
    show_legend: bool = True,
    legend_loc: str = "lower center",   # any matplotlib loc string
    legend_anchor: tuple = (0.5, -0.08),
    legend_ncol: Optional[int] = None,  # defaults to number of groups
    # --- Grid & spines ---
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    despine: bool = True,
    # --- Saving ---
    save: Optional[str] = None,    # e.g. "violin.png"
    show: bool = True,
    **kwargs,                       # passed directly to sc.pl.violin
) -> Optional[plt.Figure]:
    """
    A fully customizable wrapper around sc.pl.violin.

    Parameters
    ----------
    adata               AnnData object.
    keys                Gene(s) or obs column(s) to plot.
    groupby             Column in adata.obs to group by.
    figsize             Figure size (width, height) in inches.
    dpi                 Resolution.
    ncols               Max columns when plotting multiple keys.
    palette             Color palette – seaborn name, list of hex, or dict {group: color}.
    stripplot_color     Color of jitter dots.
    stripplot_alpha     Transparency of jitter dots (0–1).
    scale               Violin normalization: 'width', 'count', or 'area'.
    inner               Interior marks: 'box', 'quartile', 'point', 'stick', or None.
    bw                  Bandwidth smoothing for the KDE.
    cut                 Extension of violin tails beyond data range.
    stripplot           Whether to overlay a strip plot.
    jitter              Jitter amount (float) or True/False.
    size                Dot size for the strip plot.
    xlabel              X-axis label (defaults to groupby value).
    ylabel              Y-axis label.
    title               Figure-level suptitle.
    rotation            Tick label rotation in degrees (kept for scanpy compat, not applied
                        to x-tick labels since they are hidden).
    fontsize            Axis label / tick / legend fontsize.
    title_fontsize      Title fontsize.
    show_legend         Whether to draw the shared group legend.
    legend_loc          Matplotlib legend loc string, e.g. 'lower center', 'center right'.
    legend_anchor       bbox_to_anchor for the legend.
    legend_ncol         Number of columns in the legend (defaults to number of groups).
    show_grid           Show horizontal grid lines.
    grid_alpha          Grid line transparency.
    despine             Remove top and right spines.
    save                File path to save figure (e.g. 'plot.png').
    show                Call plt.show().
    **kwargs            Extra keyword arguments forwarded to sc.pl.violin.

    Returns
    -------
    fig : matplotlib Figure (or None if show=True consumed it).
    """

    # ── call scanpy violin ────────────────────────────────────────────────
    axes = sc.pl.violin(
        adata,
        keys=keys,
        groupby=groupby,
        palette=palette,
        scale=scale,
        inner=inner,
        bw=bw,
        cut=cut,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        rotation=rotation,
        show=False,      # we handle show ourselves
        **kwargs,
    )

    # sc.pl.violin may return a single Axes or a list
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    axes = [ax for ax in axes if ax is not None]

    fig = axes[0].get_figure()
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)

    # ── build shared legend handles from the first axes ───────────────────
    legend_handles: list[mpatches.Patch] = []

    if show_legend and groupby is not None:
        ax0 = axes[0]
        obs_col = adata.obs[groupby]

        # get ordered group names
        if hasattr(obs_col, "cat"):
            groups = list(obs_col.cat.categories)
        else:
            groups = list(obs_col.unique())

        # PolyCollections in a violin axes are laid out as:
        #   [violin_body_group0, violin_body_group1, ..., strip_dots, ...]
        # We grab only the first len(groups) PolyCollections (the violin bodies).
        poly_collections = [
            c for c in ax0.collections
            if "PolyCollection" in type(c).__name__
        ]

        for i, grp in enumerate(groups):
            if i < len(poly_collections):
                fc = poly_collections[i].get_facecolor()
                color = fc[0] if len(fc) > 0 else "gray"
            else:
                color = "gray"
            legend_handles.append(mpatches.Patch(facecolor=color, label=str(grp)))

    # ── per-axis styling ──────────────────────────────────────────────────
    for ax in axes:

        # ── FIX 1: remove the spurious horizontal line at y=0 ────────────
        # Scanpy / seaborn draws a thin baseline at y=0 as a Line2D.
        for line in ax.lines:
            yd = line.get_ydata()
            if len(yd) > 0 and len(set(yd.tolist())) == 1 and float(yd[0]) == 0.0:
                line.set_visible(False)

        # ── FIX 2: hide x-tick labels; use legend instead ─────────────────
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        # strip-plot dot color & alpha
        for coll in ax.collections:
            if hasattr(coll, "get_offsets") and coll.get_offsets().size > 0:
                if "PathCollection" in type(coll).__name__:
                    coll.set_color(stripplot_color)
                    coll.set_alpha(stripplot_alpha)

        # grid
        if show_grid:
            ax.yaxis.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.7)
            ax.set_axisbelow(True)
        else:
            ax.yaxis.grid(False)

        # spines
        if despine:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # tick fontsize
        ax.tick_params(labelsize=fontsize)

        # axis labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        else:
            ax.set_xlabel("")   # suppress the default groupby label per-axis

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize)

    # ── figure-level title ────────────────────────────────────────────────
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=1.02)

    # ── shared legend ─────────────────────────────────────────────────────
    if legend_handles:
        ncol = legend_ncol if legend_ncol is not None else len(legend_handles)
        fig.legend(
            handles=legend_handles,
            title=groupby,
            loc=legend_loc,
            ncol=ncol,
            bbox_to_anchor=legend_anchor,
            fontsize=fontsize,
            title_fontsize=fontsize,
            frameon=False,
        )

    fig.tight_layout()

    # ── save / show ───────────────────────────────────────────────────────
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=dpi)
        print(f"Saved → {save}")

    if show:
        plt.show()
        return None

    return fig


# ── plot_dotplot ──────────────────────────────────────────────────────────────
def plot_dotplot(
    adata: sc.AnnData,
    up_genes: list[str],
    down_genes: list[str],
    groupby: str,
    groups: list[str] | None = None,
    setting: str = "dotplot",
    n_genes: int = 10,
    use_raw: bool = True,
    cmap: str = "Reds",
    standard_scale: str = "var",
    colorbar_title: str = "Scaled\nexpression",
    dot_max: float = 1.0,
    dot_min: float = 0.05,
    smallest_dot: float = 1.0,
    figsize: tuple[int, int] = (6, 4),
    swap_axes: bool = False,
    dendrogram: bool = False,
    dot_edge_color: str = "black",
    dot_edge_lw: float = 1.0,
    var_group_rotation: float = 0,
    fig_dir: str | None = None,
    dpi: int = 150,
    sample_design: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    fig_format: tuple[str, ...] = ("png",),
) -> plt.Figure:
    """Dotplot of top differentially expressed genes grouped by UP/DOWN.

    Args:
        adata (sc.AnnData): Annotated data matrix.
        up_genes (list[str]): Ordered list of upregulated gene names.
        down_genes (list[str]): Ordered list of downregulated gene names.
        groupby (str): Column in ``adata.obs`` to group cells by.
        groups (list[str] | None): Subset of groupby categories to plot.
                               Shows all if None. Defaults to None.
        setting (str): Filename prefix used when saving. Defaults to
            ``"dotplot"``.
        n_genes (int): Number of top genes to show per group. Defaults to
            ``10``.
        use_raw (bool): Whether to use ``adata.raw``. Defaults to ``True``.
        cmap (str): Colormap for expression. Defaults to ``"Reds"``.
        standard_scale (str): Axis to standardise across — ``"var"`` or
            ``"group"``. Defaults to ``"var"``.
        colorbar_title (str): Label for the colorbar. Defaults to
            ``"Scaled\\nexpression"``.
        dot_max (float): Maximum dot size. Defaults to ``1.0``.
        dot_min (float): Minimum dot size. Defaults to ``0.05``.
        smallest_dot (float): Minimum rendered dot size in pt. Defaults to
            ``1.0``.
        figsize (tuple[int, int]): Figure dimensions as (width, height) in
            inches. Defaults to ``(6, 4)``.
        swap_axes (bool): Swap genes and groups axes. Defaults to ``False``.
        dendrogram (bool): Show dendrogram. Defaults to ``False``.
        dot_edge_color (str): Dot border color. Defaults to ``"black"``.
        dot_edge_lw (float): Dot border line width. Defaults to ``1.0``.
        var_group_rotation (float): Rotation of group labels in degrees.
            Defaults to ``0``.
        fig_dir (str | None): Directory to save the figure; skips saving
            when ``None``. Defaults to ``None``.
        dpi (int): Resolution for saved figure. Defaults to ``150``.
        fig_format (tuple[str, ...]): File format(s) to save, e.g.
            ``("png", "pdf")``. Defaults to ``("png",)``.
        sample_design (list[str] | None): Sample order for ``add_totals``.
            Requires ``color_map``. Defaults to ``None``.
        color_map (dict[str, str] | None): Mapping of sample names to colors
            for ``add_totals``. Requires ``sample_design``. Defaults to
            ``None``.
    Returns:
        plt.Figure: The rendered matplotlib figure.

    Raises:
        ValueError: If ``n_genes`` exceeds the length of either gene list.
    """
    if groups is not None:
        adata = adata[adata.obs[groupby].isin(groups)].copy()

    if n_genes > len(up_genes) or n_genes > len(down_genes):
        raise ValueError(
            f"n_genes={n_genes} exceeds the length of up_genes "
            f"({len(up_genes)}) or down_genes ({len(down_genes)})."
        )
    gene_groups = {
        "UP": up_genes[:n_genes],
        "DOWN": down_genes[:n_genes],
    }

    dp = sc.pl.dotplot(
        adata,
        gene_groups,
        var_group_rotation=var_group_rotation,
        groupby=groupby,
        #categories_order=groups,
        use_raw=use_raw,
        cmap=cmap,
        standard_scale=standard_scale,
        colorbar_title=colorbar_title,
        dot_max=dot_max,
        dot_min=dot_min,
        smallest_dot=smallest_dot,
        figsize=figsize,
        swap_axes=swap_axes,
        dendrogram=dendrogram,
        return_fig=True,
    )
    dp.style(
        dot_edge_color=dot_edge_color,
        dot_edge_lw=dot_edge_lw,
    )

    if sample_design is not None and color_map is not None:
        dp.add_totals(color=[color_map[c] for c in sample_design])

    #dp.make_figure()
    fig = dp.get_axes()["mainplot_ax"].get_figure()

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        for fmt in fig_format:
            fig.savefig(
                f"{fig_dir}/{setting}_dotplot.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
            )

    return fig

# --- archive -----------------------------------------------

def plot_umap_by_category(
    adata, 
    col='sample', 
    color_map=None, 
    ax=None, 
    title=None,
    dot_size=0.05, 
    text_size=5, 
    alpha=0.75,
    figsize=(6, 5), 
    cmap='tab20',
    legend=True, 
    legend_markerscale=20,
    xlabel='UMAP 1', 
    ylabel='UMAP 2',
    spines=False, 
    legend_order=None,
    label_cats=False, 
    label_fontsize=None, 
    label_fontweight='bold',
    label_outline=True, 
    label_arrow=False,
    label_offset=(0.5, 0.5),   # (x, y) offset in UMAP coordinate units
):
    """
    ...
    label_cats:       Show category name at median centroid (default: False)
    label_fontsize:   Font size for category labels; defaults to text_size
    label_fontweight: Font weight for category labels (default: 'bold')
    label_outline:    Add white outline around labels for readability (default: True)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import pandas as pd

    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        columns=['umap1', 'umap2'],
        index=adata.obs.index
    )
    umap_df[col] = adata.obs[col]

    categories = sorted(umap_df[col].unique())

    if color_map is None:
        _cmap = plt.colormaps.get_cmap(cmap).resampled(len(categories))
        color_map = {cat: _cmap(i) for i, cat in enumerate(categories)}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for cat in categories:
        mask = umap_df[col] == cat
        cat_umap = umap_df[mask]
        ax.scatter(
            cat_umap['umap1'],
            cat_umap['umap2'],
            label=cat,
            s=dot_size,
            alpha=alpha,
            color=color_map[cat],
        )

        if label_cats:
            cx = cat_umap['umap1'].median()
            cy = cat_umap['umap2'].median()
            fs = label_fontsize if label_fontsize is not None else text_size

            if label_arrow:
                txt = ax.annotate(
                    str(cat),
                    xy=(cx, cy),                        # arrow tip → centroid
                    xytext=(cx + label_offset[0],
                            cy + label_offset[1]),      # text position
                    fontsize=fs,
                    fontweight=label_fontweight,
                    color=color_map[cat],
                    ha='center', va='center',
                    zorder=5,
                    arrowprops=dict(
                        arrowstyle='-',                 # plain line, no arrowhead
                        color=color_map[cat],
                        lw=0.8,
                    ),
                )
            else:
                txt = ax.text(
                    cx, cy, str(cat),
                    fontsize=fs,
                    fontweight=label_fontweight,
                    ha='center', va='center',
                    color=color_map[cat],
                    zorder=5,
                )

            if label_outline:
                txt.set_path_effects([
                    pe.withStroke(linewidth=2, foreground='black')
                ])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, size=text_size)
    ax.set_ylabel(ylabel, size=text_size)
    ax.set_title(title or col, size=text_size)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(spines)

    if legend:
        if legend_order == 'auto':
            try:
                order = sorted(categories, key=lambda x: float(x))
            except (ValueError, TypeError):
                order = categories
        elif legend_order is not None:
            order = legend_order
        else:
            order = categories

        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[o] for o in order if o in label_to_handle]
        ordered_labels  = [o for o in order if o in label_to_handle]

        ax.legend(
            ordered_handles, ordered_labels,
            markerscale=legend_markerscale,
            fontsize=text_size,
            bbox_to_anchor=(1, 1),
            loc='upper left',
            frameon=False,
        )

    return ax


def plot_umap_split(
    adata,
    col: str,
    conditions: Optional[list] = None,
    color_dict: Optional[dict] = None,
    palette: str = "tab20",
    split: bool = False,
    ncol: int = 3,
    blank_pos: Optional[int] = None,
    dot_size: float = 2.0,
    bg_alpha: Optional[float] = None,
    fg_alpha: float = 0.75,
    title_fontsize: int = 7,
    legend_fontsize: int = 6,
    axis_show: bool = True,
    panel_width: float = 2.88,
    panel_height: float = 2.75,
    dpi: int = 480,
    obsm_key: str = "X_umap",
    fig_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot UMAP coloured by a metadata column, with an optional per-condition
    split view.

    Parameters
    ----------
    adata : AnnData
        AnnData object with obsm[obsm_key] and obs[col].
    col : str
        obs column to colour/split by.
    conditions : list, optional
        Ordered condition list; defaults to adata.obs[col].unique().
    color_dict : dict, optional
        Mapping of {condition: color}; auto-generated from palette if None.
    palette : str
        Seaborn palette used when color_dict is None.
    split : bool
        False → single coloured UMAP.
        True  → one panel per condition (highlight + grey background).
    ncol : int
        Number of columns in split grid (ignored when split=False).
    blank_pos : int, optional
        Zero-based panel index at which to insert a blank (invisible) axes.
        All conditions are shifted right from that position onward.
    dot_size : float
        Scatter marker size (s=).
    bg_alpha : float, optional
        Background alpha in split mode; auto-scales if None.
    fg_alpha : float
        Foreground / single-plot alpha.
    title_fontsize : int
        Per-panel title font size (split mode).
    legend_fontsize : int
        Legend font size (single mode).
    axis_show: bool
        Show axis or not
    panel_width : float
        Panel width in inches.
    panel_height : float
        Panel height in inches.
    dpi : int
        Figure DPI.
    obsm_key : str
        Key in adata.obsm for 2-D coordinates.
    fig_dir : str, optional
        Saves figure to this path if provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── shared setup ─────────────────────────────────────────────────────────
    udf = pd.DataFrame(
        adata.obsm[obsm_key],
        columns=["u1", "u2"],
        index=adata.obs.index,
    )
    udf[col] = adata.obs[col].values

    if conditions is None:
        conditions = udf[col].unique().tolist()

    if color_dict is None:
        colors = sns.color_palette(palette, len(conditions))
        color_dict = dict(zip(conditions, colors))
    elif isinstance(color_dict, str):
        color_dict = {cond: color_dict for cond in conditions}

    u1 = udf["u1"].values
    u2 = udf["u2"].values

    # ── single plot ───────────────────────────────────────────────────────────
    if not split:
        fig, ax = plt.subplots(
            figsize=(panel_width * 1.4, panel_height * 1.4),
            dpi=dpi,
        )

        for cond in conditions:
            mask = (udf[col] == cond).values
            ax.scatter(
                u1[mask], u2[mask],
                s=dot_size,
                c=[color_dict.get(cond, "steelblue")],
                alpha=fg_alpha,
                rasterized=True,
                linewidths=0,
                label=cond,
            )

        ax.legend(
            markerscale=2,
            fontsize=legend_fontsize,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
            title=col,
            title_fontsize=legend_fontsize,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if axis_show:
            ax.set_xlabel("UMAP 1", fontsize=8)
            ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_title(col, fontsize=title_fontsize + 1)
        plt.tight_layout()

    # ── split plot ────────────────────────────────────────────────────────────
    else:
        if bg_alpha is None:
            bg_alpha = float(min(0.3, 5_000 / max(len(udf), 1)))

        # Total slots = conditions + 1 blank slot (if blank_pos is set)
        n_slots = len(conditions) + (1 if blank_pos is not None else 0)
        nrow = math.ceil(n_slots / ncol)

        fig, axs = plt.subplots(
            nrows=nrow,
            ncols=ncol,
            figsize=(panel_width * ncol, panel_height * nrow),
            dpi=dpi,
        )
        axs_flat = list(axs.flat) if hasattr(axs, "flat") else [axs]

        # Build slot → condition mapping, inserting None at blank_pos
        slots = list(conditions)
        if blank_pos is not None:
            insert_at = max(0, min(blank_pos, len(slots)))
            slots.insert(insert_at, None)  # None marks the blank panel

        for i, ax in enumerate(axs_flat):
            # Hide axes beyond the used slots
            if i >= len(slots):
                ax.set_visible(False)
                continue

            cond = slots[i]

            # Blank panel
            if cond is None:
                ax.set_visible(False)
                continue

            mask = (udf[col] == cond).values

            ax.scatter(
                u1, u2,
                s=dot_size,
                c="silver",
                alpha=bg_alpha,
                rasterized=True,
                linewidths=0,
            )
            ax.scatter(
                u1[mask], u2[mask],
                s=dot_size,
                c=[color_dict.get(cond, "steelblue")],
                alpha=fg_alpha,
                rasterized=True,
                linewidths=0,
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(cond, fontsize=title_fontsize)
            ax.set_aspect("equal")
            ax.set_frame_on(False)

        if axis_show:
            fig.supxlabel("UMAP 1", fontsize=8)
            fig.supylabel("UMAP 2", fontsize=8)
            # ax.set_xlabel("UMAP 1", fontsize=8)
            # ax.set_ylabel("UMAP 2", fontsize=8)
       
        plt.tight_layout()

    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        save_path = f'{fig_dir}/umap_cond_{col}.png'
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig

