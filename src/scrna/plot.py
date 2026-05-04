from __future__ import annotations

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



def plot_umap_by_category(
    adata, col='sample', color_map=None, ax=None, title=None,
    dot_size=0.05, text_size=5, alpha=0.75,
    figsize=(6, 5), cmap='tab20',
    legend=True, legend_markerscale=20,
    xlabel='UMAP 1', ylabel='UMAP 2',
    spines=False, legend_order=None):
    """
    Plot UMAP scatter colored by a categorical column.

    Args:
        adata:                AnnData object with obsm['X_umap'] and obs[col]
        col:                  Column in adata.obs to color by (default: 'sample')
        color_map:            Dict mapping category -> color. Auto-generated if None.
        ax:                   Matplotlib Axes object. Creates new figure if None.
        title:                Plot title. Defaults to col name.
        dot_size:             Size of scatter points (default: 0.05)
        text_size:            Font size for labels, title, and legend (default: 5)
        alpha:                Transparency of points (default: 0.75)
        figsize:              Figure size as (width, height) (default: (6, 5))
        cmap:                 Colormap name for auto color generation (default: 'tab20')
        legend:               Whether to show legend (default: True)
        legend_markerscale:   Scale of legend markers (default: 20)
        xlabel:               X-axis label (default: 'UMAP 1')
        ylabel:               Y-axis label (default: 'UMAP 2')
        spines:               Whether to show plot spines/borders (default: False)
        legend_order:         List of categories for legend order. 
                              'auto' to sort numerically if possible (default: None = alphabetical)
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Build UMAP dataframe
    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        columns=['umap1', 'umap2'],
        index=adata.obs.index
    )
    umap_df[col] = adata.obs[col]

    categories = sorted(umap_df[col].unique())

    # Auto-generate color map if not provided
    if color_map is None:
        _cmap = plt.colormaps.get_cmap(cmap).resampled(len(categories))
        color_map = {cat: _cmap(i) for i, cat in enumerate(categories)}

    # Create axes if not provided
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
            color=color_map[cat]
        )

    # Axes formatting
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, size=text_size)
    ax.set_ylabel(ylabel, size=text_size)
    ax.set_title(title or col, size=text_size)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(spines)

    if legend:
        # Determine legend order
        if legend_order == 'auto':
            try:
                order = sorted(categories, key=lambda x: float(x))
            except (ValueError, TypeError):
                order = categories
        elif legend_order is not None:
            order = legend_order
        else:
            order = categories

        # Reorder handles/labels
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[o] for o in order if o in label_to_handle]
        ordered_labels = [o for o in order if o in label_to_handle]

        ax.legend(
            ordered_handles, ordered_labels,
            markerscale=legend_markerscale,
            fontsize=text_size,
            bbox_to_anchor=(1, 1),
            loc='upper left',
            frameon=False
        )

    return ax

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
    panel_width: float = 2.88,
    panel_height: float = 2.75,
    dpi: int = 480,
    obsm_key: str = "X_umap",
    save_path: Optional[str] = None,
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
    panel_width : float
        Panel width in inches.
    panel_height : float
        Panel height in inches.
    dpi : int
        Figure DPI.
    obsm_key : str
        Key in adata.obsm for 2-D coordinates.
    save_path : str, optional
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

        fig.supxlabel("UMAP 1", fontsize=8)
        fig.supylabel("UMAP 2", fontsize=8)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


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