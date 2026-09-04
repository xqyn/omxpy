"""
omxpy — single-cell embedding plotting
LUMC, 2026
 
Merged categorical/continuous UMAP (or any 2D obsm embedding) plotting.
Replaces: plot_umap, plot_umap_cont, plot_umap_split, plot_umap_by_category.
 
Usage:
    plot_embedding(adata, "cell_type")                       # categorical, auto-detected
    plot_embedding(adata, "cCARLIN_hamming_umi")              # continuous, auto-detected
    plot_embedding(adata, "pseudotime", zero_color="black")   # flag exact-zero cells
    plot_embedding(adata, "cell_type", split=True, ncol=4)    # per-category facet grid
"""

from __future__ import annotations
 
import math
from pathlib import Path
from typing import Literal, Optional, Union
 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from matplotlib.figure import Figure


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axes import Axes

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

 
import math
from pathlib import Path
from typing import Literal, Optional, Union
 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from matplotlib.figure import Figure
 
 
# ── internal draw helpers (one scatter "panel" each; not part of the public API) ──

def _sort_categories(cats) -> list:
    """Sort categories numerically if all values parse as numbers (e.g.
    leiden cluster labels '0','1',...,'22'), else fall back to lexical sort."""
    try:
        return sorted(cats, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(cats)

def _draw_panel_cat(
    ax: Axes, udf: pd.DataFrame, col: str, cats: list, color_map: dict,
    dot_size: float, alpha: float, bg_alpha: Optional[float], background: bool,
    bg_color: str, label_cats: bool, label_fontsize: float, label_fontweight: str,
    label_outline: bool, label_arrow: bool, label_offset: tuple,
) -> None:
    """Draw one categorical panel: optional grey context layer + colored scatter."""
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
 
 
def _values_to_sizes(vals: np.ndarray, size_scale: bool, dot_size: float,
                      size_range: tuple, vmin: float, vmax: float) -> np.ndarray:
    """Per-point marker sizes: scalar `dot_size` if size_scale=False, else
    linearly scaled into `size_range` based on value magnitude."""
    if not size_scale:
        return np.full(vals.shape, dot_size)
    lo, hi = size_range
    if vmax > vmin:
        norm = (np.clip(vals, vmin, vmax) - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(vals)
    return lo + norm * (hi - lo)
 
 
def _draw_panel_cont(
    ax: Axes, udf: pd.DataFrame, col: str, dot_size: float, size_scale: bool,
    size_range: tuple, alpha: float, cmap: str, na_color: str,
    na_alpha: Optional[float], zero_color: Optional[str],
    zero_alpha: Optional[float], vmin: float, vmax: float,
    background: bool, bg_color: str, bg_alpha: Optional[float],
):
    """Draw one continuous panel.
 
    Z-order: grey background -> NaN points (na_color) -> all finite points
    colored by cmap over [vmin, vmax] (range computed over ALL finite values,
    including exact zeros) -> if zero_color is set, exact-zero points are
    redrawn on top in a flat color, overriding their cmap color without
    affecting the scale/colorbar.
 
    Returns the ScalarMappable from the cmap-colored layer (for colorbar
    attachment), or None if there were no finite points to plot.
    """
    if background:
        ax.scatter(udf["u1"], udf["u2"], s=dot_size, c=bg_color,
                   alpha=bg_alpha, linewidths=0, rasterized=True)
 
    vals = udf[col].values.astype(float)
    nan_mask = np.isnan(vals)
 
    if nan_mask.any():
        nan_size = dot_size if not size_scale else size_range[0]
        ax.scatter(udf.loc[nan_mask, "u1"], udf.loc[nan_mask, "u2"],
                   s=nan_size, c=na_color,
                   alpha=na_alpha if na_alpha is not None else alpha,
                   linewidths=0, rasterized=True)
 
    sc = None
    if (~nan_mask).any():
        good = udf.loc[~nan_mask]
        good_vals = vals[~nan_mask]
        sizes = _values_to_sizes(good_vals, size_scale, dot_size, size_range, vmin, vmax)
        sc = ax.scatter(good["u1"], good["u2"], s=sizes, c=good_vals,
                        cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                        linewidths=0, rasterized=True)
 
    if zero_color is not None:
        zero_mask = (~nan_mask) & (vals == 0)
        if zero_mask.any():
            zero_size = dot_size if not size_scale else size_range[0]
            ax.scatter(udf.loc[zero_mask, "u1"], udf.loc[zero_mask, "u2"],
                       s=zero_size, c=zero_color,
                       alpha=zero_alpha if zero_alpha is not None else alpha,
                       linewidths=0, rasterized=True)
 
    return sc
 
 
# ── public API ──────────────────────────────────────────────────────────────
 
def plot_embedding(
    adata,
    col: str,
    *,
    kind: Literal["auto", "categorical", "continuous"] = "auto",
    obsm_key: str = "X_umap",
    conditions: Optional[list] = None,
    split: bool = False,
    ax: Optional[Axes] = None,          # only used when split=False
    ncol: int = 3,
    blank_pos: Optional[int] = None,
    dot_size: float = 2.0,
    alpha: float = 0.75,
    bg_color: str = "silver",           # split mode only
    bg_alpha: Optional[float] = None,   # split mode only; auto-scaled if None
    text_size: float = 6,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    axis_show: bool = True,
    spines: bool = False,
    panel_width: float = 2.88,
    panel_height: float = 2.75,
    figsize: Optional[tuple] = None,    # single-plot mode override
    dpi: int = 320,
    fig_dir: Optional[str] = None,
    # -- categorical-only --
    color_map: Optional[Union[dict, str]] = None,
    palette: str = "tab20",
    legend: bool = True,
    legend_title: Optional[str] = None,
    legend_markerscale: float = 5,
    legend_order: Optional[Union[list, str]] = None,
    label_cats: bool = False,
    label_fontsize: Optional[float] = None,
    label_fontweight: str = "bold",
    label_outline: bool = True,
    label_arrow: bool = False,
    label_offset: tuple = (0.5, 0.5),
    # -- continuous-only --
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size_scale: bool = False,
    size_range: tuple = (1.0, 30.0),
    na_color: str = "lightgrey",
    na_alpha: Optional[float] = None,
    zero_color: Optional[str] = None,
    zero_alpha: Optional[float] = None,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    cbar_extend: Literal["neither", "min", "max", "both"] = "neither",
) -> Union[Axes, Figure]:
    """Plot a 2D embedding (UMAP by default, any `adata.obsm` key) colored by
    an obs column, as a single panel or a per-category facet grid.
 
    `col` dtype (or `kind`) decides which coloring mode is used:
      - categorical: discrete `palette`/`color_map`, legend with one handle
        per category.
      - continuous: `cmap`-colored scale over [vmin, vmax] (computed over
        all finite values), NaN cells drawn in `na_color`, and optionally
        exact-zero cells flagged with `zero_color`, drawn on top so they
        stay visually distinct from NaN and from the cmap-colored range.
 
    Args:
        adata: AnnData with `obsm[obsm_key]` (2D embedding) and `obs[col]`.
        col: Column in `adata.obs` used for coloring and (if split=True)
            categorical faceting.
        kind: 'auto' detects categorical vs continuous from `adata.obs[col]`
            dtype (numeric -> continuous). Force with 'categorical' or
            'continuous' if auto-detection picks the wrong branch.
        obsm_key: Key in `adata.obsm` holding the 2D embedding, e.g.
            'X_umap' or 'X_pca'.
        conditions: Categorical mode only: ordered list of categories to
            plot (defaults to sorted unique values). Not used in continuous
            mode — see `split` below for why continuous has no faceting.
        split: If False, single panel. If True, one facet per category with
            a grey full-dataset background — categorical mode only, since
            faceting facets *by* `col`'s own discrete values, which doesn't
            exist for a continuous `col`. To facet continuous coloring by
            a separate categorical column, call in a loop over that
            column's categories instead (not built into this API).
        ax: Existing Axes to draw into. Only honored when split=False.
        ncol: Number of columns in the split grid.
        blank_pos: Zero-based slot index to insert an empty/hidden panel.
        dot_size: Scatter marker size (matplotlib `s=`), or the fixed
            reference size for NaN/zero points when `size_scale=True`.
        alpha: Foreground marker opacity.
        bg_color: Background layer color in split mode.
        bg_alpha: Background layer opacity in split mode; auto-scaled as
            `min(0.3, 5000 / n_cells)` if None.
        text_size: Base font size for axis labels/legend/colorbar/title.
        title: Panel/figure title. Defaults to `col`.
        title_fontsize: Title font size. Falls back to `text_size`.
        xlabel, ylabel: Axis label text.
        axis_show: If True, draw axis labels.
        spines: If True, show axes spines/frame.
        panel_width, panel_height: Per-panel size in inches (split mode).
        figsize: Figure size override for single-panel mode.
        dpi: Figure resolution.
        fig_dir: If provided, saves the figure to this path (.png/.pdf/.svg;
            defaults to .png if no matching suffix given).
        color_map: Categorical only. {category: color} mapping, or a single
            color string broadcast to all categories. Auto-generated from
            `palette` if None.
        palette: Categorical only. Seaborn palette name for auto color_map.
        legend: Categorical only. If True, draw a legend (single-panel mode).
        legend_title: Categorical only. Legend title. Defaults to `col`.
        legend_markerscale: Categorical only. Legend marker size multiplier.
        legend_order: Categorical only. 'auto' sorts categories numerically,
            an explicit list sets order, None uses `conditions`/discovery.
        label_cats: Categorical only. Annotate each category's centroid.
        label_fontsize: Categorical only. Falls back to `text_size`.
        label_fontweight, label_outline, label_arrow, label_offset:
            Categorical only. Centroid label styling.
        cmap: Continuous only. Colormap name for the value scale.
        vmin, vmax: Continuous only. Color scale bounds; default to
            min/max of all finite `col` values (including zero).
        size_scale: Continuous only. If True, scale marker size by value.
        size_range: Continuous only. (min, max) sizes when size_scale=True.
        na_color: Continuous only. Color for NaN cells.
        na_alpha: Continuous only. Opacity for NaN cells; falls back to `alpha`.
        zero_color: Continuous only. If set, cells with exact value 0 are
            redrawn on top in this flat color (distinct from NaN and from
            the cmap range), without affecting vmin/vmax. If None (default),
            zero is treated like any other value on the cmap scale.
        zero_alpha: Continuous only. Opacity for zero cells; falls back to `alpha`.
        colorbar: Continuous only. If True, draw a colorbar.
        cbar_label: Continuous only. Colorbar label. Defaults to `col`.
        cbar_extend: Continuous only. 'max'/'min'/'both' draws a triangular
            arrow cap on the colorbar and relabels that end's tick as
            '>=vmax' / '<=vmin' — use when values are clamped outside
            [vmin, vmax] (e.g. vmax=5 with real values up to 50) so the
            legend honestly shows the range includes an open end rather
            than implying vmax is a true data maximum. 'neither' (default)
            draws a plain colorbar with no arrow caps.
 
    Returns:
        matplotlib.axes.Axes if split=False, matplotlib.figure.Figure if split=True.
 
    Raises:
        KeyError: If `obsm_key`/`col` not found in adata.
        ValueError: If split=True is requested with kind='continuous'
            (no separate facet column available — see `split` above).
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm")
    if col not in adata.obs:
        raise KeyError(f"'{col}' not found in adata.obs")
 
    is_cont = (
        kind == "continuous"
        or (kind == "auto" and pd.api.types.is_numeric_dtype(adata.obs[col]))
    )
    if is_cont and split:
        raise ValueError(
            "split=True is only supported for categorical `col`. "
            "Continuous mode has no separate facet column in this API."
        )
 
    udf = pd.DataFrame(adata.obsm[obsm_key][:, :2], columns=["u1", "u2"],
                       index=adata.obs.index)
 
    fs_label = label_fontsize if label_fontsize is not None else text_size
    legend_title = legend_title if legend_title is not None else col
 
    # ════════════════════════════════════════════════════════ categorical ═══
    if not is_cont:
        udf[col] = adata.obs[col].values
        cats = conditions or _sort_categories(udf[col].unique())
 
        if color_map is None:
            colors = sns.color_palette(palette, len(cats))
            color_map = dict(zip(cats, colors))
        elif isinstance(color_map, str):
            color_map = {cat: color_map for cat in cats}
 
        # ---- single panel ----
        if not split:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize or (6, 5), dpi=dpi)
 
            _draw_panel_cat(ax, udf, col, cats, color_map, dot_size, alpha, None,
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
                         bbox_to_anchor=(1, 1), loc="upper left", frameon=False,
                         title=legend_title, title_fontsize=text_size)
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
                _draw_panel_cat(a, udf, col, [cat], color_map, dot_size, alpha, bg_alpha,
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
 
    # ════════════════════════════════════════════════════════ continuous ═══
    else:
        udf[col] = pd.to_numeric(adata.obs[col], errors="coerce").values
 
        vals = udf[col].values.astype(float)
        finite_vals = vals[~np.isnan(vals)]
        _vmin = vmin if vmin is not None else (float(np.min(finite_vals)) if finite_vals.size else 0.0)
        _vmax = vmax if vmax is not None else (float(np.max(finite_vals)) if finite_vals.size else 1.0)
 
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (6, 5), dpi=dpi)
 
        sc = _draw_panel_cont(ax, udf, col, dot_size, size_scale, size_range,
                              alpha, cmap, na_color, na_alpha, zero_color, zero_alpha,
                              _vmin, _vmax, background=False, bg_color=bg_color, bg_alpha=None)
 
        ax.set_xticks([]); ax.set_yticks([])
        if axis_show:
            ax.set_xlabel(xlabel, size=text_size)
            ax.set_ylabel(ylabel, size=text_size)
        ax.set_title(title or col, size=title_fontsize or text_size)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(spines)
        if colorbar and sc is not None:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                                    extend=cbar_extend)
            cb.set_label(cbar_label or col, fontsize=text_size)
            cb.ax.tick_params(labelsize=text_size)
            if cbar_extend in ("max", "min", "both"):
                ticks = cb.get_ticks()
                labels = []
                for t in ticks:
                    if cbar_extend in ("max", "both") and np.isclose(t, _vmax):
                        labels.append(f"\u2265{_vmax:g}")
                    elif cbar_extend in ("min", "both") and np.isclose(t, _vmin):
                        labels.append(f"\u2264{_vmin:g}")
                    else:
                        labels.append(f"{t:g}")
                cb.set_ticks(ticks)
                cb.set_ticklabels(labels)
        fig = ax.figure
        result = ax
 
    if fig_dir:
        save_path = Path(fig_dir)
        if save_path.suffix.lower() not in (".png", ".pdf", ".svg"):
            save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
 
    return result
 

# --------------------------------------------------

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
    fig_dir=None,  
    dpi=320,# e.g. "umap_Pou5f1.pdf"
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

    if fig_dir:
        save_path = Path(fig_dir)
        if save_path.suffix.lower() not in (".png", ".pdf", ".svg"):
            save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

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
