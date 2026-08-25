
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

def plot_umap_cont(
    adata,
    col: str,
    obsm_key: str = "X_umap",
    split: bool = False,
    split_by: Optional[str] = None,     # categorical column to facet by (split mode)
    conditions: Optional[list] = None,  # ordered categories of split_by
    ax: Optional[Axes] = None,          # only used when split=False
    ncol: int = 3,
    blank_pos: Optional[int] = None,
    dot_size: float = 2.0,
    size_scale: bool = False,           # scale dot size by value magnitude
    size_range: tuple = (1.0, 30.0),    # (min, max) marker size when size_scale=True
    alpha: float = 0.75,
    cmap: str = "viridis",
    na_color: str = "lightgrey",
    na_alpha: Optional[float] = None,   # falls back to `alpha` if None
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    bg_alpha: Optional[float] = None,   # split mode only; auto-scaled if None
    bg_color: str = "silver",           # split mode only
    text_size: float = 6,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    axis_show: bool = True,
    spines: bool = False,
    panel_width: float = 2.88,
    panel_height: float = 2.75,
    figsize: Optional[tuple] = None,    # single-plot mode override
    dpi: int = 320,
    fig_dir: Optional[str] = None,
) -> Union[Axes, Figure]:
    """Plot UMAP colored by a continuous obs column, as a single panel or a
    per-category grid faceted by a separate categorical column.

    Args:
        adata: AnnData object with `obsm[obsm_key]` (2D embedding) and
            `obs[col]` (numeric column).
        col: Continuous column in `adata.obs` used for coloring. NaN values
            are drawn in `na_color`.
        obsm_key: Key in `adata.obsm` holding the 2D embedding.
        split: If False, single panel colored continuously by `col`. If True,
            facet by `split_by`, each panel with a grey full-dataset
            background for context, colored points on top by `col`.
        split_by: Categorical column in `adata.obs` used to facet when
            split=True. Required if split=True.
        conditions: Ordered list of `split_by` categories to plot. Defaults
            to `sorted(adata.obs[split_by].unique())`.
        ax: Existing Axes to draw into. Only honored when `split=False`.
        ncol: Number of columns in the split grid.
        blank_pos: Zero-based slot index at which to insert an empty panel.
        dot_size: Base scatter marker size (matplotlib `s=`); acts as the
            fixed size when `size_scale=False`, or the minimum reference
            size for NaN points when `size_scale=True`.
        size_scale: If True, scale marker size by the (normalized) value of
            `col`, between `size_range`. Default False (fixed `dot_size`).
        size_range: (min, max) marker sizes used when `size_scale=True`.
        alpha: Foreground marker opacity for colored (non-NaN) points.
        cmap: Matplotlib/seaborn colormap name used for continuous coloring.
        na_color: Color used for NaN values.
        na_alpha: Opacity for NaN points. Falls back to `alpha` if None.
        vmin, vmax: Color scale bounds. Default to min/max of non-NaN `col`.
        bg_alpha: Background layer opacity in split mode. If None, auto-scaled
            as `min(0.3, 5000 / n_cells)`.
        bg_color: Background layer color in split mode.
        text_size: Base font size for axis labels/colorbar/title fallback.
        title: Panel/figure title. Defaults to `col` if None (single-panel).
        title_fontsize: Font size for titles. Falls back to `text_size`.
        colorbar: If True, draw a colorbar (single-panel: attached to axes;
            split mode: one shared colorbar for the figure).
        cbar_label: Label for the colorbar. Defaults to `col`.
        xlabel: X-axis label text.
        ylabel: Y-axis label text.
        axis_show: If True, draw axis labels.
        spines: If True, show axes spines/frame.
        panel_width: Per-panel width in inches (split mode).
        panel_height: Per-panel height in inches (split mode).
        figsize: Figure size override for single-panel mode.
        dpi: Figure resolution.
        fig_dir: If provided, saves the figure as PNG to `{fig_dir}/umap_{col}.png`.

    Returns:
        matplotlib.axes.Axes if split=False, matplotlib.figure.Figure if split=True.

    Raises:
        KeyError: If `obsm_key`/`col`/`split_by` not found in adata.
        ValueError: If split=True and `split_by` is not provided.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm")
    if col not in adata.obs:
        raise KeyError(f"'{col}' not found in adata.obs")
    if split and split_by is None:
        raise ValueError("split=True requires `split_by` (a categorical obs column)")
    if split and split_by not in adata.obs:
        raise KeyError(f"'{split_by}' not found in adata.obs")

    udf = pd.DataFrame(adata.obsm[obsm_key][:, :2], columns=["u1", "u2"],
                       index=adata.obs.index)
    udf[col] = pd.to_numeric(adata.obs[col], errors="coerce").values
    if split:
        udf[split_by] = adata.obs[split_by].values

    vals = udf[col].values.astype(float)
    finite_vals = vals[~np.isnan(vals)]
    if vmin is None:
        vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    if vmax is None:
        vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0

    # ---- single panel ----
    if not split:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (6, 5), dpi=dpi)

        sc = _draw_panel_cont(ax, udf, col, dot_size, size_scale, size_range,
                              alpha, cmap, na_color, na_alpha, vmin, vmax,
                              background=False, bg_color=bg_color, bg_alpha=None)

        ax.set_xticks([]); ax.set_yticks([])
        if axis_show:
            ax.set_xlabel(xlabel, size=text_size)
            ax.set_ylabel(ylabel, size=text_size)
        ax.set_title(title or col, size=title_fontsize or text_size)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(spines)
        if colorbar and sc is not None:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cbar_label or col, fontsize=text_size)
            cb.ax.tick_params(labelsize=text_size)
        fig = ax.figure
        result = ax

    # ---- split grid ----
    else:
        if bg_alpha is None:
            bg_alpha = float(min(0.3, 5_000 / max(len(udf), 1)))
        cats = conditions or sorted(udf[split_by].unique())
        slots = list(cats)
        if blank_pos is not None:
            slots.insert(max(0, min(blank_pos, len(slots))), None)
        nrow = math.ceil(len(slots) / ncol)
        fig, axs = plt.subplots(nrow, ncol,
                                figsize=(panel_width * ncol, panel_height * nrow),
                                dpi=dpi)
        axs_flat = list(axs.flat) if hasattr(axs, "flat") else [axs]
        last_sc = None
        for i, a in enumerate(axs_flat):
            if i >= len(slots) or slots[i] is None:
                a.set_visible(False)
                continue
            cat = slots[i]
            sub = udf[udf[split_by] == cat]
            sc = _draw_panel_cont(a, sub, col, dot_size, size_scale, size_range,
                                  alpha, cmap, na_color, na_alpha, vmin, vmax,
                                  background=True, bg_color=bg_color,
                                  bg_alpha=bg_alpha)
            if sc is not None:
                last_sc = sc
            a.set_xticks([]); a.set_yticks([])
            a.set_title(cat, fontsize=title_fontsize or text_size)
            a.set_aspect("equal")
            a.set_frame_on(spines)
        if axis_show:
            fig.supxlabel(xlabel, fontsize=text_size)
            fig.supylabel(ylabel, fontsize=text_size)
        if colorbar and last_sc is not None:
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=axs_flat, fraction=0.02, pad=0.02)
            cb.set_label(cbar_label or col, fontsize=text_size)
            cb.ax.tick_params(labelsize=text_size)
        else:
            plt.tight_layout()
        result = fig
        plt.close(fig)

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



# --------------------------------------------------
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def _values_to_sizes(vals: np.ndarray, size_scale: bool, dot_size: float,
                      size_range: tuple, vmin: float, vmax: float) -> np.ndarray:
    """Return per-point marker sizes. Scalar `dot_size` if size_scale=False,
    else linearly scaled into `size_range` based on value magnitude."""
    if not size_scale:
        return np.full(vals.shape, dot_size)
    lo, hi = size_range
    if vmax > vmin:
        norm = (np.clip(vals, vmin, vmax) - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(vals)
    return lo + norm * (hi - lo)


def _draw_panel_cont(ax, udf, col, dot_size, size_scale, size_range, alpha,
                      cmap, na_color, na_alpha, vmin, vmax,
                      background, bg_color, bg_alpha):
    """Draw one continuous-color panel: optional grey full-dataset background,
    then NaN points in `na_color`, then valid points colored by `col`."""
    if background:
        ax.scatter(udf["u1"], udf["u2"], s=dot_size, c=bg_color,
                   alpha=bg_alpha, linewidths=0)

    vals = udf[col].values.astype(float)
    nan_mask = np.isnan(vals)

    # NaN points, plotted first so colored points sit on top
    if nan_mask.any():
        nan_size = dot_size if not size_scale else size_range[0]
        ax.scatter(udf.loc[nan_mask, "u1"], udf.loc[nan_mask, "u2"],
                   s=nan_size, c=na_color,
                   alpha=na_alpha if na_alpha is not None else alpha,
                   linewidths=0)

    if (~nan_mask).any():
        good = udf.loc[~nan_mask]
        good_vals = vals[~nan_mask]
        sizes = _values_to_sizes(good_vals, size_scale, dot_size, size_range,
                                 vmin, vmax)
        sc = ax.scatter(good["u1"], good["u2"], s=sizes, c=good_vals,
                        cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                        linewidths=0)
        return sc
    return None


def plot_umap_cont(
    adata,
    col: str,
    obsm_key: str = "X_umap",
    split: bool = False,
    split_by: Optional[str] = None,     # categorical column to facet by (split mode)
    conditions: Optional[list] = None,  # ordered categories of split_by
    ax: Optional[Axes] = None,          # only used when split=False
    ncol: int = 3,
    blank_pos: Optional[int] = None,
    dot_size: float = 2.0,
    size_scale: bool = False,           # scale dot size by value magnitude
    size_range: tuple = (1.0, 30.0),    # (min, max) marker size when size_scale=True
    alpha: float = 0.75,
    cmap: str = "viridis",
    na_color: str = "lightgrey",
    na_alpha: Optional[float] = None,   # falls back to `alpha` if None
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    bg_alpha: Optional[float] = None,   # split mode only; auto-scaled if None
    bg_color: str = "silver",           # split mode only
    text_size: float = 6,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    colorbar: bool = True,
    cbar_label: Optional[str] = None,
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    axis_show: bool = True,
    spines: bool = False,
    panel_width: float = 2.88,
    panel_height: float = 2.75,
    figsize: Optional[tuple] = None,    # single-plot mode override
    dpi: int = 320,
    fig_dir: Optional[str] = None,
) -> Union[Axes, Figure]:
    """Plot UMAP colored by a continuous obs column, as a single panel or a
    per-category grid faceted by a separate categorical column.

    Args:
        adata: AnnData object with `obsm[obsm_key]` (2D embedding) and
            `obs[col]` (numeric column).
        col: Continuous column in `adata.obs` used for coloring. NaN values
            are drawn in `na_color`.
        obsm_key: Key in `adata.obsm` holding the 2D embedding.
        split: If False, single panel colored continuously by `col`. If True,
            facet by `split_by`, each panel with a grey full-dataset
            background for context, colored points on top by `col`.
        split_by: Categorical column in `adata.obs` used to facet when
            split=True. Required if split=True.
        conditions: Ordered list of `split_by` categories to plot. Defaults
            to `sorted(adata.obs[split_by].unique())`.
        ax: Existing Axes to draw into. Only honored when `split=False`.
        ncol: Number of columns in the split grid.
        blank_pos: Zero-based slot index at which to insert an empty panel.
        dot_size: Base scatter marker size (matplotlib `s=`); acts as the
            fixed size when `size_scale=False`, or the minimum reference
            size for NaN points when `size_scale=True`.
        size_scale: If True, scale marker size by the (normalized) value of
            `col`, between `size_range`. Default False (fixed `dot_size`).
        size_range: (min, max) marker sizes used when `size_scale=True`.
        alpha: Foreground marker opacity for colored (non-NaN) points.
        cmap: Matplotlib/seaborn colormap name used for continuous coloring.
        na_color: Color used for NaN values.
        na_alpha: Opacity for NaN points. Falls back to `alpha` if None.
        vmin, vmax: Color scale bounds. Default to min/max of non-NaN `col`.
        bg_alpha: Background layer opacity in split mode. If None, auto-scaled
            as `min(0.3, 5000 / n_cells)`.
        bg_color: Background layer color in split mode.
        text_size: Base font size for axis labels/colorbar/title fallback.
        title: Panel/figure title. Defaults to `col` if None (single-panel).
        title_fontsize: Font size for titles. Falls back to `text_size`.
        colorbar: If True, draw a colorbar (single-panel: attached to axes;
            split mode: one shared colorbar for the figure).
        cbar_label: Label for the colorbar. Defaults to `col`.
        xlabel: X-axis label text.
        ylabel: Y-axis label text.
        axis_show: If True, draw axis labels.
        spines: If True, show axes spines/frame.
        panel_width: Per-panel width in inches (split mode).
        panel_height: Per-panel height in inches (split mode).
        figsize: Figure size override for single-panel mode.
        dpi: Figure resolution.
        fig_dir: If provided, saves the figure as PNG to `{fig_dir}/umap_{col}.png`.

    Returns:
        matplotlib.axes.Axes if split=False, matplotlib.figure.Figure if split=True.

    Raises:
        KeyError: If `obsm_key`/`col`/`split_by` not found in adata.
        ValueError: If split=True and `split_by` is not provided.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm")
    if col not in adata.obs:
        raise KeyError(f"'{col}' not found in adata.obs")
    if split and split_by is None:
        raise ValueError("split=True requires `split_by` (a categorical obs column)")
    if split and split_by not in adata.obs:
        raise KeyError(f"'{split_by}' not found in adata.obs")

    udf = pd.DataFrame(adata.obsm[obsm_key][:, :2], columns=["u1", "u2"],
                       index=adata.obs.index)
    udf[col] = pd.to_numeric(adata.obs[col], errors="coerce").values
    if split:
        udf[split_by] = adata.obs[split_by].values

    vals = udf[col].values.astype(float)
    finite_vals = vals[~np.isnan(vals)]
    if vmin is None:
        vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    if vmax is None:
        vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0

    # ---- single panel ----
    if not split:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (6, 5), dpi=dpi)

        sc = _draw_panel_cont(ax, udf, col, dot_size, size_scale, size_range,
                              alpha, cmap, na_color, na_alpha, vmin, vmax,
                              background=False, bg_color=bg_color, bg_alpha=None)

        ax.set_xticks([]); ax.set_yticks([])
        if axis_show:
            ax.set_xlabel(xlabel, size=text_size)
            ax.set_ylabel(ylabel, size=text_size)
        ax.set_title(title or col, size=title_fontsize or text_size)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(spines)
        if colorbar and sc is not None:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cbar_label or col, fontsize=text_size)
            cb.ax.tick_params(labelsize=text_size)
        fig = ax.figure
        result = ax

    # ---- split grid ----
    else:
        if bg_alpha is None:
            bg_alpha = float(min(0.3, 5_000 / max(len(udf), 1)))
        cats = conditions or sorted(udf[split_by].unique())
        slots = list(cats)
        if blank_pos is not None:
            slots.insert(max(0, min(blank_pos, len(slots))), None)
        nrow = math.ceil(len(slots) / ncol)
        fig, axs = plt.subplots(nrow, ncol,
                                figsize=(panel_width * ncol, panel_height * nrow),
                                dpi=dpi)
        axs_flat = list(axs.flat) if hasattr(axs, "flat") else [axs]
        last_sc = None
        for i, a in enumerate(axs_flat):
            if i >= len(slots) or slots[i] is None:
                a.set_visible(False)
                continue
            cat = slots[i]
            sub = udf[udf[split_by] == cat]
            sc = _draw_panel_cont(a, sub, col, dot_size, size_scale, size_range,
                                  alpha, cmap, na_color, na_alpha, vmin, vmax,
                                  background=True, bg_color=bg_color,
                                  bg_alpha=bg_alpha)
            if sc is not None:
                last_sc = sc
            a.set_xticks([]); a.set_yticks([])
            a.set_title(cat, fontsize=title_fontsize or text_size)
            a.set_aspect("equal")
            a.set_frame_on(spines)
        if axis_show:
            fig.supxlabel(xlabel, fontsize=text_size)
            fig.supylabel(ylabel, fontsize=text_size)
        if colorbar and last_sc is not None:
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=axs_flat, fraction=0.02, pad=0.02)
            cb.set_label(cbar_label or col, fontsize=text_size)
            cb.ax.tick_params(labelsize=text_size)
        else:
            plt.tight_layout()
        result = fig
        plt.close(fig)

    if fig_dir:
        save_path = Path(fig_dir)
        if save_path.suffix.lower() not in (".png", ".pdf", ".svg"):
            save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return result
\
    
    
    
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

