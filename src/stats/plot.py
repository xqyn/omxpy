import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pathlib import Path


# plot.py lives at <root>/src/stats/plot.py
# .parent        → src/stats/
# .parent.parent → src/
# .parent x 3   → <root>/          (where config/ lives)
_DEFAULT_COLORS_PATH = (
    Path(__file__).parent.parent.parent / "config" / "color" / "colors.yaml"
)


def _load_colors(n: int, path: Path = _DEFAULT_COLORS_PATH) -> list[str]:
    """Load the first *n* colors from a YAML palette file.

    Args:
        n: Number of colors required.
        path: Path to the YAML file containing a top-level ``colors`` list.

    Returns:
        List of *n* hex color strings.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the palette contains fewer entries than *n*.
        KeyError: If the YAML file has no ``colors`` key.
    """
    if not path.exists():
        raise FileNotFoundError(f"Color palette not found: {path}")

    with path.open() as fh:
        palette: list[str] = yaml.safe_load(fh)["colors"]

    if len(palette) < n:
        raise ValueError(
            f"Palette at '{path}' has {len(palette)} colors, "
            f"but {n} are required for value_cols."
        )

    return palette[:n]


def _draw_stat_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_top: float,
    label: str,
    bracket_height_rel: float = 0.03,
    fontsize: int = 9,
    lw: float = 1.0,
) -> float:
    """Draw a single significance bracket between two x positions.

    Args:
        ax: Matplotlib axes to draw on.
        x1: x-coordinate of the left bar.
        x2: x-coordinate of the right bar.
        y_top: Bottom of the bracket (tip of the vertical ticks).
        label: Significance label, e.g. ``"ns"``, ``"*"``, ``"**"``,
            ``"***"``, ``"****"``.
        bracket_height_rel: Height of the vertical tick as a fraction
            of the current y-range.
        fontsize: Font size of the significance label.
        lw: Line width of the bracket.

    Returns:
        y-coordinate of the top of the drawn bracket, so the caller
        can stack the next bracket above it.
    """
    y_min, y_max = ax.get_ylim()
    h = (y_max - y_min) * bracket_height_rel

    ax.plot(
        [x1, x1, x2, x2],
        [y_top, y_top + h, y_top + h, y_top],
        lw=lw,
        c="black",
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y_top + h,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
    )

    return y_top + h


def plot_bar_group(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str] | None = None,
    color_map: list[str] | None = None,
    group_order: list[str] | None = None,
    exclude: dict | None = None,
    fig_size: tuple[float, float] = (5.75, 5.0),
    bar_width: float = 0.25,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend_kwargs: dict | None = None,
    save_path: str | None = None,
    dpi: int = 120,
    fontsize: int = 15,
    alpha_grid: float = 0.15,
    grid_linewidth: float = 1.0,
    ylim: tuple[float, float] | None = None,
    y_breaks: list[tuple[float, float]] | None = None,
    y_break_ratios: list[float] | None = None,
    jitter: bool = False,
    jitter_std: float = 0.03,
    jitter_seed: int = 920,
    sig_annotations: list[dict] | None = None,
    sig_bracket_gap: float = 0.05,
    sig_bracket_height: float = 0.03,
    sig_fontsize: int | None = None,
    sig_linewidth: float = 1.0,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Plot a grouped bar chart with error bars (range / 2) and optional jitter.

    Error bars represent half the observed range (max - min) / 2 per group.
    Individual data points can be overlaid as jittered scatter points.
    A broken y-axis can be requested via ``y_breaks`` to reveal clusters of
    values that would otherwise be compressed by outliers.
    Significance brackets can be drawn between any two groups via
    ``sig_annotations``.

    Args:
        df: Input dataframe containing at least ``group_col`` and all
            ``value_cols``.
        group_col: Column name used to define x-axis categories (e.g.
            ``"sample"``).
        value_cols: Column names to render as grouped bars within each
            category (e.g. ``["Oct4", "Sox2", "Nanog"]``). When ``None``,
            all numeric columns except ``group_col`` are used.
        color_map: Fill colours for each bar group, one entry per element in
            ``value_cols``. When ``None``, the first ``len(value_cols)``
            colours are loaded automatically from
            ``<root>/config/color/colors.yaml``.
        group_order: Explicit ordering of ``group_col`` values along the
            x-axis. When ``None``, groups appear in the order they are first
            encountered in ``df`` after any filtering.
        exclude: Row-level filters applied before aggregation. Each key is a
            column name and the corresponding value is the label to drop
            (equality match). Example: ``{"rep": "r2"}`` removes all rows
            where ``rep == "r2"``.
        fig_size: ``(width, height)`` of the figure in inches.
        bar_width: Width of a single bar in data-axis units.
        xlabel: X-axis label. Defaults to ``group_col`` when ``None``.
        ylabel: Y-axis label applied to the figure's centre. Omitted when
            ``None``.
        title: Axes title and, when ``save_path`` is provided, the filename
            suffix used when saving. No title is set when ``None``.
        legend_kwargs: Keyword arguments forwarded to ``ax.legend()``. Merged
            on top of the built-in defaults; supply to override individual
            keys only.
        save_path: Directory path or filename prefix for saving the figure.
            The file is written to ``{save_path}_{title}.png``. Skipped when
            ``None`` or when ``title`` is ``None``.
        dpi: Dots-per-inch resolution for the saved figure.
        fontsize: Font size applied to x/y labels, tick labels, title, and
            legend text.
        alpha_grid: Opacity of the horizontal grid lines.
        grid_linewidth: Line width of the horizontal grid lines.
        ylim: ``(ymin, ymax)`` limits for the y-axis. Ignored when
            ``y_breaks`` is provided. When ``None`` Matplotlib auto-scales.
        y_breaks: List of ``(ymin, ymax)`` tuples defining contiguous
            visible segments of the y-axis, ordered from **bottom to top**.
            A gap with diagonal break marks is rendered between each adjacent
            pair of segments. Mutually exclusive with ``ylim``; if both are
            supplied ``y_breaks`` takes precedence.
        y_break_ratios: Relative height ratios for each broken-axis panel,
            one entry per element in ``y_breaks``, ordered bottom to top.
            When ``None``, ratios are derived from the inverse of each
            segment span so that denser ranges receive more vertical space.
        jitter: Whether to overlay individual data points as scatter markers.
            When ``False`` (default) no points are drawn.
        jitter_std: Standard deviation of the Gaussian noise added to each
            point's x-position. Only used when ``jitter=True``.
        jitter_seed: Random seed for reproducible jitter offsets. Only used
            when ``jitter=True``.
        sig_annotations: List of significance bracket descriptors. Each entry
            is a dict with keys:

            - ``group1`` (str): Left-hand group name.
            - ``group2`` (str): Right-hand group name.
            - ``marker`` (str): Which ``value_cols`` bar to anchor on.
            - ``label`` (str): Significance label e.g. ``"*"``, ``"**"``,
              ``"***"``, ``"****"``, ``"ns"``.
            - ``y`` (float, optional): Manual y-position for the bracket
              base. When omitted, placed automatically above the tallest
              bar involved.

            Example::

                sig_annotations=[
                    {"group1": "WT", "group2": "C6",
                     "marker": "Oct4", "label": "**"},
                    {"group1": "WT", "group2": "C10",
                     "marker": "Oct4", "label": "***"},
                ]

        sig_bracket_gap: Vertical gap between bar top and bracket base,
            as a fraction of the y-axis span.
        sig_bracket_height: Height of the vertical bracket ticks, as a
            fraction of the y-axis span. Forwarded to
            ``_draw_stat_bracket``.
        sig_fontsize: Font size for significance labels. Defaults to
            ``fontsize`` when ``None``.
        sig_linewidth: Line width of the bracket lines.

    Returns:
        ``(fig, ax)`` when ``y_breaks`` is ``None``, or ``(fig, axes)`` where
        ``axes`` is a list ordered bottom-to-top when ``y_breaks`` is set.

    Raises:
        ValueError: If ``color_map`` does not have the same length as
            ``value_cols``, if any column in ``value_cols`` is absent from
            ``df``, if no numeric columns are found when ``value_cols``
            is ``None``, or if ``y_break_ratios`` length does not match
            ``y_breaks``.
        FileNotFoundError: If ``color_map`` is ``None`` and the default
            palette YAML cannot be found.

    Examples:
        Minimal call:

        >>> fig, ax = plot_bar_group(reporter_df, group_col="clone")

        With significance brackets:

        >>> fig, ax = plot_bar_group(
        ...     reporter_df,
        ...     group_col="clone",
        ...     value_cols=["Oct4", "Sox2", "Nanog"],
        ...     sig_annotations=[
        ...         {"group1": "WT", "group2": "C6",  "marker": "Oct4", "label": "**"},
        ...         {"group1": "WT", "group2": "C10", "marker": "Oct4", "label": "***"},
        ...     ],
        ...     ylabel="Relative expression",
        ... )
    """
    # --- value_cols inference --------------------------------------------
    if value_cols is None:
        value_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != group_col
        ]
        if not value_cols:
            raise ValueError(
                "No numeric columns found in df to infer value_cols. "
                "Pass value_cols explicitly."
            )

    # --- color_map -------------------------------------------------------
    if color_map is None:
        color_map = _load_colors(len(value_cols))
    elif len(color_map) != len(value_cols):
        raise ValueError(
            f"'color_map' length ({len(color_map)}) must match "
            f"'value_cols' length ({len(value_cols)})."
        )

    # --- Column validation -----------------------------------------------
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    # --- Filtering -------------------------------------------------------
    if exclude:
        for col, val in exclude.items():
            df = df.loc[df[col] != val]

    # --- Group order -----------------------------------------------------
    effective_order = (
        group_order
        if group_order is not None
        else list(dict.fromkeys(df[group_col]))
    )

    # --- Aggregation -----------------------------------------------------
    grouped = df.groupby(group_col)[value_cols]
    means   = grouped.mean().reindex(effective_order)
    errors  = (
        grouped.apply(lambda g: g.max() - g.min()).reindex(effective_order) / 2
    )

    # --- Layout ----------------------------------------------------------
    x       = np.arange(len(effective_order))
    n       = len(value_cols)
    offsets = (np.arange(n) - (n - 1) / 2) * bar_width

    use_breaks = y_breaks is not None and len(y_breaks) > 1
    n_panels   = len(y_breaks) if use_breaks else 1

    if use_breaks:
        if y_break_ratios is not None:
            if len(y_break_ratios) != len(y_breaks):
                raise ValueError(
                    f"'y_break_ratios' length ({len(y_break_ratios)}) must match "
                    f"'y_breaks' length ({len(y_breaks)})."
                )
            height_ratios = list(reversed(y_break_ratios))
        else:
            spans     = [ymax - ymin for ymin, ymax in y_breaks]
            inv_spans = [1.0 / s for s in spans]
            total     = sum(inv_spans)
            height_ratios = list(reversed([v / total for v in inv_spans]))

        fig, axes_array = plt.subplots(
            n_panels, 1,
            sharex=True,
            figsize=fig_size,
            gridspec_kw={"hspace": 0.05, "height_ratios": height_ratios},
            layout="constrained",
        )
        axes    = list(axes_array)
        y_ranges = list(reversed(y_breaks))

    else:
        fig, single_ax = plt.subplots(figsize=fig_size, layout="constrained")
        axes    = [single_ax]
        y_ranges = [ylim] if ylim is not None else [None]

    # --- Draw bars and jitter on every panel -----------------------------
    rng = np.random.default_rng(jitter_seed)

    for ax_i, ax in enumerate(axes):
        for col, color, offset in zip(value_cols, color_map, offsets):
            ax.bar(
                x + offset,
                means[col],
                width=bar_width,
                label=col if ax_i == 0 else "_nolegend_",
                color=color,
                edgecolor="black",
                yerr=errors[col].values,
                capsize=5,
                error_kw={
                    "elinewidth": 1,
                    "ecolor": "black",
                    "capthick": 0.5,
                    "linestyle": (0, (3, 5)),
                    "zorder": 2,
                },
            )

            if jitter:
                for j, group in enumerate(effective_order):
                    y_vals = df.loc[df[group_col] == group, col].values
                    noise  = rng.normal(0.0, jitter_std, size=len(y_vals))
                    ax.scatter(
                        np.full(len(y_vals), x[j] + offset) + noise,
                        y_vals,
                        color="black",
                        s=10,
                        alpha=0.7,
                        zorder=3,
                    )

        # --- Per-panel y-range -------------------------------------------
        if y_ranges[ax_i] is not None:
            ax.set_ylim(y_ranges[ax_i])

        # --- Spine / grid styling ----------------------------------------
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(
            True,
            linestyle="--",
            linewidth=grid_linewidth,
            alpha=alpha_grid,
        )
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=fontsize)

        if use_breaks:
            is_bottom = ax_i == n_panels - 1
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(is_bottom)
            ax.tick_params(top=False)
            if not is_bottom:
                ax.tick_params(bottom=False)

    # --- Significance brackets -------------------------------------------
    # if sig_annotations:
    #     _sig_fs = sig_fontsize if sig_fontsize is not None else fontsize - 2

    #     # Always draw on top-most panel (axes[0])
    #     sig_ax   = axes[0]
    #     y_lo, y_hi = sig_ax.get_ylim()
    #     y_span   = y_hi - y_lo

    #     # Track highest bracket per (group1, group2, marker) to auto-stack
    #     _y_cursor: dict[tuple, float] = {}

    #     for ann in sig_annotations:
    #         g1     = ann.get("group1")
    #         g2     = ann.get("group2")
    #         marker = ann.get("marker")
    #         label  = ann.get("label", "")

    #         # Skip silently if any key is invalid
    #         if g1 not in effective_order or g2 not in effective_order:
    #             continue
    #         if marker not in value_cols:
    #             continue

    #         col_idx = value_cols.index(marker)
    #         offset  = offsets[col_idx]

    #         x1 = x[effective_order.index(g1)] + offset
    #         x2 = x[effective_order.index(g2)] + offset

    #         # Auto y: top of tallest bar + error + gap
    #         bar_top = max(
    #             means.loc[g1, marker] + errors.loc[g1, marker],
    #             means.loc[g2, marker] + errors.loc[g2, marker],
    #         )

    #         stack_key = (min(g1, g2), max(g1, g2), marker)

    #         if "y" in ann:
    #             y_base = ann["y"]
    #         else:
    #             y_base = max(
    #                 _y_cursor.get(stack_key, bar_top + sig_bracket_gap * y_span),
    #                 bar_top + sig_bracket_gap * y_span,
    #             )

    #         # Draw bracket and get back the top y for stacking
    #         y_new_top = _draw_stat_bracket(
    #             ax=sig_ax,
    #             x1=x1,
    #             x2=x2,
    #             y_top=y_base,
    #             label=label,
    #             bracket_height_rel=sig_bracket_height,
    #             fontsize=_sig_fs,
    #             lw=sig_linewidth,
    #         )

    #         # Store top + small gap so next bracket stacks above
    #         _y_cursor[stack_key] = y_new_top + sig_bracket_gap * y_span

    # --- Significance brackets -------------------------------------------
    if sig_annotations:
        _sig_fs = sig_fontsize if sig_fontsize is not None else fontsize - 2

        sig_ax     = axes[0]
        y_lo, y_hi = sig_ax.get_ylim()
        y_span     = y_hi - y_lo

        # --- Helper ------------------------------------------------------
        def _bar_top(g, marker):
            return means.loc[g, marker] + errors.loc[g, marker]

        # --- Resolve x positions -----------------------------------------
        resolved = []
        for ann in sig_annotations:
            g1     = ann.get("group1")
            g2     = ann.get("group2")
            marker = ann.get("marker")
            label  = ann.get("label", "")

            if g1 not in effective_order or g2 not in effective_order:
                continue
            if marker not in value_cols:
                continue

            col_idx = value_cols.index(marker)
            offset  = offsets[col_idx]

            x1 = x[effective_order.index(g1)] + offset
            x2 = x[effective_order.index(g2)] + offset

            resolved.append({
                "g1"      : g1,
                "g2"      : g2,
                "marker"  : marker,
                "label"   : label,
                "x1"      : x1,
                "x2"      : x2,
                "y_manual": ann.get("y"),
            })

        # --- Merge consecutive brackets with same label ------------------
        merged = []
        i = 0
        while i < len(resolved):
            curr = dict(resolved[i])
            curr["spanned"] = [
                effective_order.index(curr["g1"]),
                effective_order.index(curr["g2"]),
            ]
            j = i + 1

            while j < len(resolved):
                nxt         = resolved[j]
                same_label  = nxt["label"]  == curr["label"]
                same_marker = nxt["marker"] == curr["marker"]
                curr_last   = max(curr["spanned"])
                nxt_first   = min(
                    effective_order.index(nxt["g1"]),
                    effective_order.index(nxt["g2"]),
                )
                consecutive = nxt_first == curr_last + 1

                if same_label and same_marker and consecutive:
                    curr["x2"] = nxt["x2"]
                    curr["g2"] = nxt["g2"]
                    curr["spanned"].append(effective_order.index(nxt["g2"]))
                    j += 1
                else:
                    break

            merged.append(curr)
            i = j

        # --- Global y_floor: highest bar+error across ALL annotations ----  ← NEW CHUNK STARTS
        all_annotated_groups = set()
        for ann in resolved:
            all_annotated_groups.add(ann["g1"])
            all_annotated_groups.add(ann["g2"])

        y_floor = max(
            _bar_top(g, ann["marker"])
            for ann in resolved
            for g in effective_order
            if g in all_annotated_groups
            if ann["marker"] in value_cols
        ) + sig_bracket_gap * y_span

        # --- Draw merged brackets ----------------------------------------  ← NEW CHUNK ENDS
        _y_cursor: dict[str, float] = {}

        for ann in merged:
            g1     = ann["g1"]
            g2     = ann["g2"]
            marker = ann["marker"]
            label  = ann["label"]
            x1     = ann["x1"]
            x2     = ann["x2"]

            if ann["y_manual"] is not None:
                y_base = ann["y_manual"]
            else:
                y_base = max(
                    _y_cursor.get(marker, y_floor),
                    y_floor,
                )

            y_new_top = _draw_stat_bracket(
                ax=sig_ax,
                x1=x1,
                x2=x2,
                y_top=y_base,
                label=label,
                bracket_height_rel=sig_bracket_height,
                fontsize=_sig_fs,
                lw=sig_linewidth,
            )

            prev_label = _y_cursor.get(f"{marker}_label")
            if prev_label != label:
                _y_cursor[marker]            = y_new_top + sig_bracket_gap * y_span
                _y_cursor[f"{marker}_label"] = label
            else:
                _y_cursor[marker] = y_new_top + sig_bracket_gap * y_span

    # --- Broken-axis diagonal break marks --------------------------------
    if use_breaks:
        d = 0.012
        for i in range(n_panels - 1):
            ax_top = axes[i]
            ax_bot = axes[i + 1]

            for ax_mark, y_pos in ((ax_top, 0), (ax_bot, 1)):
                kwargs_break = dict(
                    transform=ax_mark.transAxes,
                    color="black",
                    clip_on=False,
                    linewidth=0.8,
                )
                ax_mark.plot((-d, +d), (y_pos - d, y_pos + d), **kwargs_break)
                ax_mark.plot((1 - d, 1 + d), (y_pos - d, y_pos + d), **kwargs_break)

    # --- x-axis labels (bottom-most panel) -------------------------------
    bottom_ax = axes[-1] if use_breaks else axes[0]
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(effective_order, fontsize=fontsize)
    bottom_ax.set_xlabel(
        xlabel if xlabel is not None else group_col, fontsize=fontsize
    )

    # --- Title -----------------------------------------------------------
    top_ax = axes[0]
    if title:
        top_ax.set_title(title, fontsize=fontsize)

    # --- Legend ----------------------------------------------------------
    _legend_kw: dict = {
        "loc"          : "upper right",
        "fontsize"     : fontsize - 2,
        "frameon"      : False,
        "ncol"         : n,
        "bbox_to_anchor": (1, 1),
    }
    if legend_kwargs:
        _legend_kw.update(legend_kwargs)
    top_ax.legend(**_legend_kw)

    # --- Shared y-label --------------------------------------------------
    if ylabel is not None:
        if use_breaks:
            fig.text(
                0.02, 0.5, ylabel,
                va="center", ha="center",
                rotation="vertical",
                fontsize=fontsize,
            )
        else:
            axes[0].set_ylabel(ylabel, fontsize=fontsize)

    if save_path and title:
        fig.savefig(f"{save_path}_{title}.png", dpi=dpi, bbox_inches="tight")

    return (fig, axes) if use_breaks else (fig, axes[0])