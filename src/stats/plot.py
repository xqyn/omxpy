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
    jitter: bool = False,
    jitter_std: float = 0.03,
    jitter_seed: int = 920,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a grouped bar chart with error bars (range / 2) and optional jitter.

    Error bars represent half the observed range (max - min) / 2 per group.
    Individual data points can be overlaid as jittered scatter points.

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
        ylabel: Y-axis label. Omitted when ``None``.
        title: Axes title and, when ``save_path`` is provided, the filename
            suffix used when saving. No title is set when ``None``.
        legend_kwargs: Keyword arguments forwarded to ``ax.legend()``. Merged
            on top of the built-in defaults; supply to override individual
            keys only.
        save_path: Directory path or filename prefix for saving the figure.
            The file is written to ``{save_path}_{title}.png``. Nothing is
            saved when ``None``.
        dpi: Dots-per-inch resolution for the saved figure.
        fontsize: Font size applied to x/y labels, tick labels, title, and
            legend text.
        alpha_grid: Opacity of the horizontal grid lines.
        grid_linewidth: Line width of the horizontal grid lines.
        ylim: ``(ymin, ymax)`` limits for the y-axis. When ``None`` (default)
            Matplotlib auto-scales the axis.
        jitter: Whether to overlay individual data points as scatter markers.
            When ``False`` (default) no points are drawn.
        jitter_std: Standard deviation of the Gaussian noise added to each
            point's x-position. Only used when ``jitter=True``.
        jitter_seed: Random seed for reproducible jitter offsets. Only used
            when ``jitter=True``.

    Returns:
        A ``(fig, ax)`` tuple of the Matplotlib figure and axes objects,
        allowing the caller to apply further customisation after the call.

    Raises:
        ValueError: If ``color_map`` does not have the same length as
            ``value_cols``, if any column in ``value_cols`` is absent from
            ``df``, or if no numeric columns are found when ``value_cols``
            is ``None``.
        FileNotFoundError: If ``color_map`` is ``None`` and the default
            palette YAML cannot be found.

    Examples:
        Minimal call — ``value_cols`` and ``color_map`` inferred automatically:

        >>> fig, ax = plot_bar_group(reporter_df, group_col="clone")

        Explicit markers, colours from YAML:

        >>> markers = ["Oct4", "Sox2", "Nanog"]
        >>> fig, ax = plot_bar_group(reporter_df, "clone", value_cols=markers)

        Override colours and x-axis order:

        >>> fig, ax = plot_bar_group(
        ...     reporter_df,
        ...     group_col="clone",
        ...     value_cols=["Oct4", "Sox2", "Nanog"],
        ...     color_map=["#E69F00", "#56B4E9", "#009E73"],
        ...     group_order=["WT", "KO_1", "KO_2"],
        ...     ylabel="Relative expression",
        ...     title="Pluripotency factors",
        ... )

        With jitter and a replicate column excluded:

        >>> fig, ax = plot_bar_group(
        ...     reporter_df,
        ...     group_col="clone",
        ...     value_cols=["Oct4", "Sox2"],
        ...     exclude={"rep": "r2"},
        ...     jitter=True,
        ...     jitter_std=0.02,
        ... )

        Save to disk and continue customising the returned axes:

        >>> fig, ax = plot_bar_group(
        ...     reporter_df,
        ...     group_col="clone",
        ...     save_path="results/figures",
        ...     title="reporter_markers",
        ... )
        >>> ax.set_ylim(0, 1.2)
        >>> fig.savefig("results/figures/reporter_markers_final.png", dpi=300)
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
        else list(dict.fromkeys(df[group_col]))  # stable insertion order
    )

    # --- Aggregation -----------------------------------------------------
    grouped = df.groupby(group_col)[value_cols]
    means = grouped.mean().reindex(effective_order)
    errors = (
        grouped.apply(lambda g: g.max() - g.min()).reindex(effective_order) / 2
    )

    # --- Layout ----------------------------------------------------------
    x = np.arange(len(effective_order))
    n = len(value_cols)
    offsets = (np.arange(n) - (n - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=fig_size)

    for col, color, offset in zip(value_cols, color_map, offsets):
        ax.bar(
            x + offset,
            means[col],
            width=bar_width,
            label=col,
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
            rng = np.random.default_rng(jitter_seed)
            for j, group in enumerate(effective_order):
                y_vals = df.loc[df[group_col] == group, col].values
                noise = rng.normal(0.0, jitter_std, size=len(y_vals))
                ax.scatter(
                    np.full(len(y_vals), x[j] + offset) + noise,
                    y_vals,
                    color="black",
                    s=10,
                    alpha=0.7,
                    zorder=3,
                )

    # --- Axes ------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(effective_order, fontsize=fontsize)
    ax.set_xlabel(xlabel if xlabel is not None else group_col, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis="both", labelsize=fontsize)
    

    _legend_kw = {
        "loc": "upper right",
        "fontsize": fontsize - 2,
        "frameon": False,
        "ncol": n,
        "bbox_to_anchor": (1, 1),
    }
    if legend_kwargs:
        _legend_kw.update(legend_kwargs)
    ax.legend(**_legend_kw)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, alpha=alpha_grid)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_{title}.png", dpi=dpi, bbox_inches="tight")

    return fig, ax