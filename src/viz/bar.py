import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_bar(
    df: pd.DataFrame,
    title: str = "bar_plot",
    color_bar: list | None = None,
    height: float = 7,
    width: float = 12,
    bar_width: float = 0.25,
    bar_spacing: float = 1.0,
    xlabel: str = "samples",
    ylabel: str = "count",
    yscale: str = "log",
    ysticks: list | None = None,
    gap: float | None = None,
    annotate: bool = True,
    rotate_xticks: int = 45,
    show_legend: bool = True,
    show_grid: str | None = None,   # None | 'major' | 'minor' | 'both'
    bar_border: bool = True,        # black edge on bars
    fig_dir: str | None = None,
) -> plt.Figure:
    """
    Generic grouped bar plot for any tabular data.

    Args:
        df: DataFrame with samples as index, features as columns.
        title: Figure title and output filename stem.
        color_bar: Colors per feature; cycles if fewer than columns.
        height: Figure height in inches.
        width: Figure width in inches.
        bar_width: Width of individual bars.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        yscale: Y-axis scale — 'log' or 'linear'.
        ysticks: Explicit y-tick positions; auto-computed if None.
        gap: Vertical offset between bar top and annotation text.
        annotate: Whether to annotate bars with numeric values.
        rotate_xticks: Rotation angle for x-tick labels.
        fig_dir: Directory to save figure; skips save if None.

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If df is empty or yscale is invalid.

    Example:
        >>> df = pd.DataFrame(
        ...     {"feature1": [1200, 3400, 2100],
        ...      "feature2": [800,  4100, 1750],
        ...      "feature3": [300,  900,  600]},
        ...     index=["sample_A", "sample_B", "sample_C"],
        ... )
        >>> fig = plot_bar(df, title="my_experiment", ylabel="count")
    """
    # --- validation -----------------------------------------------------------
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('"df" must be a non-empty pandas DataFrame.')
    if yscale not in ("log", "linear"):
        raise ValueError('"yscale" must be "log" or "linear".')

    # --- colours --------------------------------------------------------------
    palette = list(plt.get_cmap("tab10").colors)
    colors = palette if color_bar is None else color_bar

    # --- layout ---------------------------------------------------------------
    plt.close("all")
    fig, ax = plt.subplots(figsize=(width, height))

    n_groups = len(df.index)
    n_features = len(df.columns)
    bar_pos = np.arange(n_groups) * bar_spacing

    # --- bars -----------------------------------------------------------------
    for i, feature in enumerate(df.columns):
        values = df[feature].values
        offset = (i - n_features / 2 + 0.5) * bar_width
        positions = bar_pos + offset

        ax.bar(
            positions,
            values,
            width=bar_width,
            label=feature,
            color=colors[i % len(colors)],
            edgecolor="black" if bar_border else "none",
            linewidth=0.8 if bar_border else 0,
        )

        if annotate:
            col_gap = df[feature].mean() * 0.05 if gap is None else gap
            for pos, val in zip(positions, values):
                ax.text(
                    pos,
                    val + col_gap,
                    str(val),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=9,
                )

    # --- axes: x --------------------------------------------------------------
    ax.set_xlabel(xlabel)
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(df.index, rotation=rotate_xticks, ha="right")
    # minor x ticks at group midpoints (visual separators)
    if n_groups > 1 and show_grid in ("minor", "both"):
        ax.set_xticks([x + bar_spacing / 2 for x in bar_pos[:-1]], minor=True)
        #ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.4)

    # --- axes: y --------------------------------------------------------------
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)

    if yscale == "log":
        if ysticks is None:
            pos_vals = df.values[df.values > 0]
            if pos_vals.size == 0:
                raise ValueError("Log scale requires at least one positive value.")
            log_min = int(np.floor(np.log10(pos_vals.min())))
            log_max = int(np.ceil(np.log10(df.values.max())))
            ysticks = np.logspace(log_min, log_max, log_max - log_min + 1)

        ax.set_yticks(ysticks)
        ax.set_yticklabels(
            [
                f"$10^{{{int(np.log10(y))}}}$"
                if np.isclose(y, 10 ** round(np.log10(y)))
                else f"${y / 10 ** int(np.log10(y)):.1f}\\times10^{{{int(np.log10(y))}}}$"
                for y in ysticks
            ]
        )
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs="auto", numticks=100)
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        #ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.35)

    # --- cosmetics ------------------------------------------------------------
    ax.set_title(title, pad=12)
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1), framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    
    # --- grid -----------------------------------------------------------------
    VALID_GRID = {None, "major", "minor", "both"}
    if show_grid not in VALID_GRID:
        raise ValueError(f'"show_grid" must be one of {VALID_GRID}.')

    if show_grid in ("major", "both"):
        ax.grid(True, which="major", linestyle="--", alpha=0.6)
    if show_grid in ("minor", "both"):
        ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.35)
        if n_groups > 1:                                        # x separators only with minor/both
            ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.4)
    if show_grid is None:
        ax.grid(False)

    # --- save -----------------------------------------------------------------
    if fig_dir is not None:
        fig.savefig(f"{fig_dir}/{title}.png", bbox_inches="tight", dpi=150)

    plt.close()
    return fig