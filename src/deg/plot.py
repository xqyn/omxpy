import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from deg import deg


def plot_volcano(
    df,
    lfc_col="log2FoldChange",
    padj_col="padj",
    gene_col="gene",
    fc_thresh=1.0,
    p_thresh=0.05,
    top_n=5,
    xlim=(-2.5, 2.5),
    figsize=(7, 6),
    title="Volcano Plot",
    up_color="#E24B4A",
    down_color="#378ADD",
    ns_color="grey",
    point_size=15,
    alpha=0.6,
    label_fontsize=7,
    axis_fontsize=12,
    title_fontsize=13,
    legend_fontsize=9,
    save_path=None,
    ax=None,
):
    """Plot a volcano plot from differential expression results.

    Internally calls :func:`deg.deg` to classify genes as UP, DOWN, or NO DE,
    then renders a scatter plot of log2 fold change vs. -log10 adjusted
    p-value. Threshold lines are drawn at ``p_thresh`` and ``±fc_thresh``.
    The top ``top_n`` most significant up- and down-regulated genes are
    labeled on the plot.

    Args:
        df (pd.DataFrame):
            Raw differential expression dataframe. Passed directly to
            :func:`deg.deg`, so NA rows are handled there.
        lfc_col (str, optional):
            Column name for log2 fold change values. Defaults to
            ``"log2FoldChange"``.
        padj_col (str, optional):
            Column name for adjusted p-values. Defaults to ``"padj"``.
        gene_col (str, optional):
            Column name for gene names used as text labels. Defaults to
            ``"gene"``.
        fc_thresh (float, optional):
            Absolute log2 fold change threshold passed to :func:`deg.deg`
            and used to draw vertical dashed lines. Defaults to 1.0.
        p_thresh (float, optional):
            Adjusted p-value threshold passed to :func:`deg.deg` and used
            to draw the horizontal dashed line. Defaults to 0.05.
        top_n (int, optional):
            Number of top significant genes to label per direction, ranked
            by smallest adjusted p-value. Defaults to 5.
        xlim (tuple[float, float], optional):
            X-axis limits as ``(xmin, xmax)``. Defaults to ``(-2.5, 2.5)``.
        figsize (tuple[float, float], optional):
            Figure size in inches, used only when ``ax`` is ``None``.
            Defaults to ``(7, 6)``.
        title (str, optional):
            Plot title. Defaults to ``"Volcano Plot"``.
        up_color (str, optional):
            Color for significantly upregulated genes. Defaults to
            ``"#E24B4A"``.
        down_color (str, optional):
            Color for significantly downregulated genes. Defaults to
            ``"#378ADD"``.
        ns_color (str, optional):
            Color for non-significant genes. Defaults to ``"grey"``.
        point_size (float, optional):
            Scatter point marker size. Defaults to 15.
        alpha (float, optional):
            Scatter point transparency. Defaults to 0.6.
        label_fontsize (float, optional):
            Font size for gene name labels. Defaults to 7.
        axis_fontsize (float, optional):
            Font size for axis labels. Defaults to 12.
        title_fontsize (float, optional):
            Font size for the plot title. Defaults to 13.
        legend_fontsize (float, optional):
            Font size for the legend text. Defaults to 9.
        save_path (str or None, optional):
            File path to save the figure (e.g. ``"volcano.png"``). If
            ``None``, the figure is not saved. Defaults to ``None``.
        ax (matplotlib.axes.Axes or None, optional):
            Existing axes to draw on. If ``None``, a new figure and axes
            are created. Defaults to ``None``.

    Returns:
        matplotlib.axes.Axes:
            The axes object containing the rendered volcano plot.

    Raises:
        KeyError:
            If any of ``lfc_col``, ``padj_col``, or ``gene_col`` are missing
            from ``df``.
        ValueError:
            If ``top_n`` is negative.

    Example:
        >>> import pandas as pd
        >>> from plot import plot_volcano
        >>> df = pd.read_csv("deseq2_results.csv", index_col=0)
        >>> ax = plot_volcano(df, title="My Comparison")
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
    """
    if gene_col not in df.columns:
        raise KeyError(f"Column '{gene_col}' not found in dataframe.")
    if top_n < 0:
        raise ValueError("top_n must be >= 0.")

    # ── classify genes ────────────────────────────────────────────────────────
    plot_df = deg(
        df,
        lfc=fc_thresh,
        padj=p_thresh,
        lfc_col=lfc_col,
        padj_col=padj_col,
    )

    # ── compute -log10(padj) ──────────────────────────────────────────────────
    plot_df["-log10padj"] = -np.log10(plot_df[padj_col].clip(lower=1e-300))

    # ── map DEG label → color ─────────────────────────────────────────────────
    color_map = {"UP": up_color, "DOWN": down_color, "NO DE": ns_color}
    plot_df["color"] = plot_df["deg"].map(color_map).fillna(ns_color)

    # ── select genes to label ─────────────────────────────────────────────────
    top_up   = plot_df[plot_df["deg"] == "UP"].nsmallest(top_n, padj_col)
    top_down = plot_df[plot_df["deg"] == "DOWN"].nsmallest(top_n, padj_col)
    to_label = pd.concat([top_up, top_down])

    # ── draw plot ─────────────────────────────────────────────────────────────
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        plot_df[lfc_col],
        plot_df["-log10padj"],
        c=plot_df["color"],
        alpha=alpha,
        s=point_size,
        linewidths=0,
    )

    ax.axhline(-np.log10(p_thresh), color="grey", linestyle="--", linewidth=0.8)
    ax.axvline( fc_thresh,          color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_thresh,          color="grey", linestyle="--", linewidth=0.8)

    for _, row in to_label.iterrows():
        ax.text(
            row[lfc_col] + 0.05,
            row["-log10padj"] + 0.05,
            row[gene_col],
            fontsize=label_fontsize,
        )

    ax.set_xlim(*xlim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("log2 fold change", fontsize=axis_fontsize)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=axis_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    n_up   = (plot_df["deg"] == "UP").sum()
    n_down = (plot_df["deg"] == "DOWN").sum()

    patches = [
        mpatches.Patch(color=up_color,   label=f"Up ({n_up})"),
        mpatches.Patch(color=down_color, label=f"Down ({n_down})"),
        mpatches.Patch(color=ns_color,   label="NS"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=legend_fontsize)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
