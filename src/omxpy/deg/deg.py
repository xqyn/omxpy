import numpy as np
import pandas as pd


def deg(
    df,
    lfc=0.25,
    padj=0.05,
    lfc_col="logfoldchanges",
    padj_col="pvals_adj",
    up_label="UP",
    down_label="DOWN",
    no_de_label="NO_DE",
    verbose=True,
):
    """Classify genes as UP, DOWN, or non-differentially expressed.

    Annotates genes from DESeq2 (or equivalent) results based on log2 fold
    change and adjusted p-value thresholds. NA values in either the fold
    change or adjusted p-value column are dropped before classification.
    The returned dataframe is sorted by log2 fold change in descending order.

    Args:
        df (pd.DataFrame):
            Dataframe containing differential expression results, typically
            from DESeq2 or a similar tool.
        lfc (float, optional):
            Absolute log2 fold change threshold. Genes with
            ``|log2FoldChange| > lfc`` and ``padj < padj`` are called
            significant. Defaults to 1.0.
        padj (float, optional):
            Adjusted p-value threshold. Defaults to 0.05.
        lfc_col (str, optional):
            Column name for log2 fold change values. Defaults to
            ``"log2FoldChange"``.
        padj_col (str, optional):
            Column name for adjusted p-values. Defaults to ``"padj"``.
        up_label (str, optional):
            Label assigned to upregulated genes. Defaults to ``"UP"``.
        down_label (str, optional):
            Label assigned to downregulated genes. Defaults to ``"DOWN"``.
        no_de_label (str, optional):
            Label assigned to non-differentially expressed genes. Defaults to
            ``"NO DE"``.

    Returns:
        pd.DataFrame:
            Copy of the input dataframe with NA rows removed, an added
            ``"deg"`` column containing the classification labels, and rows
            sorted by ``lfc_col`` in descending order.

    Raises:
        KeyError:
            If ``lfc_col`` or ``padj_col`` are not found in ``df``.

    Example:
        >>> import pandas as pd
        >>> from deg import deg
        >>> results = pd.read_csv("deseq2_results.csv", index_col=0)
        >>> df_annotated = deg(results, lfc=1.0, padj=0.05)
        >>> df_annotated["deg"].value_counts()
    """
    for col in (lfc_col, padj_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe.")
    
    df_de = df.copy()
    df_de = df_de[df_de[lfc_col].notna()]   # drop NA log2FoldChange
    df_de = df_de[df_de[padj_col].notna()]  # drop NA adjusted p-value

    df_de["deg"] = no_de_label
    df_de.loc[(df_de[lfc_col] >  lfc) & (df_de[padj_col] < padj), "deg"] = up_label
    df_de.loc[(df_de[lfc_col] < -lfc) & (df_de[padj_col] < padj), "deg"] = down_label

    df_de = df_de.sort_values(lfc_col, ascending=False)

    # ── summary print ─────────────────────────────────────────────────────────
    if verbose:
        n_up   = (df_de["deg"] == up_label).sum()
        n_down = (df_de["deg"] == down_label).sum()
        n_ns   = (df_de["deg"] == no_de_label).sum()
        print(
            f"\n[ DEG Summary ]\n"
            f"  Threshold  : |log2FC| > {lfc}  &  padj < {padj}\n"
            f"  ─────────────────────────────\n"
            f"  ▲ UP       : {n_up:>6} genes\n"
            f"  ▼ DOWN     : {n_down:>6} genes\n"
            f"  ◼ NO DE    : {n_ns:>6} genes\n"
            f"  ─────────────────────────────\n"
            f"  Total      : {len(df_de):>6} genes\n"
        )

    return df_de
