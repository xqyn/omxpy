"""GO Enrichment Analysis.

Python translation of the original R implementation using gseapy and
matplotlib. Supports up- and down-regulated gene sets across MF, CC,
and BP GO ontologies.

Dependencies:
  pip install gseapy pandas matplotlib numpy
"""

import math
import os
import textwrap
import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

try:
  import gseapy as gp
except ImportError as exc:
  raise ImportError('Install gseapy: pip install gseapy') from exc


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ONTOLOGY = {
    'MF': 'Molecular Function',
    'CC': 'Cellular Component',
    'BP': 'Biological Process',
}

_GO_CMAP = LinearSegmentedColormap.from_list('go_cmap', ['#05474A', '#8CD0D5'])


# ---------------------------------------------------------------------------
# format_scientific_expr
# ---------------------------------------------------------------------------

def format_scientific_expr(x: float) -> str:
  """Formats a float as a Unicode scientific-notation string for plot labels.

  Args:
    x: The numeric value to format.

  Returns:
    A string in the form ``"1.23 × 10⁻⁴"``, or ``"0"`` when *x* is zero.

  Example:
    >>> format_scientific_expr(0.000123)
    '1.23 × 10⁻⁴'
  """
  if x == 0:
    return '0'

  exp = int(math.floor(math.log10(abs(x))))
  coef = x / 10 ** exp
  sup_map = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
  sup_exp = str(exp).translate(sup_map)
  return f'{coef:.2g} × 10{sup_exp}'


# ---------------------------------------------------------------------------
# go_enrichment
# ---------------------------------------------------------------------------

def go_enrichment(
    genes: Optional[list[str]] = None,
    regulation: str = 'Upregulated',
    ont: str = 'MF',
    pvalue_cutoff: float = 0.05,
    qvalue_cutoff: float = 0.05,
    organism: str = 'human',
) -> pd.DataFrame:
  """Performs GO enrichment analysis for a set of gene symbols.

  Queries the Enrichr API via gseapy and post-filters results to the
  requested p-value and q-value thresholds. GeneRatio values are
  negated for down-regulated genes so they render as leftward bars.

  Args:
    genes: Gene symbols to test for enrichment.
    regulation: Direction label — ``'Upregulated'`` or ``'Downregulated'``.
    ont: GO ontology key: ``'MF'``, ``'CC'``, or ``'BP'``.
    pvalue_cutoff: Adjusted p-value threshold (default ``0.05``).
    qvalue_cutoff: Q-value / FDR threshold (default ``0.05``).
    organism: Organism string passed to gseapy (default ``'human'``).

  Returns:
    A DataFrame with processed enrichment results, or an empty DataFrame
    when no significant terms are found.
  """
  if not genes:
    return pd.DataFrame()

  print(f'GO {regulation} for {ONTOLOGY[ont]}...')

  enr = gp.enrichr(
      gene_list=genes,
      gene_sets=f'GO_{ont}',
      organism='human',
      cutoff=0.05,
      no_plot=True,
  )

  results = enr.results
  if results is None or results.empty:
    print(f'No {regulation} GO terms found for: {ONTOLOGY[ont]}!')
    return pd.DataFrame()

  results = results[
      (results['Adjusted P-value'] <= pvalue_cutoff) &
      (results['Adjusted P-value'] <= qvalue_cutoff)
  ].copy()

  if results.empty:
    print(f'No {regulation} GO terms found for: {ONTOLOGY[ont]}!')
    return pd.DataFrame()

  results = results.rename(columns={
      'Term': 'Description',
      'Adjusted P-value': 'p.adjust',
      'P-value': 'pvalue',
      'Overlap': 'GeneRatio_raw',
      'Genes': 'geneID',
  })

  def _parse_ratio(ratio_str: str) -> float:
    """Parses a ``'k/n'`` overlap string into a float ratio."""
    try:
      numerator, denominator = ratio_str.split('/')
      return int(numerator) / int(denominator)
    except (ValueError, AttributeError):
      return float('nan')

  results['GeneRatio'] = results['GeneRatio_raw'].apply(_parse_ratio)
  results['Count'] = results['GeneRatio_raw'].apply(
      lambda s: int(s.split('/')[0]) if '/' in str(s) else np.nan
  )
  results['Regulation'] = regulation

  if regulation == 'Downregulated':
    results['GeneRatio'] = -results['GeneRatio']

  return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# combine_go_data
# ---------------------------------------------------------------------------

def combine_go_data(
    go_up: Optional[pd.DataFrame] = None,
    go_down: Optional[pd.DataFrame] = None,
    ont: str = 'MF',
    top: int = 10,
    go_id: bool = True,
) -> pd.DataFrame:
  """Combines top GO terms from up- and down-regulated enrichment results.

  Selects up to *top* terms from each direction, removes rows with missing
  Count values, and optionally appends the GO ID to each term description.

  Args:
    go_up: Enrichment results for up-regulated genes (from
      :func:`go_enrichment`). ``None`` is treated as an empty DataFrame.
    go_down: Enrichment results for down-regulated genes (from
      :func:`go_enrichment`). ``None`` is treated as an empty DataFrame.
    ont: GO ontology key used in warning messages.
    top: Maximum number of terms to retain from each direction (default
      ``10``).
    go_id: If ``True``, appends the GO ID in parentheses to each
      ``Description`` value (default ``True``).

  Returns:
    A combined DataFrame ready for :func:`plot_go_terms`, or an empty
    DataFrame when both inputs are empty.

  Raises:
    UserWarning: When both *go_up* and *go_down* are empty.
  """
  go_up = go_up if go_up is not None else pd.DataFrame()
  go_down = go_down if go_down is not None else pd.DataFrame()

  if go_up.empty and go_down.empty:
    warnings.warn(
        f'No GO term found in {ONTOLOGY[ont]} for both GO up and down. '
        'Consider increasing pvalue_cutoff and qvalue_cutoff.'
    )
    return pd.DataFrame()

  parts = []
  if not go_up.empty:
    parts.append(go_up.iloc[: min(top, len(go_up))])
  if not go_down.empty:
    parts.append(go_down.iloc[: min(top, len(go_down))])

  go_combine = pd.concat(parts, ignore_index=True).dropna(subset=['Count'])

  if go_id and 'ID' in go_combine.columns:
    go_combine['Description'] = (
        go_combine['Description'] + ' (' + go_combine['ID'] + ')'
    )

  return go_combine.reset_index(drop=True)


# ---------------------------------------------------------------------------
# _make_colorbar  (private helper)
# ---------------------------------------------------------------------------

def _make_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    pvals: pd.Series,
    setting: str,
    ont: str,
    pvalue_cutoff: float,
    qvalue_cutoff: float,
) -> None:
  """Adds a log-scaled p-value colorbar to *ax*.

  Args:
    fig: The parent Figure.
    ax: The Axes to attach the colorbar to.
    pvals: Series of adjusted p-values used to set the normalisation range.
    setting: Analysis label shown in the colorbar title.
    ont: GO ontology key (``'MF'``, ``'CC'``, or ``'BP'``).
    pvalue_cutoff: P-value cutoff shown in the label.
    qvalue_cutoff: Q-value cutoff shown in the label.
  """
  vmin, vmax = pvals.min(), pvals.max()
  norm = LogNorm(vmin=vmin if vmin != vmax else vmin / 10, vmax=vmax)

  sm = plt.cm.ScalarMappable(cmap=_GO_CMAP, norm=norm)
  sm.set_array([])

  cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
  cbar.set_label(
      f'{setting}:\n{ONTOLOGY[ont]}\n'
      f'(p_value < {pvalue_cutoff}\n q_value < {qvalue_cutoff})\n\nadj P-value',
      fontsize=8,
  )

  min_exp = math.floor(math.log10(vmin) / 2) * 2
  p_breaks = [10 ** e for e in range(min_exp, -1, 2)]
  if p_breaks:
    cbar.set_ticks(p_breaks)
    cbar.set_ticklabels([format_scientific_expr(b) for b in p_breaks])

  cbar.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# plot_go_terms
# ---------------------------------------------------------------------------

def plot_go_terms(
    go_combine: pd.DataFrame,
    setting: str = 'go_analysis',
    ont: str = 'MF',
    pvalue_cutoff: float = 0.05,
    qvalue_cutoff: float = 0.05,
    gene_ratio_col: str = 'GeneRatio',
    figure_dir: Optional[str] = None,
    width: float = 10,
    height: float = 7,
    wrap_width: int = 50,
    fig_format: Union[str, list[str]] = 'png',
) -> plt.Figure:
  """Plots GO enrichment terms as a horizontal diverging bar chart.

  Bars extend rightward for up-regulated terms and leftward for
  down-regulated terms, coloured by adjusted p-value on a log scale.
  Gene counts are printed at the tip of each bar.

  Args:
    go_combine: Combined GO enrichment data produced by
      :func:`combine_go_data`.
    setting: Label used in the colorbar title and output filename
      (default ``'go_analysis'``).
    ont: GO ontology key: ``'MF'``, ``'CC'``, or ``'BP'`` (default
      ``'MF'``).
    pvalue_cutoff: P-value threshold displayed in the colorbar label
      (default ``0.05``).
    qvalue_cutoff: Q-value threshold displayed in the colorbar label
      (default ``0.05``).
    gene_ratio_col: Column name containing GeneRatio values (default
      ``'GeneRatio'``).
    figure_dir: Directory in which to save the figure. When ``None`` the
      figure is not written to disk (default ``None``).
    width: Figure width in inches (default ``10``).
    height: Figure height in inches (default ``7``).
    wrap_width: Maximum characters per line for y-axis label wrapping
      (default ``50``).
    fig_format: File extension(s) for saved figures, e.g. ``'png'`` or
      ``['png', 'pdf']`` (default ``'png'``).

  Returns:
    The completed :class:`matplotlib.figure.Figure`.

  Raises:
    ValueError: When *go_combine* is missing required columns.
  """
  required_cols = {'Description', 'p.adjust', 'Count', 'Regulation'}
  missing_cols = required_cols - set(go_combine.columns)
  if missing_cols:
    raise ValueError(
        f'go_combine is missing required columns: {missing_cols}'
    )

  if isinstance(fig_format, str):
    fig_format = [fig_format]

  # Prepare data
  df = go_combine.copy().reset_index(drop=True)
  df['label'] = df['Description'].apply(
      lambda t: '\n'.join(textwrap.wrap(str(t), wrap_width))
  )
  df = df.sort_values(gene_ratio_col)

  ratios = df[gene_ratio_col].values
  max_gr = math.ceil(ratios.max() * 100) / 100
  min_gr = math.floor(ratios.min() * 100) / 100

  # Avoid a zero-only axis when all values share a sign
  if min_gr > 0 and max_gr > 0:
    min_gr = 0.01
  elif min_gr < 0 and max_gr < 0:
    max_gr = -0.01

  # Build x-axis ticks, excluding zero
  tick_vals = [
      v / 100
      for v in range(int(min_gr * 100), int(max_gr * 100) + 1)
      if v != 0
  ]

  norm = LogNorm(vmin=df['p.adjust'].min(), vmax=df['p.adjust'].max())

  # Build figure
  fig, ax = plt.subplots(figsize=(width, height))

  bar_colors = _GO_CMAP(norm(df['p.adjust'].values))
  bars = ax.barh(df['label'], df[gene_ratio_col], color=bar_colors)

  # Count labels at bar tips
  label_offset = abs(max_gr - min_gr) * 0.02
  for bar, row in zip(bars, df.itertuples()):
    x = getattr(row, gene_ratio_col)
    is_up = row.Regulation == 'Upregulated'
    ax.text(
        x + (label_offset if is_up else -label_offset),
        bar.get_y() + bar.get_height() / 2,
        str(int(row.Count)),
        va='center',
        ha='left' if is_up else 'right',
        fontsize=9,
        color='black',
    )

  # Centre line
  ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

  # UP / DOWN direction labels when both sides are present
  has_up = (df[gene_ratio_col] > 0).any()
  has_down = (df[gene_ratio_col] < 0).any()
  if has_up and has_down:
    y_annot = -1.0
    arrow_kw = dict(arrowstyle='-', lw=3)
    ax.annotate(
        '',
        xy=(max_gr + 0.01, y_annot - 0.15),
        xytext=(-0.0025, y_annot - 0.15),
        xycoords='data',
        arrowprops={**arrow_kw, 'color': '#08519C'},
    )
    ax.annotate(
        '',
        xy=(min_gr - 0.01, y_annot - 0.15),
        xytext=(0.0025, y_annot - 0.15),
        xycoords='data',
        arrowprops={**arrow_kw, 'color': '#F48D79'},
    )
    label_kw = dict(va='center', ha='center', fontsize=8,
                    color='white', fontweight='bold')
    ax.text(0.02, y_annot - 0.5, 'UP',
            bbox=dict(boxstyle='round,pad=0.2', fc='#08519C', ec='none'),
            **label_kw)
    ax.text(-0.02, y_annot - 0.5, 'DOWN',
            bbox=dict(boxstyle='round,pad=0.2', fc='#F48D79', ec='none'),
            **label_kw)

  # Axis limits, ticks, and labels
  ax.set_xlim(
      min_gr * 1.25 if min_gr < 0 else 0,
      max_gr * 1.25 if max_gr > 0 else 0,
  )
  ax.set_xticks(tick_vals)
  ax.set_xticklabels(
      [str(abs(v)) for v in tick_vals],
      rotation=45,
      va='top',
      fontsize=9,
  )
  ax.set_xlabel('Gene Ratio', fontsize=11)
  ax.set_ylabel('GO Terms', fontsize=11)
  ax.tick_params(axis='y', labelsize=9)
  for tick_label in ax.get_yticklabels():
    tick_label.set_fontweight('bold')
  ax.grid(False)

  _make_colorbar(
      fig, ax,
      pvals=df['p.adjust'],
      setting=setting,
      ont=ont,
      pvalue_cutoff=pvalue_cutoff,
      qvalue_cutoff=qvalue_cutoff,
  )

  plt.tight_layout()

  # Save
  if figure_dir is not None:
    os.makedirs(figure_dir, exist_ok=True)
    for ext in fig_format:
      filepath = os.path.join(
          figure_dir,
          f'{setting}_{ont}_pv_{pvalue_cutoff}_qv_{qvalue_cutoff}_go_term.{ext}',
      )
      fig.savefig(filepath, dpi=320, bbox_inches='tight', facecolor='white')
      print(f'Saved figure to: {filepath}')

  return fig


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  up_genes = ['TP53', 'EGFR', 'MYC', 'BRCA1', 'CDK2']
  down_genes = ['PTEN', 'RB1', 'VHL', 'APC', 'SMAD4']

  go_up = go_enrichment(up_genes, regulation='Upregulated', ont='BP')
  go_down = go_enrichment(down_genes, regulation='Downregulated', ont='BP')

  go_data = combine_go_data(go_up, go_down, ont='BP', top=10, go_id=True)

  if not go_data.empty:
    fig = plot_go_terms(
        go_data,
        setting='example',
        ont='BP',
        figure_dir='./go_plots/',
        fig_format=['png', 'pdf'],
    )
    plt.show()