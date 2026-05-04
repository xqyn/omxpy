'''
project: epi_mem
XQ - Leiden UMC

'''
import pandas as pd


# --- function ----------------------------------------
def summary_adatas(adatas):
    '''summerize information from adata'''
    summary_adatas = pd.DataFrame([
        {'sample': sample, 'cells': adata.n_obs, 'genes': adata.n_vars}
        for sample, adata in adatas.items()
        ])
    return print(summary_adatas)


def check_adata_normalization(adata):
    """Diagnose the normalization state of an AnnData object.

    Inspects ``adata.X``, row sums, and ``adata.uns`` to infer which
    preprocessing steps (library-size normalization, log1p transformation,
    z-score scaling) have been applied, and prints a human-readable summary.

    Args:
        adata (anndata.AnnData): Annotated data matrix. ``adata.X`` may be
            a dense ``numpy.ndarray`` or a sparse ``scipy.sparse`` matrix.

    Returns:
        None: Results are printed to stdout.

    Prints:
        max value (float): Maximum expression value in ``adata.X``.
        min value (float): Minimum expression value in ``adata.X``.
        integer values (bool): ``True`` if all values are integers,
            suggesting raw counts.
        has negatives (bool): ``True`` if any value is negative,
            suggesting ``sc.pp.scale()`` was applied.
        equal row sums (bool): ``True`` if all cells have equal total
            counts (rtol=1e-3), suggesting ``sc.pp.normalize_total()``
            was applied.
        log1p in uns (bool): ``True`` if ``'log1p'`` key exists in
            ``adata.uns``, confirming ``sc.pp.log1p()`` was applied.
        adata.raw (bool): ``True`` if ``adata.raw`` is not ``None``,
            indicating raw counts were saved prior to normalization.

    Example:
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> sc.pp.normalize_total(adata, target_sum=1e4)
        >>> sc.pp.log1p(adata)
        >>> check_normalization(adata)
        max value      : 9.894
        min value      : 0.000
        integer values : False   ← raw counts if True
        has negatives  : False   ← sc.pp.scale applied if True
        equal row sums : False   ← normalize_total applied if True
        log1p in uns   : True    ← sc.pp.log1p applied if True
        adata.raw      : False   ← raw saved before norm if True
    """

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    row_sums = X.sum(axis=1)
    is_integer   = np.allclose(X % 1, 0)
    is_negative  = X.min() < 0
    is_libsize   = np.allclose(row_sums, row_sums[0], rtol=1e-3)
    is_log1p     = 'log1p' in adata.uns

    print(f"max value      : {X.max():.3f}")
    print(f"min value      : {X.min():.3f}")
    print(f"integer values : {is_integer}   ← raw counts if True")
    print(f"has negatives  : {is_negative}  ← sc.pp.scale applied if True")
    print(f"equal row sums : {is_libsize}   ← normalize_total applied if True")
    print(f"log1p in uns   : {is_log1p}     ← sc.pp.log1p applied if True")
    print(f"adata.raw      : {adata.raw is not None}  ← raw saved before norm if True")
