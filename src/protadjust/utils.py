import statsmodels.api as sm
import numpy as np
import pandas as pd
import logging
import scipy
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'regress_out_covariates_matrix',
    'rank_INT_matrix',
    'standardize_matrix',
    'scramble_matrix',
    'extract_principal_components',
]


def extract_principal_components(matrix: pd.DataFrame, n_components: int = 10, column_prefix='Protein PC ') -> pd.DataFrame:
    """Extract principal components from a DataFrame using SVD.

    Args:
        matrix: Input data matrix (samples x proteins).
        n_components: Number of principal components to extract.
        column_prefix: Prefix for PC column names.

    Returns:
        DataFrame of shape (n_samples, n_components) with PC scores.
    """
    centered = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    centered = np.nan_to_num(centered)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = U[:, :n_components] * S[:n_components]
    return pd.DataFrame(pcs, index=matrix.index,
                        columns=[f"{column_prefix}{i + 1}" for i in range(n_components)])


def standardize_matrix(matrix: pd.DataFrame, n_jobs=4) -> pd.DataFrame:
    """Z-score standardize each column (protein) independently.

    Args:
        matrix: Input DataFrame (samples x proteins).
        n_jobs: Number of parallel jobs.

    Returns:
        Standardized DataFrame with mean=0 and std=1 per protein.
    """
    transformed = _apply_parallel(matrix, _standardize_series, n_jobs=n_jobs)
    assert transformed.shape == matrix.shape
    assert (matrix.index.values == transformed.index.values).all()
    return transformed


def rank_INT_matrix(matrix: pd.DataFrame, c=3.0 / 8, stochastic=True, seed=42, n_jobs=4) -> pd.DataFrame:
    """Apply rank-based inverse normal transformation (RINT) column-wise.

    Args:
        matrix: Input DataFrame (samples x proteins).
        c: Blom's constant for the RINT formula.
        stochastic: If True, ties are broken randomly before ranking.
        seed: Random seed for stochastic tie-breaking.
        n_jobs: Number of parallel jobs.

    Returns:
        RINT-transformed DataFrame.
    """
    sample_ids = matrix.index.values
    transformed = _apply_parallel(
        matrix,
        lambda col: _rank_INT_series(col, c=c, stochastic=stochastic, seed=seed),
        n_jobs=n_jobs,
    )
    assert transformed.shape == matrix.shape
    assert (sample_ids == transformed.index.values).all()
    return transformed


def scramble_matrix(matrix: pd.DataFrame, random_state=42) -> pd.DataFrame:
    """Shuffle each column independently (useful for permutation testing).

    Args:
        matrix: Input DataFrame (samples x proteins).
        random_state: Random seed.

    Returns:
        Scrambled DataFrame with the same index as the input.
    """
    orig_idx = matrix.index
    scrambled = matrix.apply(
        lambda col: col.sample(frac=1, random_state=random_state).reset_index(drop=True), axis=0
    )
    scrambled.set_index(orig_idx, inplace=True)
    assert scrambled.shape == matrix.shape
    assert (matrix.index.values == scrambled.index.values).all()
    return scrambled


def regress_out_covariates_matrix(
    matrix: pd.DataFrame,
    covariates_df: pd.DataFrame,
    columns_to_correct: list[str] = None,
    output_dir: Path = None,
    n_jobs=4,
) -> pd.DataFrame:
    """Regress covariates out of every protein column using OLS.

    Args:
        matrix: Protein intensity matrix (samples x proteins).
        covariates_df: Covariate DataFrame aligned to the same sample index.
        columns_to_correct: Subset of protein columns to correct; defaults to all.
        output_dir: If given, per-protein regression stats are saved here.
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame of OLS residuals with the same shape as the input matrix.
    """
    if not matrix.index.equals(covariates_df.index):
        logger.error("Matrix indices: %s", matrix.index)
        logger.error("Covariates indices: %s", covariates_df.index)
        raise ValueError("Matrix and covariates DataFrame must have the same index")

    sample_ids = matrix.index.values

    if columns_to_correct is None:
        columns_to_correct = matrix.columns.tolist()
    else:
        matrix = matrix[columns_to_correct]

    # Process proteins with fewest NAs first (better joblib memory reuse)
    na_counts = matrix.isna().sum()
    sorted_columns = na_counts.sort_values(ascending=True).index
    matrix = matrix[sorted_columns]

    corrected = _apply_parallel(
        matrix,
        lambda col: _regress_out_covariates_series(col, covariates_df, output_dir=output_dir),
        n_jobs=n_jobs,
    )

    corrected = corrected[columns_to_correct]

    assert corrected.shape == (len(sample_ids), len(columns_to_correct))
    assert (sample_ids == corrected.index.values).all()
    return corrected


def _regress_out_covariates_series(
    series: pd.Series, covariates_df: pd.DataFrame, output_dir: Path = None
) -> pd.Series:
    """OLS regression of a single protein series against covariates.

    Rows with NaN in the protein or any covariate are excluded from the fit.
    The returned series is reindexed to the original index (NaNs re-inserted).

    Args:
        series: Protein intensity series (samples as index).
        covariates_df: Covariate DataFrame (same sample index as series).
        output_dir: If given, save per-protein regression stats TSV here.

    Returns:
        OLS residuals reindexed to the original sample index.
    """
    assert covariates_df.shape[0] == series.shape[0]
    assert 'eid' not in covariates_df.columns, "Transformed covariates should not contain 'eid'"

    orig_idx = series.index

    # Remove rows where any covariate is NaN
    valid_cov = ~covariates_df.isna().any(axis=1)
    covariates_df = covariates_df[valid_cov]
    series = series[valid_cov]

    # Remove rows where the protein is NaN
    valid_prot = ~np.isnan(series)
    series = series[valid_prot]
    covariates_df = covariates_df[valid_prot]

    # Drop covariates that became constant after row filtering
    covariates_df = covariates_df.loc[:, covariates_df.nunique() > 1]

    logger.debug("Fitting OLS for %s with n=%d samples, %d covariates",
                 series.name, series.shape[0], covariates_df.shape[1])

    covariates_df = sm.add_constant(covariates_df, has_constant='raise')
    model = sm.OLS(series.astype(np.float32), covariates_df)
    results = model.fit()

    residuals = pd.Series(results.resid, name=series.name, index=series.index).reindex(orig_idx)

    if output_dir is not None:
        stats_df = pd.DataFrame({'beta': results.params, 'pvalue': results.pvalues})
        stats_df.to_csv(output_dir / f"{series.name}_regression_stats.tsv", sep='\t')

    return residuals


def _t_to_normal_transform_series(series: pd.Series, degree_freedom: float) -> pd.Series:
    """Transform a series of t-distributed z-scores to standard normal z-scores.

    Uses the probability integral transform:
        z_normal = Φ⁻¹(F_t(z | df))

    Args:
        series: Series of t-distributed z-scores.
        degree_freedom: Degrees of freedom of the t-distribution.

    Returns:
        Series of standard-normal z-scores.
    """
    arr = series.to_numpy()
    mask = np.isnan(arr)
    pv = np.full_like(arr, np.nan, dtype=np.float64)
    z_normal = np.full_like(arr, np.nan, dtype=np.float64)
    pv[~mask] = scipy.stats.t.cdf(arr[~mask], degree_freedom)
    z_normal[~mask] = stats.norm.ppf(pv[~mask])
    return pd.Series(z_normal, index=series.index, name=series.name)


def t_to_normal_transform_matrix(matrix: pd.DataFrame, degree_freedom: np.ndarray, n_jobs=4) -> pd.DataFrame:
    """Apply t-to-normal transformation column-wise in parallel.

    Args:
        matrix: DataFrame of t-distributed z-scores (samples x proteins).
        degree_freedom: Per-protein degrees of freedom (length = n_proteins).
        n_jobs: Number of parallel jobs.

    Returns:
        DataFrame of standard-normal z-scores.
    """
    result_list = Parallel(n_jobs=n_jobs)(
        delayed(_t_to_normal_transform_series)(matrix[col], degree_freedom[i])
        for i, col in enumerate(tqdm(matrix.columns))
    )
    transformed = pd.concat(result_list, axis=1)
    transformed.columns = matrix.columns
    assert transformed.shape == matrix.shape
    assert (matrix.index.values == transformed.index.values).all()
    return transformed


def _apply_parallel(df: pd.DataFrame, func, n_jobs=4, axis=0) -> pd.DataFrame:
    """Apply a function to each column (axis=0) or row (axis=1) in parallel.

    Args:
        df: Input DataFrame.
        func: Function accepting a Series and returning a Series.
        n_jobs: Number of parallel workers.
        axis: 0 for column-wise, 1 for row-wise.

    Returns:
        Concatenated result DataFrame.
    """
    if axis == 0:
        result_list = Parallel(n_jobs=n_jobs)(delayed(func)(df[col]) for col in tqdm(df.columns))
        res = pd.concat(result_list, axis=1)
        res.columns = df.columns
        return res
    elif axis == 1:
        result_list = Parallel(n_jobs=n_jobs)(delayed(func)(row) for _, row in tqdm(df.iterrows()))
        res = pd.concat(result_list, axis=0)
        res.index = df.index
        return res
    else:
        raise ValueError("axis must be 0 (columns) or 1 (rows)")


def _rank_to_normal(rank, c, n):
    x = (rank - c) / (n - 2 * c + 1)
    return scipy.stats.norm.ppf(x)


def _rank_INT_series(series: pd.Series, c=3.0 / 8, stochastic=True, seed=42) -> pd.Series:
    """Rank-based inverse normal transformation for a single protein series.

    NaN values are excluded from ranking and re-inserted afterwards.

    Args:
        series: Protein intensity series.
        c: Blom's constant.
        stochastic: If True, ties are broken randomly.
        seed: Random seed for stochastic tie-breaking.

    Returns:
        RINT-transformed series aligned to the original index.
    """
    assert isinstance(series, pd.Series)
    assert isinstance(c, float)
    assert isinstance(stochastic, bool)

    orig_idx = series.index
    series = series.loc[~pd.isnull(series)]

    if stochastic:
        series = series.loc[np.random.permutation(series.index)]
        rank = scipy.stats.rankdata(series, method="ordinal")
    else:
        rank = scipy.stats.rankdata(series, method="average")

    transformed = pd.Series(_rank_to_normal(rank, c=c, n=len(rank)), index=series.index)
    return transformed.reindex(orig_idx)


def _standardize_series(series: pd.Series) -> pd.Series:
    """Z-score standardize a single protein series (ddof=1, NaN-safe).

    Args:
        series: Protein intensity series.

    Returns:
        Standardized series aligned to the original index.
    """
    orig_idx = series.index
    series = series.loc[~pd.isnull(series)]
    return ((series - series.mean()) / series.std(ddof=1)).reindex(orig_idx)
