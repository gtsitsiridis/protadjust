from pathlib import Path
import logging
from typing import Protocol

import pandas as pd
import numpy as np
from tqdm import tqdm

from . import utils
from .dataset import ProteomicsDataset

logger = logging.getLogger(__name__)


class ProteomicsAdjuster(Protocol):
    """Protocol satisfied by every concrete adjuster class."""

    def adjust(self, proteomics_path: Path, index_col: str, output_dir: Path) -> ProteomicsDataset:
        """Adjust protein intensities and return a ProteomicsDataset.

        Args:
            proteomics_path: Parquet file with samples as rows and proteins as columns.
            index_col: Name of the sample-identifier column.
            output_dir: Directory for method-specific output files.

        Returns:
            ProteomicsDataset with adjusted data.
        """
        ...



class StandardAdjuster:
    """Z-score standardise each protein independently across samples.

    Proteins with more than 30 % missing values are removed before adjustment.
    """

    def adjust(self, proteomics_path: Path, index_col: str, output_dir: Path) -> ProteomicsDataset:
        """Z-score standardise the proteomics matrix.

        Args:
            proteomics_path: Path to input parquet (samples x proteins).
            index_col: Sample identifier column.
            output_dir: Unused; present for protocol compatibility.

        Returns:
            ProteomicsDataset with z-score normalised values.
        """
        df = pd.read_parquet(proteomics_path, engine='fastparquet', index=False).set_index(index_col)
        df = df.loc[:, np.mean(np.isnan(df), axis=0) < 0.3]
        result = utils.standardize_matrix(df, n_jobs=4)
        return ProteomicsDataset(data=result.values, sample_ids=result.index.tolist(),
                                 protein_ids=result.columns.tolist())


class RINTAdjuster:
    """Rank-based inverse normal transformation (RINT) applied per protein.

    Proteins with more than 30 % missing values are removed before adjustment.
    """

    def adjust(self, proteomics_path: Path, index_col: str, output_dir: Path) -> ProteomicsDataset:
        """Apply RINT to the proteomics matrix.

        Args:
            proteomics_path: Path to input parquet (samples x proteins).
            index_col: Sample identifier column.
            output_dir: Unused; present for protocol compatibility.

        Returns:
            ProteomicsDataset with RINT-transformed values.
        """
        df = pd.read_parquet(proteomics_path, engine='fastparquet', index=False).set_index(index_col)
        df = df.loc[:, np.mean(np.isnan(df), axis=0) < 0.3]
        result = utils.rank_INT_matrix(df, n_jobs=4)
        return ProteomicsDataset(data=result.values, sample_ids=result.index.tolist(),
                                 protein_ids=result.columns.tolist())


class RegressionAdjuster:
    """OLS covariate regression adjustment.

    Regresses out covariates (and optionally proteomics PCs) via OLS and
    returns the residuals. Binary covariates with fewer than 10 samples in
    either group are removed automatically. Constant covariates are also removed.

    Args:
        covariate_path: Parquet file of covariates (samples x covariates).
        covariate_index_col: Sample identifier column in the covariate file.
        n_proteomics_pcs: Number of proteomics PCs to add as extra covariates.
        n_jobs: Parallel workers for per-protein operations.
    """

    def __init__(
        self,
        covariate_path: Path,
        covariate_index_col: str,
        n_proteomics_pcs: int = None,
        n_jobs: int = 4,
    ):
        cov_df = pd.read_parquet(covariate_path, engine='fastparquet', index=False)
        cov_df[covariate_index_col] = cov_df[covariate_index_col].astype(str)
        cov_df = cov_df.set_index(covariate_index_col)

        to_exclude = []
        for col in cov_df.columns:
            counts = cov_df[col].value_counts()
            if len(counts) == 1 or (len(counts) == 2 and any(c < 10 for c in counts)):
                to_exclude.append(col)
        logger.info("Excluding %d low-variability covariates", len(to_exclude))
        cov_df = cov_df.drop(columns=to_exclude)

        self.covariate_df = cov_df
        self.n_proteomics_pcs = n_proteomics_pcs
        self.n_jobs = n_jobs

    def adjust(self, proteomics_path: Path, index_col: str, output_dir: Path) -> ProteomicsDataset:
        """Regress out covariates via OLS and return residuals.

        Args:
            proteomics_path: Path to input parquet (samples x proteins).
            index_col: Sample identifier column.
            output_dir: Per-protein regression stats are saved here.

        Returns:
            ProteomicsDataset with OLS residuals.
        """
        df = pd.read_parquet(proteomics_path, engine='fastparquet', index=False)
        df[index_col] = df[index_col].astype(str)
        df = df.set_index(index_col)
        df = df.loc[:, np.mean(np.isnan(df), axis=0) < 0.3]

        cov_df = self.covariate_df

        if self.n_proteomics_pcs is not None:
            pcs = utils.extract_principal_components(df, n_components=self.n_proteomics_pcs)
            cov_df = pd.concat([cov_df, pcs], axis=1)

        cov_df.to_parquet(output_dir / 'regression_covariates.parquet')

        common = df.index.intersection(cov_df.index)
        logger.info("Common samples: %d | proteomics-only: %d | covariate-only: %d",
                    len(common),
                    len(df.index.difference(cov_df.index)),
                    len(cov_df.index.difference(df.index)))
        df = df.loc[common]
        cov_df = cov_df.loc[common]

        logger.info("Regressing out covariates")
        corrected = utils.regress_out_covariates_matrix(df, cov_df, n_jobs=self.n_jobs,
                                                        output_dir=output_dir)

        return ProteomicsDataset(data=corrected.values, sample_ids=corrected.index.tolist(),
                                 protein_ids=corrected.columns.tolist())


class ProteinRegressionAdjuster:
    """Per-protein OLS regression against a matching gene-specific covariate (e.g. PRS).

    For each protein, a covariate column with the same name is looked up in
    *protein_covariate_path*. Proteins without a matching column are skipped.
    Returns OLS residuals.

    Args:
        protein_covariate_path: Parquet file of per-protein covariates.
        covariate_index_col: Sample identifier column in that file.
        n_jobs: Parallel workers (currently unused; kept for API consistency).
    """

    def __init__(self, protein_covariate_path: Path, covariate_index_col: str, n_jobs: int = 4):
        self.protein_covariate_df = (
            pd.read_parquet(protein_covariate_path, engine='fastparquet', index=False)
            .set_index(covariate_index_col)
        )
        self.n_jobs = n_jobs

    def adjust(self, proteomics_path: Path, index_col: str, output_dir: Path) -> ProteomicsDataset:
        """Regress each protein against its matching per-protein covariate.

        Args:
            proteomics_path: Path to input parquet (samples x proteins).
            index_col: Sample identifier column.
            output_dir: Unused; present for protocol compatibility.

        Returns:
            ProteomicsDataset containing only proteins that had a matching covariate.
        """
        df = pd.read_parquet(proteomics_path, engine='fastparquet', index=False)
        df[index_col] = df[index_col].astype(str)
        df = df.set_index(index_col)
        df = df.loc[:, np.mean(np.isnan(df), axis=0) < 0.3]

        prot_cov = self.protein_covariate_df

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in prot_cov.columns:
            prot_cov[col] = pd.to_numeric(prot_cov[col], errors='coerce')

        common = df.index.intersection(prot_cov.index)
        logger.info("Common samples: %d | proteomics-only: %d | covariate-only: %d",
                    len(common),
                    len(df.index.difference(prot_cov.index)),
                    len(prot_cov.index.difference(df.index)))
        df = df.loc[common]
        prot_cov = prot_cov.loc[common]

        adjusted_proteins = []
        n_with, n_without = 0, 0

        for protein in tqdm(df.columns, desc="Adjusting proteins"):
            if protein not in prot_cov.columns:
                n_without += 1
                if n_without <= 5:
                    logger.warning("No covariate found for %s, skipping", protein)
                continue

            prot_series = df[protein]
            cov_series = prot_cov[protein]
            valid = prot_series.notna() & cov_series.notna()

            if valid.sum() <= 10:
                n_without += 1
                if n_without <= 5:
                    logger.warning("Insufficient valid samples for %s (%d), skipping",
                                   protein, valid.sum())
                continue

            adjusted = utils._regress_out_covariates_series(
                series=prot_series[valid],
                covariates_df=prot_cov.loc[valid, [protein]],
            )
            adjusted_proteins.append(adjusted.reindex(df.index))
            n_with += 1

        logger.info("Proteins adjusted: %d | skipped: %d", n_with, n_without)
        result = pd.concat(adjusted_proteins, axis=1)

        return ProteomicsDataset(data=result.values, sample_ids=result.index.tolist(),
                                 protein_ids=result.columns.tolist())


class ProtriderAdjuster:
    """Autoencoder-based outlier detection via the external *protrider* library.

    Wraps the Protrider autoencoder and exposes its most commonly used
    configuration parameters directly. If ``pval_dist='t'``, the raw
    t-distributed z-scores are transformed to N(0,1) before returning.
    Intermediate outputs (residuals, p-values, log2fc, …) can optionally be
    saved to the output directory.

    Args:
        n_jobs: Number of parallel workers for post-processing steps.
        save_extra_files: If True, write intermediate Protrider outputs
            (autoencoder output, residuals, p-values, adjusted p-values,
            log2fc) to the output directory.
        max_allowed_NAs_per_protein: Proteins with a higher fraction of
            missing values are dropped before the autoencoder runs.
        log_func_name: Name of the log-transformation applied to raw
            intensities inside Protrider (e.g. ``'log2'``). ``None`` skips
            the transform.
        autoencoder_training: Whether to train the autoencoder. Set to
            ``False`` to skip training and use a pre-initialised model.
        n_layers: Number of hidden layers in the autoencoder.
        n_epochs: Maximum number of training epochs.
        lr: Learning rate for autoencoder training.
        find_q_method: Method used to determine the latent dimension
            (e.g. ``'OHT'``).
        init_pca: Initialise autoencoder weights from PCA loadings.
        common_degrees_freedom: If True, fit a single degrees-of-freedom
            parameter shared across all proteins (t-distribution).
        pval_dist: Distribution used by Protrider for p-value computation,
            forwarded directly to ``ProtriderConfig``. Defaults to ``'t'``.
        transform_to_normal: If True (default), apply a probability-integral
            transform to convert the Protrider z-scores to N(0,1). Requires
            ``pval_dist='t'`` so that per-protein degrees of freedom are
            available.
    """

    def __init__(self, n_jobs: int = 4, save_extra_files: bool = False,
                 max_allowed_NAs_per_protein: float = 0.3, log_func_name: str = None,
                 autoencoder_training: bool = True, n_layers: int = 1, n_epochs: int = 1000, lr: float = 0.001,
                 find_q_method: str = 'OHT', init_pca: bool = True, common_degrees_freedom: bool = False,
                 pval_dist: str = 't', transform_to_normal: bool = True):

        self.n_jobs = n_jobs
        self.save_extra_files = save_extra_files
        self.transform_to_normal = transform_to_normal

        self.protrider_params = dict(max_allowed_NAs_per_protein=max_allowed_NAs_per_protein,
                                     log_func_name=log_func_name,
                                     autoencoder_training=autoencoder_training, n_layers=n_layers,
                                     n_epochs=n_epochs, lr=lr,
                                     find_q_method=find_q_method, init_pca=init_pca,
                                     common_degrees_freedom=common_degrees_freedom, pval_dist=pval_dist,
                                     pval_adj='by', pval_sided='two-sided',
                                     outlier_threshold=0.1, verbose=True)

        try:
            import protrider
            self.protrider = protrider
        except ImportError as exc:
            logger.error("protrider is not installed. Install with: pip install protadjust[protrider]")
            raise exc

    def adjust(self, proteomics_path: Path, index_col: str, output_dir: Path) -> ProteomicsDataset:
        """Run the Protrider autoencoder pipeline.

        Args:
            proteomics_path: Path to input parquet (samples x proteins).
            index_col: Sample identifier column.
            output_dir: Directory for optional extra output files.

        Returns:
            ProteomicsDataset with Protrider z-scores. If
            ``transform_to_normal=True`` and ``pval_dist='t'``, z-scores are
            transformed from the t-distribution to N(0,1) via the probability
            integral transform.
        """

        config = self.protrider.ProtriderConfig(
            input_intensities=str(proteomics_path),
            input_format='proteins_as_columns',
            index_col=index_col,
            **self.protrider_params,
        )
        result, model_info, fit_parameters, gs_results = self.protrider.run(config=config)

        if self.save_extra_files:
            logger.info("Saving Protrider outputs to %s", output_dir)
            fit_parameters.to_csv(output_dir)
            if gs_results is not None:
                gs_results.to_csv(output_dir)
            model_info.save(output_dir)
            for name, df_attr in [
                ('output', result.df_out),
                ('residuals', result.df_res),
                ('pvals', result.df_pvals),
                ('pvals_adj', result.df_pvals_adj),
                ('log2fc', result.log2fc),
            ]:
                p = output_dir / f'{name}.parquet'
                df_attr.rename_axis('sample').to_parquet(p)
                logger.info("Saved %s to %s", name, p)

        zscore_df = result.df_Z
        if self.transform_to_normal and self.protrider_params.get('pval_dist') == 't':
            logger.info("Transforming z-scores from t-distribution to N(0,1)")
            zscore_df = utils.t_to_normal_transform_matrix(
                zscore_df, degree_freedom=fit_parameters.degrees_freedoms, n_jobs=self.n_jobs
            )

        return ProteomicsDataset(
            data=zscore_df.values,
            sample_ids=zscore_df.index.tolist(),
            protein_ids=zscore_df.columns.tolist(),
        )
