"""Command-line interface for protadjust."""
import logging
from pathlib import Path

import click

from .adjustment import StandardAdjuster, RINTAdjuster, RegressionAdjuster, ProteinRegressionAdjuster, ProtriderAdjuster, ProteomicsAdjuster

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared options (used by multiple commands)
# ---------------------------------------------------------------------------

_INPUT_PATH = click.argument('input_path', type=click.Path(exists=True, path_type=Path))
_OUTPUT_DIR = click.argument('output_dir', type=click.Path(path_type=Path))
_INDEX_COL = click.option(
    '--index-col', default='sample', show_default=True,
    help='Sample identifier column in the input parquet.',
)
_N_JOBS = click.option(
    '--n-jobs', type=int, default=4, show_default=True,
    help='Number of parallel workers.',
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(adjuster: ProteomicsAdjuster, input_path: Path, output_dir: Path, index_col: str):
    """Run adjuster, persist result, print summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running %s adjustment on %s with index column '%s'", adjuster.__class__.__name__, input_path, index_col)
    try:
        dataset = adjuster.adjust(input_path, index_col, output_dir)
        out_path = output_dir / 'adjusted_proteomics.parquet'
        dataset.persist(out_path, index_col=index_col)
        click.echo(f"Saved adjusted proteomics to {out_path}")
        click.echo(f"Shape: {dataset.data.shape[0]} samples x {dataset.data.shape[1]} proteins")
    except Exception as exc:
        logger.exception("Adjustment failed: %s", exc)
        raise click.Abort() from exc


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG).')
@click.pass_context
def cli(ctx, verbose: int):
    """protadjust – protein intensity adjustment tool.

    Adjust a proteomics parquet matrix (samples × proteins) using one of
    several statistical methods and write the result to an output directory
    as adjusted_proteomics.parquet.
    """
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
    )
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# z-score
# ---------------------------------------------------------------------------

@cli.command('z-score')
@_INPUT_PATH
@_OUTPUT_DIR
@_INDEX_COL
def zscore(input_path: Path, output_dir: Path, index_col: str):
    """Z-score standardise each protein across samples (mean=0, std=1).

    Proteins with more than 30 % missing values are removed before adjustment.
    """
    _run(StandardAdjuster(), input_path, output_dir, index_col)


# ---------------------------------------------------------------------------
# rint
# ---------------------------------------------------------------------------

@cli.command()
@_INPUT_PATH
@_OUTPUT_DIR
@_INDEX_COL
def rint(input_path: Path, output_dir: Path, index_col: str):
    """Rank-based inverse normal transformation (RINT) per protein.

    Maps each protein's intensity distribution to N(0,1) via rank-based
    quantile normalisation (Blom's formula). Ties are broken stochastically.
    Proteins with more than 30 % missing values are removed before adjustment.
    """
    _run(RINTAdjuster(), input_path, output_dir, index_col)


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------

@cli.command()
@_INPUT_PATH
@_OUTPUT_DIR
@_INDEX_COL
@_N_JOBS
@click.option('--covariate-path', required=True, type=click.Path(exists=True, path_type=Path),
              help='Parquet file of covariates (samples x covariates).')
@click.option(
    '--covariate-index-col', default='eid', show_default=True,
    help='Sample identifier column in the covariate file.')
@click.option('--n-pcs', type=int, default=None,
              help='Number of proteomics PCs to include as extra covariates.')
@click.option('--normalization', type=click.Choice(['rint', 'standard']), default='rint',
              show_default=True, help='Normalisation applied before and after regression.')
def regression(input_path: Path, output_dir: Path, covariate_path: Path,
               covariate_index_col: str, n_pcs: int, normalization: str,
               index_col: str, n_jobs: int):
    """Three-step covariate regression adjustment.

    1. Normalise raw intensities (RINT or z-score).
    2. Regress out covariates via OLS (optionally including proteomics PCs).
    3. Normalise residuals again.

    Per-protein regression statistics are saved to OUTPUT_DIR. Covariates with
    low variability or insufficient group sizes are filtered automatically.
    """
    adjuster = RegressionAdjuster(
        covariate_path=covariate_path,
        covariate_index_col=covariate_index_col,
        n_proteomics_pcs=n_pcs,
        normalization=normalization,
        n_jobs=n_jobs,
    )
    _run(adjuster, input_path, output_dir, index_col)


# ---------------------------------------------------------------------------
# protein-regression
# ---------------------------------------------------------------------------

@cli.command('protein-regression')
@_INPUT_PATH
@_OUTPUT_DIR
@_INDEX_COL
@_N_JOBS
@click.option('--protein-covariate-path', required=True,
              type=click.Path(exists=True, path_type=Path),
              help='Per-protein covariate parquet (e.g. PRS scores).')
@click.option(
    '--protein-covariate-index-col', default='sample', show_default=True,
    help='Sample identifier column in the per-protein covariate file.')
def protein_regression(input_path: Path, output_dir: Path, protein_covariate_path: Path,
                       protein_covariate_index_col: str, index_col: str, n_jobs: int):
    """Per-protein OLS regression against a matching gene-specific covariate.

    For each protein, looks up a covariate column with the same name in
    PROTEIN_COVARIATE_PATH (e.g. polygenic risk scores). Proteins without a
    matching column or with fewer than 10 valid samples are skipped. Residuals
    are z-score standardised before returning.
    """
    adjuster = ProteinRegressionAdjuster(
        protein_covariate_path=protein_covariate_path,
        covariate_index_col=protein_covariate_index_col,
        n_jobs=n_jobs,
    )
    _run(adjuster, input_path, output_dir, index_col)


# ---------------------------------------------------------------------------
# protrider
# ---------------------------------------------------------------------------

@cli.command()
@_INPUT_PATH
@_OUTPUT_DIR
@_INDEX_COL
@_N_JOBS
@click.option('--save-extra-files', is_flag=True, default=False,
              help='Persist intermediate Protrider outputs (residuals, p-values, log2fc, …).')
@click.option('--pval-dist', type=click.Choice(['t', 't-raw', 'normal']), default='t',
              show_default=True,
              help=(
                  't: fit t-distribution and transform z-scores to N(0,1) (default). '
                  't-raw: fit t-distribution, return raw t z-scores. '
                  'normal: fit normal distribution.'
              ))
@click.option('--max-nas', 'max_allowed_NAs_per_protein', type=float, default=0.3,
              show_default=True, help='Maximum fraction of missing values allowed per protein.')
@click.option('--log-func', 'log_func_name', type=str, default=None,
              help='Log-transformation applied to raw intensities inside Protrider (e.g. log2).')
@click.option('--no-autoencoder-training', 'autoencoder_training', is_flag=True, default=True,
              flag_value=False, help='Skip autoencoder training and use PCA initialisation only.')
@click.option('--n-layers', type=int, default=1, show_default=True,
              help='Number of hidden layers in the autoencoder.')
@click.option('--n-epochs', type=int, default=1000, show_default=True,
              help='Maximum number of training epochs.')
@click.option('--lr', type=float, default=0.001, show_default=True,
              help='Learning rate for autoencoder training.')
@click.option('--find-q-method', type=str, default='OHT', show_default=True,
              help='Method used to determine the latent dimension.')
@click.option('--no-init-pca', 'init_pca', is_flag=True, default=True,
              flag_value=False, help='Disable PCA initialisation of autoencoder weights.')
@click.option('--common-degrees-freedom', is_flag=True, default=False,
              help='Fit a single degrees-of-freedom parameter shared across all proteins.')
def protrider(input_path: Path, output_dir: Path, index_col: str, n_jobs: int,
              save_extra_files: bool, pval_dist: str,
              max_allowed_NAs_per_protein: float, log_func_name: str,
              autoencoder_training: bool, n_layers: int, n_epochs: int, lr: float,
              find_q_method: str, init_pca: bool, common_degrees_freedom: bool):
    """Autoencoder-based outlier detection via the Protrider library.

    Fits a low-rank autoencoder to model expected protein intensities and
    returns per-sample z-scores. With --pval-dist t (default), z-scores are
    computed under a t-distribution and transformed to N(0,1) via the
    probability-integral transform. Use --pval-dist t-raw to skip that
    transform, or --pval-dist normal to fit a normal distribution instead.

    Requires the optional [protrider] extra: pip install protadjust[protrider]
    """
    _run(ProtriderAdjuster(
             n_jobs=n_jobs,
             save_extra_files=save_extra_files,
             pval_dist='t' if pval_dist in ('t', 't-raw') else 'normal',
             transform_to_normal=(pval_dist == 't'),
             max_allowed_NAs_per_protein=max_allowed_NAs_per_protein,
             log_func_name=log_func_name,
             autoencoder_training=autoencoder_training,
             n_layers=n_layers,
             n_epochs=n_epochs,
             lr=lr,
             find_q_method=find_q_method,
             init_pca=init_pca,
             common_degrees_freedom=common_degrees_freedom,
         ),
         input_path, output_dir, index_col)
