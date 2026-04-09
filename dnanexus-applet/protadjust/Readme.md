<!-- dx-header -->
# ProtAdjust (DNAnexus Platform App)

Adjust protein abundance using z-score standardisation, rank-based inverse normal transformation (RINT), OLS covariate regression, per-protein regression, or Protrider autoencoder-based outlier detection.
<!-- /dx-header -->

## Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `method` | string | Yes | Adjustment method: `z-score`, `rint`, `regression`, `protein-regression`, or `protrider` |
| `input_parquet` | file | Yes | Proteomics parquet file — rows = samples, columns = proteins |
| `index_col` | string | No | Sample identifier column in the input file (default: `sample`) |
| `covariates_parquet` | file | regression only | Covariates parquet — rows = samples, columns = covariates |
| `covariate_index_col` | string | No | Sample identifier column in the covariates file (default: `eid`) |
| `n_pcs` | int | No | Number of proteomics PCs to add as extra covariates (regression only) |
| `protein_covariates_parquet` | file | protein-regression only | Per-protein covariates parquet, e.g. PRS scores — columns must match protein names |
| `pval_dist` | string | No | P-value distribution for Protrider: `t` (default), `t-raw`, or `normal` |
| `max_nas` | float | No | Max missing value fraction per protein before dropping (protrider, default: `0.3`) |
| `n_layers` | int | No | Number of autoencoder hidden layers (protrider, default: `1`) |
| `n_epochs` | int | No | Maximum training epochs (protrider, default: `1000`) |
| `lr` | float | No | Autoencoder learning rate (protrider, default: `0.001`) |
| `find_q_method` | string | No | Method to determine the latent dimension (protrider, default: `OHT`) |
| `no_autoencoder_training` | boolean | No | Skip autoencoder training and use PCA initialisation only (protrider) |
| `no_init_pca` | boolean | No | Disable PCA initialisation of autoencoder weights (protrider) |
| `common_degrees_freedom` | boolean | No | Fit a single shared degrees-of-freedom across all proteins (protrider) |

## Output

| Name | Type | Description |
|---|---|---|
| `output_parquet` | file | Adjusted protein abundance values (`adjusted_proteomics.parquet`) |

## Methods

| Method | Description |
|---|---|
| `z-score` | Z-score standardisation per protein |
| `rint` | Rank-based inverse normal transformation (RINT) |
| `regression` | OLS covariate regression — returns residuals |
| `protein-regression` | Per-protein OLS regression against a gene-specific covariate (e.g. PRS) |
| `protrider` | Autoencoder-based outlier detection via Protrider |

## Usage notes

- Input parquet must have samples as rows and proteins as columns, with a sample identifier column (default name: `sample`).
- Proteins with more than 30 % missing values are dropped automatically.
- For `regression`, covariates with low variability or fewer than 10 samples in either group are filtered out automatically. Per-protein regression statistics are included in the output.
- `protrider` expects log-transformed input (e.g. UKB NPX values). For mass spectrometry data on a linear scale (e.g. iBAQ), log-transform before passing to the applet.

## Instance type

The default instance is `mem1_ssd1_v2_x16` (16 cores, ~120 GB RAM). For large datasets or `protrider`, consider upgrading to `mem2_ssd1_v2_x32`.
