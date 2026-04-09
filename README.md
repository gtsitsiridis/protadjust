# protadjust

Protein abundance adjustment tool.

## Methods

| Command | Description |
|---|---|
| `z-score` | Z-score standardisation per protein |
| `rint` | Rank-based inverse normal transformation (RINT) |
| `regression` | Three-step covariate regression (normalise → OLS → normalise) |
| `protein-regression` | Per-protein regression against a gene-specific covariate (e.g. PRS) |
| `protrider` | Autoencoder-based outlier detection via [Protrider](https://github.com/gagneurlab/PROTRIDER) |

## Installation

```bash
# clone and install
git clone <repo-url>
cd protadjust
uv sync
```

With the optional Protrider method:

```bash
uv sync --extra protrider
```

## Usage

```bash
protadjust z-score input.parquet output/
protadjust rint input.parquet output/

protadjust regression input.parquet output/ \
    --covariate-path covariates.parquet

protadjust protein-regression input.parquet output/ \
    --protein-covariate-path prs.parquet

protadjust protrider input.parquet output/
```

The `protrider` command accepts a `--log-func` option for the log-transformation applied to raw intensities before the autoencoder runs. For already log-transformed data (e.g. NPX values from UK Biobank), leave it at the default (`None`, no transform). For mass spectrometry data (e.g. iBAQ), specify a transform such as `--log-func log`.

Input parquet: rows = samples, columns = proteins (plus a sample identifier column, default `sample`).
Output: `output/adjusted_proteomics.parquet`.

Use `-v` / `-vv` for INFO / DEBUG logging.

## Docker

```bash
docker build -t protadjust .
mkdir -p output/
docker run \
  -v $PWD/sample_data:/data/input:ro \
  -v $PWD/output:/data/output \
  protadjust rint /data/input/random_proteomics.parquet /data/output/
```

## Development

```bash
uv sync
uv run pytest tests/ -v
```

## License

MIT
