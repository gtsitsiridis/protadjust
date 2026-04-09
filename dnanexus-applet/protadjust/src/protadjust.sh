#!/bin/bash
# protadjust 0.0.1

set -euxo pipefail

main() {
    IMAGE="ghcr.io/gtsitsiridis/protadjust:latest"

    echo "Pulling Docker image $IMAGE ..."
    docker pull "$IMAGE"

    # --- download inputs ---
    dx download "$input_parquet" -o input.parquet

    mkdir -p output

    # --- build command ---
    CMD="-v $method /data/input.parquet /data/output/ --index-col ${index_col:-sample}"

    case "$method" in
        regression)
            if [ -z "${covariates_parquet:-}" ]; then
                dx-jobutil-report-error "covariates_parquet is required for the regression method"
            fi
            dx download "$covariates_parquet" -o covariates.parquet
            CMD="$CMD --covariate-path /data/covariates.parquet"
            CMD="$CMD --covariate-index-col ${covariate_index_col:-eid}"
            if [ -n "${n_pcs:-}" ]; then
                CMD="$CMD --n-pcs $n_pcs"
            fi
            ;;
        protein-regression)
            if [ -z "${protein_covariates_parquet:-}" ]; then
                dx-jobutil-report-error "protein_covariates_parquet is required for the protein-regression method"
            fi
            dx download "$protein_covariates_parquet" -o protein_covariates.parquet
            CMD="$CMD --protein-covariate-path /data/protein_covariates.parquet"
            ;;
        protrider)
            CMD="$CMD --pval-dist ${pval_dist:-t}"
            CMD="$CMD --max-nas ${max_nas:-0.3}"
            CMD="$CMD --n-layers ${n_layers:-1}"
            CMD="$CMD --n-epochs ${n_epochs:-1000}"
            CMD="$CMD --lr ${lr:-0.001}"
            CMD="$CMD --find-q-method ${find_q_method:-OHT}"
            if [ "${no_autoencoder_training:-false}" = "true" ]; then
                CMD="$CMD --no-autoencoder-training"
            fi
            if [ "${no_init_pca:-false}" = "true" ]; then
                CMD="$CMD --no-init-pca"
            fi
            if [ "${common_degrees_freedom:-false}" = "true" ]; then
                CMD="$CMD --common-degrees-freedom"
            fi
            ;;
    esac

    # --- run ---
    docker run --rm \
        -v "$PWD:/data" \
        "$IMAGE" \
        $CMD

    # --- upload output ---
    output_parquet=$(dx upload output/adjusted_proteomics.parquet --brief)
    dx-jobutil-add-output output_parquet "$output_parquet" --class=file
}
