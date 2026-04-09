"""Tests for protadjust adjustment methods.

Inspired by the test suite in abprot.
"""
import numpy as np
import pandas as pd
import pytest

from protadjust.adjustment import (
    RegressionAdjuster,
    StandardAdjuster,
    RINTAdjuster,
    ProteinRegressionAdjuster,
)
from protadjust.dataset import ProteomicsDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proteomics(tmp_path, n_samples=20, n_proteins=10, seed=42, suffix=''):
    np.random.seed(seed)
    samples = [f'sample_{i}' for i in range(n_samples)]
    proteins = [f'protein_{i}' for i in range(n_proteins)]
    data = np.random.randn(n_samples, n_proteins)
    df = pd.DataFrame(data, columns=proteins)
    df.insert(0, 'sample', samples)
    path = tmp_path / f'proteomics{suffix}.parquet'
    df.to_parquet(path, index=False)
    return path, samples, proteins


def _make_covariates(tmp_path, samples, seed=42, suffix=''):
    np.random.seed(seed)
    df = pd.DataFrame({
        'eid': samples,
        'age': np.random.randint(40, 70, len(samples)),
        'sex': np.random.choice([0, 1], len(samples)),
    })
    path = tmp_path / f'covariates{suffix}.parquet'
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# StandardAdjuster
# ---------------------------------------------------------------------------

class TestStandardAdjuster:
    def test_init(self):
        assert StandardAdjuster() is not None

    def test_returns_proteomics_dataset(self, tmp_path):
        path, samples, proteins = _make_proteomics(tmp_path)
        result = StandardAdjuster().adjust(path, 'sample', tmp_path)
        assert isinstance(result, ProteomicsDataset)

    def test_output_shape(self, tmp_path):
        n_samples, n_proteins = 20, 10
        path, samples, proteins = _make_proteomics(tmp_path, n_samples, n_proteins)
        result = StandardAdjuster().adjust(path, 'sample', tmp_path)
        assert result.data.shape[0] == n_samples
        assert result.data.shape[1] <= n_proteins
        assert len(result.sample_ids) == n_samples
        assert len(result.protein_ids) <= n_proteins

    def test_normalization(self, tmp_path):
        """Each protein column should have mean≈0 and std≈1 after z-score."""
        n_samples, n_proteins = 100, 20
        np.random.seed(0)
        data = np.random.randn(n_samples, n_proteins) * 10 + 50
        samples = [f's_{i}' for i in range(n_samples)]
        proteins = [f'p_{i}' for i in range(n_proteins)]
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        path = tmp_path / 'prot.parquet'
        df.to_parquet(path, index=False)

        result = StandardAdjuster().adjust(path, 'sample', tmp_path)
        result_df = pd.DataFrame(result.data, columns=result.protein_ids)

        assert np.allclose(result_df.mean(), 0, atol=0.15)
        assert (result_df.std() > 0.9).all() and (result_df.std() < 1.1).all()

    def test_filters_high_missingness_proteins(self, tmp_path):
        n_samples, n_proteins = 20, 10
        np.random.seed(0)
        data = np.random.randn(n_samples, n_proteins)
        data[:16, 0] = np.nan  # 80 % missing → should be filtered
        data[:4, 1] = np.nan   # 20 % missing → kept

        proteins = [f'protein_{i}' for i in range(n_proteins)]
        samples = [f's_{i}' for i in range(n_samples)]
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        path = tmp_path / 'prot.parquet'
        df.to_parquet(path, index=False)

        result = StandardAdjuster().adjust(path, 'sample', tmp_path)
        assert 'protein_0' not in result.protein_ids
        assert 'protein_1' in result.protein_ids

    def test_direct_instantiation(self):
        assert isinstance(StandardAdjuster(), StandardAdjuster)


# ---------------------------------------------------------------------------
# RINTAdjuster
# ---------------------------------------------------------------------------

class TestRINTAdjuster:
    def test_init(self):
        assert RINTAdjuster() is not None

    def test_output_shape(self, tmp_path):
        n_samples, n_proteins = 20, 10
        path, _, _ = _make_proteomics(tmp_path, n_samples, n_proteins)
        result = RINTAdjuster().adjust(path, 'sample', tmp_path)
        assert result.data.shape[0] == n_samples
        assert result.data.shape[1] <= n_proteins

    def test_approximate_normality(self, tmp_path):
        """After RINT each protein should have mean≈0 and std≈1."""
        n_samples, n_proteins = 100, 5
        np.random.seed(1)
        data = np.random.exponential(scale=2.0, size=(n_samples, n_proteins))
        samples = [f's_{i}' for i in range(n_samples)]
        proteins = [f'p_{i}' for i in range(n_proteins)]
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        path = tmp_path / 'prot.parquet'
        df.to_parquet(path, index=False)

        result = RINTAdjuster().adjust(path, 'sample', tmp_path)
        result_df = pd.DataFrame(result.data, columns=result.protein_ids, index=result.sample_ids)

        for protein in result.protein_ids:
            col = result_df[protein].dropna()
            assert np.abs(col.mean()) < 0.2, f"{protein} mean {col.mean():.3f} not close to 0"
            assert 0.8 < col.std() < 1.2, f"{protein} std {col.std():.3f} not close to 1"

    def test_filters_high_missingness_proteins(self, tmp_path):
        n_samples, n_proteins = 20, 10
        np.random.seed(2)
        data = np.random.randn(n_samples, n_proteins)
        data[:16, 0] = np.nan
        data[:4, 1] = np.nan

        proteins = [f'protein_{i}' for i in range(n_proteins)]
        samples = [f's_{i}' for i in range(n_samples)]
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        path = tmp_path / 'prot.parquet'
        df.to_parquet(path, index=False)

        result = RINTAdjuster().adjust(path, 'sample', tmp_path)
        assert 'protein_0' not in result.protein_ids
        assert 'protein_1' in result.protein_ids

    def test_direct_instantiation(self):
        assert isinstance(RINTAdjuster(), RINTAdjuster)


# ---------------------------------------------------------------------------
# RegressionAdjuster
# ---------------------------------------------------------------------------

class TestRegressionAdjuster:
    def test_init_basic(self, tmp_path):
        """Adjuster stores covariates and filters low-variability columns."""
        samples = [str(i) for i in range(12)]
        cov = pd.DataFrame({
            'eid': samples,
            'age': [50, 60, 45, 55, 58, 52, 61, 48, 56, 59, 53, 57],
            'sex': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # filtered: <10 per group
            'bmi': [25.0, 28.0, 22.0, 30.0, 26.5, 27.0, 29.0, 24.0, 31.0, 26.0, 28.5, 25.5],
        })
        cov_path = tmp_path / 'cov.parquet'
        cov.to_parquet(cov_path, index=False)

        adj = RegressionAdjuster(cov_path, 'eid', n_proteomics_pcs=2)
        assert adj.n_proteomics_pcs == 2
        # sex should be filtered (only 12 samples → <10 in one group)
        assert 'age' in adj.covariate_df.columns
        assert 'bmi' in adj.covariate_df.columns

    def test_filters_low_variability_covariates(self, tmp_path):
        samples = [str(i) for i in range(10)]
        cov = pd.DataFrame({
            'eid': samples,
            'age': [50, 60, 45, 55, 58, 52, 61, 48, 56, 59],
            'constant': [1] * 10,          # filtered
            'almost_constant': [1] * 9 + [0],  # filtered (only 1 other value)
            'rare_binary': [0] * 9 + [1],   # filtered (<10 of one value)
        })
        cov_path = tmp_path / 'cov.parquet'
        cov.to_parquet(cov_path, index=False)

        adj = RegressionAdjuster(cov_path, 'eid')
        assert 'constant' not in adj.covariate_df.columns
        assert 'almost_constant' not in adj.covariate_df.columns
        assert 'rare_binary' not in adj.covariate_df.columns
        assert 'age' in adj.covariate_df.columns

    def test_output_format(self, tmp_path):
        n_samples, n_proteins = 20, 10
        prot_path, samples, _ = _make_proteomics(tmp_path, n_samples, n_proteins)
        cov_path = _make_covariates(tmp_path, samples)

        adj = RegressionAdjuster(cov_path, 'eid', n_proteomics_pcs=2)
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)

        assert result.data.shape[0] == n_samples
        assert result.data.shape[1] <= n_proteins
        assert len(result.sample_ids) == n_samples

    def test_filters_high_missingness_proteins(self, tmp_path):
        n_samples, n_proteins = 20, 10
        np.random.seed(3)
        data = np.random.randn(n_samples, n_proteins)
        data[:16, 0] = np.nan  # 80 % → filtered
        data[:4, 1] = np.nan   # 20 % → kept

        proteins = [f'protein_{i}' for i in range(n_proteins)]
        samples = [f's_{i}' for i in range(n_samples)]
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        prot_path = tmp_path / 'prot.parquet'
        df.to_parquet(prot_path, index=False)
        cov_path = _make_covariates(tmp_path, samples)

        adj = RegressionAdjuster(cov_path, 'eid')
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)

        assert 'protein_0' not in result.protein_ids
        assert 'protein_1' in result.protein_ids

    def test_no_pcs(self, tmp_path):
        prot_path, samples, _ = _make_proteomics(tmp_path)
        cov_path = _make_covariates(tmp_path, samples)
        adj = RegressionAdjuster(cov_path, 'eid', n_proteomics_pcs=None)
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)
        assert result.data.shape[0] == 20

    def test_direct_instantiation(self, tmp_path):
        cov_path = _make_covariates(tmp_path, ['1', '2'])
        assert isinstance(RegressionAdjuster(cov_path, 'eid'), RegressionAdjuster)


# ---------------------------------------------------------------------------
# ProteinRegressionAdjuster
# ---------------------------------------------------------------------------

class TestProteinRegressionAdjuster:
    def _make_prs(self, tmp_path, samples, proteins, seed=10):
        np.random.seed(seed)
        data = np.random.randn(len(samples), len(proteins))
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        path = tmp_path / 'prs.parquet'
        df.to_parquet(path, index=False)
        return path

    def test_init(self, tmp_path):
        samples = [f's_{i}' for i in range(15)]
        proteins = ['APOE', 'LDLR', 'PCSK9']
        prs_path = self._make_prs(tmp_path, samples, proteins)

        adj = ProteinRegressionAdjuster(prs_path, 'sample', n_jobs=2)
        assert adj.n_jobs == 2
        assert adj.protein_covariate_df.shape == (15, 3)
        assert list(adj.protein_covariate_df.columns) == proteins

    def test_output_format(self, tmp_path):
        n_samples = 20
        proteins = ['APOE', 'LDLR', 'PCSK9', 'CETP', 'APOB']
        samples = [f's_{i}' for i in range(n_samples)]

        np.random.seed(5)
        data = np.random.randn(n_samples, len(proteins)) * 10 + 50
        df = pd.DataFrame(data, columns=proteins)
        df.insert(0, 'sample', samples)
        prot_path = tmp_path / 'prot.parquet'
        df.to_parquet(prot_path, index=False)

        prs_path = self._make_prs(tmp_path, samples, proteins)

        adj = ProteinRegressionAdjuster(prs_path, 'sample')
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)

        assert result.data.shape[0] == n_samples
        assert result.data.shape[1] == len(proteins)
        assert set(result.protein_ids) == set(proteins)

    def test_regression_removes_covariate_effect(self, tmp_path):
        """Correlation between protein and its PRS should drop after adjustment."""
        n_samples = 100
        samples = [f's_{i}' for i in range(n_samples)]
        protein = 'APOE'

        np.random.seed(7)
        prs = np.random.randn(n_samples)
        prot_values = 2 * prs + np.random.randn(n_samples) * 0.1  # strong correlation

        df_prot = pd.DataFrame({'sample': samples, protein: prot_values})
        prot_path = tmp_path / 'prot.parquet'
        df_prot.to_parquet(prot_path, index=False)

        df_prs = pd.DataFrame({'sample': samples, protein: prs})
        prs_path = tmp_path / 'prs.parquet'
        df_prs.to_parquet(prs_path, index=False)

        corr_before = np.corrcoef(prot_values, prs)[0, 1]
        assert corr_before > 0.9

        adj = ProteinRegressionAdjuster(prs_path, 'sample')
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)

        corr_after = np.corrcoef(result.data[:, 0], prs)[0, 1]
        assert np.abs(corr_after) < 0.3, f"Correlation after adjustment too high: {corr_after:.3f}"

    def test_skips_proteins_without_covariate(self, tmp_path):
        n_samples = 20
        samples = [f's_{i}' for i in range(n_samples)]
        prot_proteins = ['APOE', 'LDLR', 'NO_MATCH']
        prs_proteins = ['APOE', 'LDLR']

        np.random.seed(8)
        df_prot = pd.DataFrame(np.random.randn(n_samples, 3), columns=prot_proteins)
        df_prot.insert(0, 'sample', samples)
        prot_path = tmp_path / 'prot.parquet'
        df_prot.to_parquet(prot_path, index=False)

        df_prs = pd.DataFrame(np.random.randn(n_samples, 2), columns=prs_proteins)
        df_prs.insert(0, 'sample', samples)
        prs_path = tmp_path / 'prs.parquet'
        df_prs.to_parquet(prs_path, index=False)

        adj = ProteinRegressionAdjuster(prs_path, 'sample')
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)

        assert set(result.protein_ids) == set(prs_proteins)
        assert 'NO_MATCH' not in result.protein_ids

    def test_missing_values_preserved(self, tmp_path):
        n_samples = 30
        samples = [f's_{i}' for i in range(n_samples)]
        protein = 'APOE'

        np.random.seed(9)
        prs = np.random.randn(n_samples)
        prot_values = np.random.randn(n_samples) * 10 + 50
        prot_values[:5] = np.nan  # first 5 are missing

        df_prot = pd.DataFrame({'sample': samples, protein: prot_values})
        prot_path = tmp_path / 'prot.parquet'
        df_prot.to_parquet(prot_path, index=False)

        df_prs = pd.DataFrame({'sample': samples, protein: prs})
        prs_path = tmp_path / 'prs.parquet'
        df_prs.to_parquet(prs_path, index=False)

        adj = ProteinRegressionAdjuster(prs_path, 'sample')
        out = tmp_path / 'out'
        out.mkdir()
        result = adj.adjust(prot_path, 'sample', out)

        assert result.data.shape == (n_samples, 1)
        assert np.isnan(result.data[:5, 0]).all()
        assert not np.isnan(result.data[5:, 0]).any()

    def test_direct_instantiation(self, tmp_path):
        prs_path = self._make_prs(tmp_path, ['s_0', 's_1'], ['APOE'])
        assert isinstance(ProteinRegressionAdjuster(prs_path, 'sample'), ProteinRegressionAdjuster)


# ---------------------------------------------------------------------------
# ProteomicsDataset
# ---------------------------------------------------------------------------

class TestProteomicsDataset:
    def test_persist_and_reload(self, tmp_path):
        np.random.seed(0)
        data = np.random.randn(10, 5)
        ds = ProteomicsDataset(data=data,
                               sample_ids=[f's{i}' for i in range(10)],
                               protein_ids=[f'p{i}' for i in range(5)])
        out = tmp_path / 'out.parquet'
        ds.persist(out, index_col='sample')
        ds2 = ProteomicsDataset.from_parquet(out, index_col='sample')

        assert ds2.data.shape == data.shape
        assert ds2.sample_ids == ds.sample_ids
        assert ds2.protein_ids == ds.protein_ids
        assert np.allclose(ds2.data, data)
