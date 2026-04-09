from pathlib import Path
import logging
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


class ProteomicsDataset:
    """Holds adjusted proteomics data (samples x proteins).

    Args:
        data: numpy array of shape (n_samples, n_proteins)
        protein_ids: list of protein identifiers
        sample_ids: list of sample identifiers
    """

    def __init__(self, data, protein_ids, sample_ids):
        self.data = data
        self.protein_ids = protein_ids
        self.sample_ids = sample_ids

    @classmethod
    def from_parquet(cls, proteomics_path: Path, index_col: str = 'sample') -> 'ProteomicsDataset':
        """Load a ProteomicsDataset from a parquet file.

        Args:
            proteomics_path: Path to parquet file (samples x proteins, index_col as sample column)
            index_col: Name of the sample index column

        Returns:
            ProteomicsDataset instance
        """
        if proteomics_path.suffix != '.parquet':
            raise ValueError("proteomics_path must be a parquet file")

        df = pl.read_parquet(proteomics_path)
        data = df.select(pl.exclude(index_col)).to_numpy()
        sample_ids = df.select(index_col).to_series().to_list()
        protein_ids = [col for col in df.columns if col != index_col]

        return cls(data=data, protein_ids=protein_ids, sample_ids=sample_ids)

    def persist(self, path: Path, index_col: str = 'sample'):
        """Save the dataset to a parquet file.

        Args:
            path: Destination path for the parquet file
            index_col: Column name to use for the sample index
        """
        df = pd.DataFrame(self.data, index=self.sample_ids, columns=self.protein_ids)
        df.index.name = index_col
        df.to_parquet(path, index=True)
