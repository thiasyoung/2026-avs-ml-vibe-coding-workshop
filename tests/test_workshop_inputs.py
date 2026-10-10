"""Small data-contract checks; these never execute notebooks or train models."""

from __future__ import annotations

import unittest
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


class WorkshopInputTests(unittest.TestCase):
    def assert_parquet_columns(self, relative_path: str, required: set[str]) -> None:
        path = ROOT / relative_path
        self.assertTrue(path.is_file(), f"Missing tracked workshop input: {relative_path}")
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        self.assertGreater(parquet_file.metadata.num_rows, 0)
        self.assertTrue(required.issubset(set(schema.names)), schema.names)

    def test_boiling_point_input_contract(self) -> None:
        self.assert_parquet_columns(
            "01_boiling_point/data/CompoundData.parquet", {"SMILES", "BP"}
        )

    def test_ir_input_contract(self) -> None:
        self.assert_parquet_columns(
            "02_ir_functional_groups/data/IR_v09_df_trimmed_500_points.parquet",
            {"x", "y", "NIST_molecule"},
        )

    def test_htem_input_contract(self) -> None:
        self.assert_parquet_columns(
            "03_htem_conductivity/data/Filtered_Libraries_Data_wTauc_interpolated_trans.parquet",
            {"interpolated_transmittances", "log10_conductivity"},
        )


if __name__ == "__main__":
    unittest.main()
