from pathlib import Path

import numpy as np
import pandas as pd

from mimic.datasets.modal_dataset import ModalDataset

"""
Dataset Description:
- Gender: bool
- Age: float64
- Admission Type: int64
- Ethnicity: OneHotEncoded
- Insurance: OneHotEncoded
- FCU: OneHotEncoded
"""


class DemographicDataset(ModalDataset):
    def __init__(self, data_path: Path | str, pat_ids: tuple[int]):
        super().__init__(data_path, pat_ids)

        try:
            self.data: pd.DataFrame = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Provided data path is not a valid .csv file: `{data_path}`"
            )

        self.data.set_index('pat_id', inplace=True)

    def _getitem_single(self, idx: int) -> np.ndarray:
        pat_id = self.pat_ids[idx]
        return self.data.loc[pat_id].values.astype(np.float64)

    def _getitem_multiple(self, idxs: slice | np.ndarray) -> np.ndarray:
        pat_ids = self.pat_ids[idxs]
        return self.data.loc[pat_ids].values.astype(np.float64)
