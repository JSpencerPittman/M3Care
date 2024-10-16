from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

Batch = np.ndarray | tuple[np.ndarray, np.ndarray]


class ModalDataset(ABC, Dataset):
    def __init__(self, data_path: Path | str, pat_ids: tuple[int]):
        # Verify the data path
        if not (data_path := Path(data_path)).exists():
            raise FileNotFoundError(
                f"DemographicDataset: Dataset path does not exist: `{data_path}`."
            )

        self.data_path: Path = data_path
        self.pat_ids: tuple[int] = pat_ids

    def __len__(self) -> int:
        return len(self.pat_ids)

    def __getitem__(self, loc: int | slice) -> Batch:
        if isinstance(loc, int):
            return self._getitem_single(loc)
        elif isinstance(loc, slice):
            return self._getitem_multiple(loc)
        else:
            raise TypeError(
                f"ModalDataset: Expected int or slice not `{type(loc)}`."
            )

    @abstractmethod
    def _getitem_single(self, idx: int) -> Batch:
        pass

    @abstractmethod
    def _getitem_multiple(self, idxs: slice) -> Batch:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(len={self.__len__()})"

    def __repr__(self) -> str:
        return self.__str__()
