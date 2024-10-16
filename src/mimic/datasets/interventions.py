from pathlib import Path

import numpy as np
import pandas as pd

from mimic.datasets.modal_dataset import ModalDataset
from mimic.utils import pad_axis, padded_stack


class InterventionsDataset(ModalDataset):
    def __init__(self, data_path: Path | str, pat_ids: tuple[int], seq_len_pad: int):
        super().__init__(data_path, pat_ids)
        self.seq_len_pad = seq_len_pad

        try:
            self.data: pd.DataFrame = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Provided data path is not a valid .csv file: `{data_path}`"
            )

        self.data.set_index(['pat_id', 'hours_in'], inplace=True)

    def _getitem_single(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch: sequence_length x vital_features (S x E)
        Mask: sequence_length (S)
        """

        # Extract data
        pat_id = self.pat_ids[idx]
        batch = self.data.loc[pat_id]
        num_entries = batch.shape[0]
        batch = pad_axis(batch.values, self.seq_len_pad, 0)

        # Create mask
        mask = np.ones(self.seq_len_pad)
        mask[num_entries:] = 0

        return batch, mask

    def _getitem_multiple(self, idxs: slice) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch: batch_size x sequence_length x vital_features (B x S x E)
        Mask: batch_size x sequence_length (B x S)
        """

        # Extract data
        pat_ids = self.pat_ids[idxs]
        batch = self.data.loc[pat_ids].groupby(level=0).apply(
            lambda x: x.values).values.tolist()
        num_entries_all_pats = [pat.shape[0] for pat in batch]
        batch = padded_stack(*batch, pad_to=[self.seq_len_pad, -1])

        # Create mask
        masks = np.ones((len(pat_ids), self.seq_len_pad))
        for sidx, num_entries in enumerate(num_entries_all_pats):
            masks[sidx, num_entries:] = 0

        return batch, masks
