from pathlib import Path

import h5py
import numpy as np

from mimic.datasets.modal_dataset import ModalDataset
from mimic.utils import padded_stack


class StaticNotesDataset(ModalDataset):
    def __init__(self, data_path: Path | str, pat_ids: tuple[int]):
        super().__init__(data_path, pat_ids)

        with h5py.File(self.data_path, 'r') as f:
            self.existing_ids = set([int(k.split('_')[-1])
                                    for k in list(f.keys())])

    def _getitem_single(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch: sequence_length (S)
        Mask: sequence_length (1)
        """

        pat_id = self.pat_ids[idx]

        batch, mask = np.zeros(1), np.zeros(1)

        if pat_id in self.existing_ids:
            with h5py.File(self.data_path, 'r') as f:
                batch = f[f'pat_id_{pat_id}']['discharge'][:]
                mask = np.ones(1)

        return batch, mask

    def _getitem_multiple(self, idxs: slice) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch: batch_size x sequence_length (B x S)
        Mask: batch_size x sequence_length (B)
        """

        pat_ids = self.pat_ids[idxs]
        batch_size = len(pat_ids)
        # Find all patient ids with statics notes.
        matched_ids = set([pat_id
                           for pat_id in pat_ids
                           if pat_id in self.existing_ids])

        if len(matched_ids):
            # Extract batch
            batch = []
            with h5py.File(self.data_path, 'r') as f:
                for pat_id in pat_ids:
                    if pat_id in matched_ids:
                        batch.append(f[f'pat_id_{pat_id}']['discharge'][:])
                    else:
                        batch.append(np.zeros(0))
            batch = padded_stack(*batch)

            # Construct mask
            mask = ((batch != 0).sum(axis=1) > 0).astype(np.float64)

        else:
            # No matches
            batch = np.zeros((batch_size, 1))
            mask = np.zeros(batch_size)

        return batch, mask
