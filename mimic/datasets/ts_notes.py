from pathlib import Path

import h5py
import numpy as np

from mimic.datasets.modal_dataset import ModalDataset
from mimic.utils import padded_stack


class TSNotesDataset(ModalDataset):
    def __init__(self,
                 data_path: Path | str,
                 pat_ids: tuple[int],
                 time_dim: int):
        super().__init__(data_path, pat_ids)
        self.time_dim = time_dim

        with h5py.File(self.data_path, 'r') as f:
            self.existing_ids = set([int(k.split('_')[-1])
                                    for k in list(f.keys())])

    def _getitem_single(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample: time_dim x sequence_length (T x S)
        Mask: time_dim (T)
        """

        pat_id = self.pat_ids[idx]

        batch, mask = np.zeros(1), np.zeros(1)

        if pat_id in self.existing_ids:
            with h5py.File(self.data_path, 'r') as f:
                batch, mask = self._h5_group_numpy_parse(f[f'pat_id_{pat_id}'])

        return batch, mask

    def _getitem_multiple(self, idxs: slice) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample: batch_size x time_dim x sequence_length (B x T x S)
        Mask: batch_size x time_dim x sequence_length (B x T)
        """

        pat_ids = self.pat_ids[idxs]

        batch, mask = [], []

        for pat_id in pat_ids:
            if pat_id in self.existing_ids:
                with h5py.File(self.data_path, 'r') as f:
                    sample, sample_mask = self._h5_group_numpy_parse(
                        f[f'pat_id_{pat_id}'])
                    batch.append(sample)
                    mask.append(sample_mask)
            else:
                batch.append(np.zeros((self.time_dim, 1)))
                mask.append(np.zeros(self.time_dim))

        batch = padded_stack(*batch)
        mask = np.stack(mask)

        return batch, mask

    def _h5_group_numpy_parse(self, group: h5py.Group):
        """
        Sample: time_dim x sequence_length (T x S)
        Mask: time_dim x sequence_length (T)
        """

        notes, times = [], []

        # Sort keys by time.
        keys = sorted(group.keys())
        for key in keys:
            notes.append(group[key][:])
            times.append(int(key.split('_')[-1]))

        note_dim = notes[0].shape[-1]
        sample = np.zeros((self.time_dim, note_dim))
        mask = np.zeros(self.time_dim)

        for note, time in zip(notes, times):
            sample[time] = note
            mask[time] = 1

        return sample, mask
