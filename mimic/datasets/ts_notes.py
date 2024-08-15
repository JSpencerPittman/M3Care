from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mimic.datasets.modal_dataset import ModalDataset
from mimic.utils import padded_stack
from mimic.vocab import Vocab


class TSNotesDataset(ModalDataset):
    def __init__(self, data_path: Path | str, pat_ids: tuple[int], vocab: Vocab, time_dim: int):
        super().__init__(data_path, pat_ids)
        self.vocab = vocab
        self.time_dim = time_dim

        with h5py.File(self.data_path, 'r') as f:
            self.existing_ids = set([int(k.split('_')[-1])
                                    for k in list(f.keys())])

    def _getitem_single(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample: time_dim x sequence_length (T x S)
        Mask: time_dim x sequence_length (T x S)
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
        Mask: batch_size x time_dim x sequence_length (B x T x S)
        """

        pat_ids = self.pat_ids[idxs]

        batch, mask, max_seq_len = [], [], 0

        for pat_id in pat_ids:
            if pat_id in self.existing_ids:
                with h5py.File(self.data_path, 'r') as f:
                    sample, sample_mask = self._h5_group_numpy_parse(
                        f[f'pat_id_{pat_id}'])
                    batch.append(sample)
                    mask.append(sample_mask)
                    max_seq_len = max(max_seq_len, sample.shape[-1])
            else:
                batch.append(np.zeros((0, 0)))
                mask.append(np.zeros((0, 0)))

        batch = padded_stack(*batch, pad_to=(-1, max_seq_len))
        mask = padded_stack(*mask, pad_to=(-1, max_seq_len))

        return batch, mask

    def _h5_group_numpy_parse(self, group: h5py.Group):
        """
        Sample: time_dim x sequence_length (T x S)
        Mask: time_dim x sequence_length (T x S)
        """

        notes, times = [], []

        # group_key = 'group_<group-idx>_time_<time>'
        # Sort keys by group index.
        group_keys = sorted(group.keys(),
                            key=lambda key: int(key.split('_')[1]))
        for group_key in group_keys:
            notes.append(group[group_key][:])
            times.append(int(group_key.split('_')[-1]))

        max_len = max(len(note) for note in notes)
        sample = np.zeros((self.time_dim, max_len))
        mask = np.zeros_like(sample)

        for note, time in zip(notes, times):
            sample[time, :len(note)] = note
            mask[time, :len(note)] = 1

        return sample, mask
