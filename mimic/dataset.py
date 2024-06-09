from torch.utils.data import Dataset
from mimic.vocab import Vocab
from mimic.utils import pad_axis, padded_stack, pad_missing
import numpy as np
import pandas as pd
import os
import h5py

TIMESERIES_DIM = 150
NOTES_TIME_DIM = 128


class MIMICDataset(Dataset):
    def __init__(self, processed_dir: str, train: bool):
        self.processed_dir = processed_dir

        data_path = os.path.join(processed_dir, ('train' if train else 'test'))
        index_path = os.path.join(
            processed_dir, f"{'train' if train else 'test'}_idxs.npy")

        try:
            self.indexes = np.load(index_path)
            self.demographics = pd.read_csv(
                os.path.join(data_path, 'demographic.csv'))
            self.vitals = pd.read_csv(os.path.join(data_path, 'vitals.csv'))
            self.interventions = pd.read_csv(
                os.path.join(data_path, 'interventions.csv'))
            self.labels = pd.read_csv(os.path.join(data_path, 'labels.csv'))

            self.vocab = Vocab.from_json(
                os.path.join(processed_dir, 'vocab.json'))
            self.notes_static_path = os.path.join(data_path, 'notes_static.h5')
            # self.notes_ts_path = os.path.join(data_path, 'notes_ts.h5')
        except FileNotFoundError as e:
            print("Make sure data has been processed: ", e)
            return

        self.demographics.set_index('pat_id', inplace=True)
        self.vitals.set_index(['pat_id', 'hours_in'], inplace=True)
        self.interventions.set_index(['pat_id', 'hours_in'], inplace=True)
        self.labels = self.labels.set_index('pat_id')['Dead']

        with h5py.File(self.notes_static_path, 'r') as f:
            self.nst_ids = set([int(k.split('_')[-1]) for k in list(f.keys())])
        # with h5py.File(self.notes_ts_path, 'r') as f:
        #     self.nts_ids = set([int(k.split('_')[-1]) for k in list(f.keys())])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item_idx):
        pat_id = self.indexes[item_idx]
        if type(pat_id) == np.ndarray:
            pat_id.sort()

        dem = self.demographics.loc[pat_id]
        dem = dem.values.astype(np.float64)

        vit = self.vitals.loc[pat_id]
        vit, vit_msk = self._format_ts_batch(vit)

        itv = self.interventions.loc[pat_id]
        itv, itv_msk = self._format_ts_batch(itv)

        if type(pat_id) != np.ndarray:
            # nst, nts, nst_msk, nts_msk, lbl = self._getpatient_notes(pat_id)
            nst, nst_msk, lbl = self._getpatient_notes(pat_id)
        else:
            # nst, nts, nst_msk, nts_msk, lbl = self._getpatients_notes(pat_id)
            nst, nst_msk, lbl = self._getpatients_notes(pat_id)

        return dem, vit, itv, nst, vit_msk, itv_msk, nst_msk, lbl

    def _getpatient_notes(self, pat_id):
        # nst, nts = np.zeros(0), np.empty(0)
        # nst_msk, nts_msk = np.zeros(1), np.zeros((NOTES_TIME_DIM, 1))
        nst, nst_msk = np.zeros(), np.zeros(1)

        if pat_id in self.nst_ids:
            with h5py.File(self.notes_static_path, 'r') as f:
                nst = f[f'pat_id_{pat_id}'][:]
                nst_msk = np.ones(len(nst))

        # if pat_id in self.nts_ids:
        #     with h5py.File(self.notes_ts_path, 'r') as f:
        #         nts, nts_msk = self._format_notes_ts_group(
        #             f[f'pat_id_{pat_id}'])

        lbl = self.labels.loc[pat_id]

        return nst, nst_msk, lbl

    def _getpatients_notes(self, pat_ids):
        batch_size = len(pat_ids)

        ### --- Static Notes --- ###

        # Extract the  notes
        match_nst = set(
            [pat_id for pat_id in pat_ids if pat_id in self.nst_ids])
        nst, nst_lens = [], np.zeros(batch_size, dtype=int)
        with h5py.File(self.notes_static_path, 'r') as f:
            for pidx, pid in enumerate(pat_ids):
                if pid in match_nst:
                    nst.append(f[f'pat_id_{pid}'][:])
                    nst_lens[pidx] = len(nst[-1])

        # Pad the notes & create masks
        if len(match_nst):
            nst = padded_stack(nst)
            nst = pad_missing(nst, nst_lens > 0)

            nst_msk = np.zeros((batch_size, nst_lens.max()))
            for idx, nst_len in enumerate(nst_lens):
                nst_msk[idx, :nst_len] = 1

        else:
            nst = np.zeros((batch_size, 1))
            nst_msk = np.zeros((batch_size, 1))

        ### --- Notes time-series --- ###

        # Extract the notes
        # match_nts = set(
        #     [pat_id for pat_id in pat_ids if pat_id in self.nts_ids])
        # nts, nts_msk = [], [np.zeros((NOTES_TIME_DIM, 1))] * batch_size
        # with h5py.File(self.notes_ts_path, 'r') as f:
        #     for pidx, pid in enumerate(pat_ids):
        #         if pid in match_nts:
        #             gnotes, gmask = self._format_notes_ts_group(
        #                 f[f'pat_id_{pid}'])
        #             nts.append(gnotes)
        #             nts_msk[pidx] = gmask

        # nts_msk = padded_stack(nts_msk)

        # Pad the notes
        # if len(match_nts):
        #     nts = padded_stack(nts)
        #     nts = pad_missing(nts, np.sum(nts_msk, axis=(1, 2)) > 0)
        # else:
        #     nts = np.zeros((len(pat_ids), NOTES_TIME_DIM, 1))

        lbl = self.labels.loc[pat_ids].values

        return nst, nst_msk, lbl

    @staticmethod
    def _format_ts_batch(batch_ts_df):
        if batch_ts_df.index.nlevels == 1:
            mask = np.ones(TIMESERIES_DIM)
            mask[batch_ts_df.shape[0]:] = 0

            batch_ts_padded = pad_axis(batch_ts_df.values, TIMESERIES_DIM, 0)
            return batch_ts_padded, mask

        batch_ts = batch_ts_df.groupby(level=0).apply(
            lambda x: x.values).values.tolist()

        masks = np.ones((len(batch_ts), TIMESERIES_DIM))
        for sidx, sample in enumerate(batch_ts):
            masks[sidx, sample.shape[0]:] = 0

        batch_ts = padded_stack(batch_ts, fill_dims=[TIMESERIES_DIM, -1])
        return batch_ts, masks

    @staticmethod
    def _format_notes_static(nst):
        max_len = max([len(note) for note in nst])
        return np.array([np.pad(note, (0, max_len-len(note))) for note in nst])

    @staticmethod
    def _format_notes_ts_group(nts_group):
        group_size = len(nts_group)
        times, notes, lens = [0]*group_size, [0] * \
            group_size, np.zeros(group_size, dtype=int)

        for d in nts_group.keys():
            _, gidx, _, time = d.split('_')
            gidx, time = int(gidx), int(time)
            times[gidx] = time
            notes[gidx] = nts_group[d][:]
            lens[gidx] = len(notes[gidx])

        mask = np.zeros((NOTES_TIME_DIM, lens.max()))
        for t, l in zip(times, lens):
            mask[t, :l] = 1

        notes = padded_stack(notes)
        notes = pad_missing(notes, mask.sum(axis=1) > 0)

        return notes, mask
