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
            self.notes_ts_path = os.path.join(data_path, 'notes_ts.h5')
        except FileNotFoundError as e:
            print("Make sure data has been processed: ", e)
            return

        self.demographics.set_index('pat_id', inplace=True)
        self.vitals.set_index(['pat_id', 'hours_in'], inplace=True)
        self.interventions.set_index(['pat_id', 'hours_in'], inplace=True)
        self.labels = self.labels.set_index('pat_id')['Dead']

        with h5py.File(self.notes_static_path, 'r') as f:
            self.nst_ids = set([int(k.split('_')[-1]) for k in list(f.keys())])
        with h5py.File(self.notes_ts_path, 'r') as f:
            self.nts_ids = set([int(k.split('_')[-1]) for k in list(f.keys())])

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

        lbl = self.labels.loc[pat_id].values

        if type(pat_id) != np.ndarray:
            nst, nts, nst_msk, nts_msk = self._getpatient_notes(pat_id)
        else:
            nst, nts, nst_msk, nts_msk = self._getpatients_notes(pat_id)

        return dem, vit, itv, nst, nts, vit_msk, itv_msk, nst_msk, nts_msk, lbl

    def _getpatient_notes(self, pat_id):
        nst, nts = np.zeros(0), np.empty(0)
        nst_msk, nts_msk = np.zeros(1), np.zeros(NOTES_TIME_DIM)

        if pat_id in self.nst_ids:
            with h5py.File(self.notes_static_path, 'r') as f:
                nst = f[f'pat_id_{pat_id}'][:]
                nst_msk[0] = 1

        if pat_id in self.nts_ids:
            with h5py.File(self.notes_ts_path, 'r') as f:
                nts, nts_msk = self._format_notes_ts_group(
                    f[f'pat_id_{pat_id}'])

        return nst, nts, nst_msk, nts_msk

    def _getpatients_notes(self, pat_ids):
        match_nst = set(
            [pat_id for pat_id in pat_ids if pat_id in self.nst_ids])
        nst, nst_msk = [], np.zeros(len(pat_ids))
        with h5py.File(self.notes_static_path, 'r') as f:
            for pidx, pid in enumerate(pat_ids):
                if pid in match_nst:
                    nst.append(f[f'pat_id_{pid}'][:])
                    nst_msk[pidx] = 1

        if len(match_nst):
            nst = padded_stack(nst)
            nst = pad_missing(nst, nst_msk)
        else:
            nst = np.zeros((len(pat_ids), 1))

        match_nts = set(
            [pat_id for pat_id in pat_ids if pat_id in self.nts_ids])
        nts, nts_msk = [], np.zeros((len(pat_ids), NOTES_TIME_DIM))
        with h5py.File(self.notes_ts_path, 'r') as f:
            for pidx, pid in enumerate(pat_ids):
                if pid in match_nts:
                    gnotes, gmask = self._format_notes_ts_group(
                        f[f'pat_id_{pid}'])
                    nts.append(gnotes)
                    nts_msk[pidx] = gmask

        if len(match_nts):
            nts = padded_stack(nts)
            nts = pad_missing(nts, nts_msk.sum(axis=1) > 0)
        else:
            nts = np.zeros((len(pat_ids), NOTES_TIME_DIM, 1))

        return nst, nts, nst_msk, nts_msk

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
        times, notes = [0]*group_size, [0]*group_size
        for d in nts_group.keys():
            _, gidx, _, time = d.split('_')
            gidx, time = int(gidx), int(time)
            times[gidx] = time
            notes[gidx] = nts_group[d][:]

        mask = np.zeros(NOTES_TIME_DIM)
        for idx in times:
            mask[idx] = 1

        notes = padded_stack(notes)
        notes = pad_missing(notes, mask)

        return notes, mask
