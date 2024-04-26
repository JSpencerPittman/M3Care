from torch.utils.data import Dataset
from vocab import Vocab
import numpy as np
import pandas as pd
import os
import h5py


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

        with h5py.File(self.notes_static_path, 'r') as f:
            self.nst_ids = set([int(k.split('_')[-1]) for k in list(f.keys())])
        with h5py.File(self.notes_ts_path, 'r') as f:
            self.nts_ids = set([int(k.split('_')[-1]) for k in list(f.keys())])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item_idx):
        pat_id = self.indexes[item_idx]

        dem = self.demographics.loc[pat_id]
        dem = np.array(dem.values).astype(np.float64)

        vit = self.vitals.loc[pat_id]
        vit = self._format_ts_batch(vit)

        itv = self.interventions.loc[pat_id]
        itv = self._format_ts_batch(itv)

        if type(pat_id) != np.ndarray:
            nst, nts, missing = self._getpatient_notes(pat_id)
        else:
            nst, nts, missing = self._getpatients_notes(pat_id)

        return dem, vit, itv, nst, nts, missing

    def _getpatient_notes(self, pat_id):
        nst, nts = np.empty(0), (np.empty(0), np.empty(0), np.empty(0))
        missing = [True, True]

        if pat_id in self.nst_ids:
            with h5py.File(self.notes_static_path, 'r') as f:
                nst = f[f'row_{pat_id}'][:]
                missing[0] = False

        if pat_id in self.nts_ids:
            with h5py.File(self.notes_ts_path, 'r') as f:
                nts = self._format_notes_ts_group(f[f'pat_id_{pat_id}'])
                missing[1] = False

        return nst, nts, missing

    def _getpatients_notes(self, pat_ids):
        nst, nts = [], []
        missing = []

        missing_st = []
        match_ids = set(
            [pat_id for pat_id in pat_ids if pat_id in self.nst_ids])
        with h5py.File(self.notes_static_path, 'r') as f:
            for pat_id in pat_ids:
                if pat_id in match_ids:
                    nst.append(f[f'row_{pat_id}'][:])
                    missing_st.append(False)
                else:
                    missing_st.append(True)

        missing_ts = []
        match_ids = set(
            [pat_id for pat_id in pat_ids if pat_id in self.nts_ids])
        with h5py.File(self.notes_ts_path, 'r') as f:
            for pat_id in pat_ids:
                if pat_id in match_ids:
                    nts.append(self._format_notes_ts_group(
                        f[f'pat_id_{pat_id}']))
                    missing_ts.append(False)
                else:
                    missing_ts.append(True)

        nst = self._format_notes_static(nst)
        missing = np.array(list(zip(missing_st, missing_ts)))

        return nst, nts, missing

    @staticmethod
    def _format_ts_batch(batch_ts_df):
        if batch_ts_df.index.nlevels == 1:
            return batch_ts_df.values

        batch_ts = batch_ts_df.groupby(level=0).apply(
            lambda x: x.values).values.tolist()
        max_seq_len = max([seq.shape[0] for seq in batch_ts])

        for i, seq in enumerate(batch_ts):
            null_rows = np.zeros(
                (max_seq_len-seq.shape[0], batch_ts_df.shape[1]))
            batch_ts[i] = np.vstack([seq, null_rows])

        return np.array(batch_ts)

    @staticmethod
    def _format_notes_static(nst):
        max_len = max([len(note) for note in nst])
        return np.array([np.pad(note, (0, max_len-len(note))) for note in nst])

    @staticmethod
    def _format_notes_ts_group(nts_group):
        group_size = len(nts_group)
        times, cats, notes = [0]*group_size, [0]*group_size, [0]*group_size
        for d in nts_group.keys():
            _, gidx, _, time, _, cat = d.split('_')
            gidx, time, cat = int(gidx), int(
                time), np.array([int(c) for c in cat])
            times[gidx] = time
            cats[gidx] = cat
            notes[gidx] = nts_group[d][:]

        times, cats = np.array(times), np.array(cats)

        max_note_len = max([len(note) for note in notes])
        notes = np.array([np.pad(note, (0, max_note_len-len(note)))
                         for note in notes])

        return times, cats, notes
