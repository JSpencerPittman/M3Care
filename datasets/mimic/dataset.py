from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


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
            self.notes_static = pd.read_csv(
                os.path.join(data_path, 'notes_static.csv'))
            self.notes_ts = pd.read_csv(
                os.path.join(data_path, 'notes_ts.csv'))
        except FileNotFoundError as e:
            print("Make sure data has been processed: ", e)
            return

        self.demographics.set_index('pat_id', inplace=True)
        self.vitals.set_index(['pat_id', 'hours_in'], inplace=True)
        self.interventions.set_index(['pat_id', 'hours_in'], inplace=True)
        self.notes_static.set_index('pat_id', inplace=True)
        self.notes_ts.set_index(['pat_id', 'hours_in'], inplace=True)

        self.nst_ids = set(self.notes_static.index.values)
        self.nts_ids = set(self.notes_ts.index.get_level_values(0).values)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item_idx):
        pat_id = self.indexes[item_idx]

        dem = self.demographics.loc[pat_id]

        vit = self.vitals.loc[pat_id]
        vit = self._format_ts_batch(vit)

        itv = self.interventions.loc[pat_id]
        itv = self._format_ts_batch(itv)

        # Static Notes
        nst = ''
        if type(pat_id) != np.ndarray and pat_id in self.nst_ids:
            nst = self.notes_static[pat_id].TEXT
        elif type(pat_id) == np.ndarray:
            pat_ids, notes_st = pat_id, []
            for pat_id in pat_ids:
                if pat_id in self.nst_ids:
                    notes_st.append(self.notes[pat_id].TEXT)
                else:
                    notes_st.append('')

        # Time Series Notes
        nts = []
        if type(pat_id) != np.ndarray:
            if pat_id not in self.nts_ids:
                nts = [], [], None
            else:
                nts = self._format_notes_ts_batch(self.notes_ts.loc[pat_id, :])
        else:
            pat_ids = pat_id
            match_ids = [
                pat_id for pat_id in pat_ids if pat_id in self.nts_ids]
            batch_nts_ts = self.notes_ts.loc[match_ids, :]
            batch_nts_ts = self._format_notes_ts_batch(batch_nts_ts)

            for pat_id in pat_ids:
                if pat_id in match_ids:
                    nts.append(batch_nts_ts[match_ids.index(pat_id)])
                else:
                    nts.append([[], [], None])

        return dem, vit, itv, nst, nts

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
    def _format_notes_ts_batch(batch_nts_df):
        if batch_nts_df.index.nlevels == 1:
            time = batch_nts_df.index.tolist()
            note = batch_nts_df.TEXT.tolist()
            cat = batch_nts_df.drop('TEXT', axis=1).values
            return time, note, cat

        times = batch_nts_df.groupby(level=0).apply(
            lambda g: g.index.get_level_values(1).tolist()).tolist()
        notes = batch_nts_df.groupby(level=0).apply(
            lambda g: g.TEXT.tolist()).tolist()
        cats = batch_nts_df.groupby(level=0).apply(
            lambda g: g.drop('TEXT', axis=1).values).tolist()

        return [[time, note, cat] for time, note, cat in zip(times, notes, cats)]
