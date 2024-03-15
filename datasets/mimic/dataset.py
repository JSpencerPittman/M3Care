from torch.utils.data import Dataset
from enum import Enum
import pandas as pd
import numpy as np
import os


class DatasetType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class MIMICDataset(Dataset):
    def __init__(self, processed_dir: str, dataset_type: DatasetType):
        self.processed_dir = processed_dir
        self.dataset_type = dataset_type

        if dataset_type == DatasetType.TRAIN:
            self.data_path = os.path.join(self.processed_dir, 'train/')
            self.index_path = os.path.join(
                self.processed_dir, 'train_idxs.npy')
        elif dataset_type == DatasetType.VAL:
            self.data_path = os.path.join(self.processed_dir, 'val/')
            self.index_path = os.path.join(self.processed_dir, 'val_idxs.npy')
        elif dataset_type == DatasetType.TEST:
            self.data_path = os.path.join(self.processed_dir, 'test/')
            self.index_path = os.path.join(self.processed_dir, 'test_idxs.npy')
        else:
            raise ValueError("Invalid dataset type.")

        try:
            self.idxs = np.load(self.index_path)
            self.demographics = pd.read_csv(
                os.path.join(self.data_path, 'demographic.csv'))
            self.vitals = pd.read_csv(
                os.path.join(self.data_path, 'vitals.csv'))
            self.interventions = pd.read_csv(
                os.path.join(self.data_path, 'interventions.csv'))
            self.notes_static = pd.read_csv(
                os.path.join(self.data_path, 'notes_static.csv'))
            self.notes_ts = pd.read_csv(
                os.path.join(self.data_path, 'notes_ts.csv'))
        except FileNotFoundError as e:
            print("Make sure data has been processed: ", e)

        self.demographics.set_index('pat_id', inplace=True)
        self.vitals.set_index(['pat_id', 'hours_in'], inplace=True)
        self.interventions.set_index(['pat_id', 'hours_in'], inplace=True)
        self.notes_static.set_index('pat_id', inplace=True)
        self.notes_ts.set_index(['pat_id', 'hours_in'], inplace=True)

        self.nst_idxs = set(self.notes_static.index.values)
        self.nts_idxs = set(self.notes_ts.index.get_level_values(0).values)

    def __len__(self):
        return self.demographics.shape[0]

    def __getitem__(self, idx):
        demographic = self.demographics.loc[idx]
        if idx not in self.idxs:
            raise KeyError("invalid key for the dataset.")

        dem = self.demographics.loc[idx]
        vit = self.vitals.loc[idx]
        itv = self.interventions.loc[idx]

        nst = self.notes_static.loc[idx] if idx in self.nst_idxs else None
        nts = self.notes_ts.loc[idx] if idx in self.nts_idxs else None

        return dem, vit, itv, nst, nts
