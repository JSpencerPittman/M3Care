from torch.utils.data import Dataset
import numpy as np
import math


class KFoldDatasetLoader(object):
    def __init__(self, dataset: Dataset, kfolds: int, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.m = len(dataset)
        self.num_folds = kfolds

        all_indices = np.arange(self.m)
        np.random.seed(seed)
        np.random.shuffle(all_indices)

        self.split_sizes = np.array([math.floor(self.m/kfolds)] * kfolds)
        for fidx in range(self.m % kfolds):
            self.split_sizes[fidx] += 1

        self.split_indices = []
        last_index = 0
        for fidx, size in enumerate(self.split_sizes):
            bound = last_index + size
            self.split_indices.append(all_indices[last_index:bound])
            last_index = bound

        self.batch_size = batch_size

        # Tracking state
        self.batch_idx = 0
        self.fold_idx = 0
        self.train_mode = True

        self.train_num_batches = 0
        self.train_indices = np.empty(0)

        self.val_num_batches = 0
        self.val_indices = np.empty(0)

        self._initialize_fold_state()

    def next(self):
        if self.end():
            return None

        lidx = self.batch_size * self.batch_idx
        ridx = lidx + self.batch_size
        indices = self.train_indices[lidx:ridx] if self.train_mode else self.val_indices[lidx:ridx]
        self.batch_idx += 1

        return self.dataset[indices]

    def next_fold(self) -> bool:
        if self.end_fold():
            return False

        self.fold_idx += 1

        self._initialize_fold_state()
        return True

    def end(self):
        num_batches = self.train_num_batches if self.train_mode else self.val_num_batches
        return self.batch_idx == num_batches

    def end_fold(self) -> bool:
        self.fold_idx = min(self.fold_idx + 1, self.kfolds)

        if self.fold_idx == self.kfolds:
            return False

    def reset(self) -> bool:
        self.fold_idx = 0
        self._initialize_fold_state()

    def train(self):
        self.train_mode = True
        self.batch_idx = 0

    def val(self):
        self.train_mode = False
        self.batch_idx = 0

    def _initialize_fold_state(self):
        self.train_mode = True

        self.batch_idx = 0

        self.train_indices = []
        for fidx in range(self.num_folds):
            if fidx != self.fold_idx:
                self.train_indices.append(self.split_indices[fidx])
        self.train_indices = np.concatenate(self.train_indices)
        self.val_indices = self.split_indices[self.fold_idx]

        self.train_num_batches = math.floor(
            len(self.train_indices) / self.batch_size)
        self.val_num_batches = math.floor(
            len(self.val_indices) / self.batch_size)
