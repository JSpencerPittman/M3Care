from torch.utils.data import Dataset
import numpy as np
import math


class KFoldDatasetLoader(object):
    def __init__(self,
                 dataset: Dataset,
                 kfolds: int,
                 batch_size: int,
                 seed: int = 42):
        """
        Constructor for KFoldDatasetLoader.

        Args:
            dataset (Dataset): The dataset to be loaded.
            kfolds (int): The number of folds in the dataset.
            batch_size (int): Batch size.
            seed (int, optional): Random seed. Defaults to 42.
        """

        self.dataset = dataset
        self.size = len(dataset)
        self.kfolds = kfolds
        self.batch_size = batch_size

        indices = np.arange(self.size)
        np.random.seed(seed)
        np.random.shuffle(indices)

        self.fold_sizes = np.array([math.floor(self.size/kfolds)] * kfolds)
        for fidx in range(self.size % kfolds):
            self.fold_sizes[fidx] += 1

        self.fold_indices = [indices[(s := sum(self.fold_sizes[:fidx])):s+fold_size]
                             for fidx, fold_size in enumerate(self.fold_sizes)]

        # Tracking state
        self.train_batch_idx = 0
        self.train_fold_idx = 1
        self.train_round_end = False

        self.val_batch_idx = 0
        self.val_fold_idx = 0
        self.val_round_end = False

        self.is_train = True
        self.round_end = False
        self.end = False

    def next_batch(self):
        """
        Load the next batch.
        """

        if self.round_end:
            return None, None
        elif self.is_train:
            return self._next_train_batch()
        else:
            return self._next_val_batch()

    def next_round(self):
        """
        Switch to the next round. This will select the next fold to be used for
        validation.
        """

        if self.end:
            return None, None
        self._init_round(self.val_fold_idx + 1)

    def train(self):
        """
        Toggle to training mode.
        """

        self.is_train = True

    def val(self):
        """
        Toggle to validation mode.
        """

        self.is_train = False

    def reset(self):
        self._init_round(0)

    def _init_round(self, val_fold_idx: int):
        """
        Initalize round. Each round is defined by the fold currently serving as the
        validation fold.

        Args:
            val_fold_idx (int): The fold to be used for validation.
        """

        self.train_batch_idx = 0
        self.train_fold_idx = 0 if val_fold_idx != 0 else 1
        self.train_round_end = False

        self.val_batch_idx = 0
        self.val_fold_idx = val_fold_idx
        self.val_round_end = False

        self.is_train = True
        self.round_end = False
        self.end = False

    def _next_train_batch(self):
        """
        Load the next training batch.
        """

        if self.train_round_end:
            return None, None

        rem_samples = self._samples_remaining_in_fold(self.train_batch_idx,
                                                      self.train_fold_idx)
        if rem_samples == 0:
            self.train_batch_idx = 0
            self.train_fold_idx += 1
            return self._next_train_batch()

        start = self.train_batch_idx * self.batch_size
        end = start + min(self.batch_size, rem_samples)
        batch_indices = self.fold_indices[self.train_fold_idx][start:end]

        self.train_batch_idx += 1
        self._check_train_round_end()

        return self.dataset[batch_indices], end-start

    def _next_val_batch(self):
        """
        Load the next validation batch.
        """

        if self.val_round_end:
            return None, None

        rem_samples = self._samples_remaining_in_fold(self.val_batch_idx,
                                                      self.val_fold_idx)
        start = self.val_batch_idx * self.batch_size
        end = start + min(self.batch_size, rem_samples)
        batch_indices = self.fold_indices[self.val_fold_idx][start:end]

        self.val_batch_idx += 1
        self._check_val_round_end()

        return self.dataset[batch_indices], end-start

    def _samples_remaining_in_fold(self,
                                   batch_idx: int,
                                   fold_idx: int,
                                   include_batch: bool = True) -> int:
        """
        Count how many samples are left in the selected fold.

        Args:
            batch_idx (int): Index of the batch within the fold - not across all folds.
            fold_idx (int): Index of the fold.
            include_batch (bool, optional): Include the selected batch in the returned
                count. Defaults to True.

        Returns:
            int: The number of remaning samples in the selected fold.
        """
        loc = (batch_idx + (not include_batch)) * self.batch_size
        num_samples_in_fold = self.fold_sizes[fold_idx]

        return max(num_samples_in_fold - loc, 0)

    def _train_folds_remaining(self, fold_idx: int, include_fold: bool = True):
        """
        Count how many folds are remaining for the current round.

        Args:
            fold_idx (int): Index of the fold.
            include_fold (bool, optional): Include the selected fold in the returned
                count. Defaults to True.

        Returns:
            int: The number of remaning folds in the current round.
        """

        if not include_fold:
            fold_idx += 1
        rem = self.kfolds - fold_idx
        if self.val_fold_idx >= fold_idx:
            rem -= 1
        return max(rem, 0)

    def _rounds_remaining(self, include_round: bool = True):
        """
        Count how many rounds are remaining. Determined by how many folds are left to
        use for validation.

        Args:
            include_round (bool, optional): Include the current round in the count.

        Returns:
            int: The number of remaning rounds.
        """

        return max(self.kfolds - self.val_fold_idx - (not include_round), 0)

    def _check_train_round_end(self):
        """
        Check if the current round's training segment has loaded all batches.
        """

        rem_folds = self._train_folds_remaining(self.train_fold_idx, include_fold=False)
        if rem_folds > 0:
            return

        rem_samples = self._samples_remaining_in_fold(self.train_batch_idx,
                                                      self.train_fold_idx)
        if rem_samples > 0:
            return

        self.train_round_end = True
        self._check_round_end()

    def _check_val_round_end(self):
        """
        Check if the current round's validation segment has loaded all batches.
        """

        rem_samples = self._samples_remaining_in_fold(self.val_batch_idx,
                                                      self.val_fold_idx)
        if rem_samples > 0:
            return

        self.val_round_end = True
        self._check_round_end()

    def _check_round_end(self):
        """
        Check if the current round has loaded all training and validation batches.
        If the round has ended a secondary check is made to determine if any rounds
        remain, and if not then the loader has reached the end state.
        """

        self.round_end = self.train_round_end and self.val_round_end

        if self.round_end:
            rem_rounds = self._rounds_remaining(include_round=False)
            if rem_rounds == 0:
                self.end = True
