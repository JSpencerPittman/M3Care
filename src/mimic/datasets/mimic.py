from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mimic.datasets.demographic import DemographicDataset
from mimic.datasets.interventions import InterventionsDataset
from mimic.datasets.modal_dataset import ModalDataset
from mimic.datasets.static_notes import StaticNotesDataset
from mimic.datasets.ts_notes import TSNotesDataset
from mimic.datasets.vitals import VitalsDataset
from mimic.vocab import Vocab

PathLike = Path | str
VITALS_SEQ_LEN = 150
INTERVENTIONS_SEQ_LEN = 150
TS_NOTES_TIME_DIM = 128


class AuxillaryPaths(TypedDict):
    pat_ids: Optional[PathLike]
    vocab: Optional[PathLike]


class DatasetPaths(TypedDict):
    demographic: Optional[PathLike]
    vitals: Optional[PathLike]
    interventions: Optional[PathLike]
    static_notes: Optional[PathLike]
    ts_notes: Optional[PathLike]


class MimicDataset(Dataset):
    """
    This class aggregates all modalities for the MIMIC-III dataset.

    Used modalities:
    - Demographics (Tabular)
    - Vitals (Time-series)
    - Interventions (Time-series)
    - Static Notes (Natural Language)
    - Time Series Notes (Time-series & Natural Language)
    """

    # Each dataset requires specific auxillaries to accompany it.
    Dependencies: dict[str, str] = {
        "demographic": {'pat_ids'},
        "vitals": {'pat_ids'},
        "interventions": {'pat_ids'},
        "static_notes": {'pat_ids', 'vocab'},
        "ts_notes": {'pat_ids', 'vocab'}
    }

    # Ordering of datasets
    Dataset_Order = ('demographic',
                     'vitals',
                     'interventions',
                     'static_notes',
                     'ts_notes')

    def __init__(self,
                 dataset_paths: DatasetPaths,
                 auxillary_paths: AuxillaryPaths,
                 device: str = 'cpu'):
        """
        Constructor for MimicDataset.

        Firstly, the dataset and auxillary paths are verified to ensure that all
        provided entries are either None or describe a valid existant path. Secondly,
        All of the auxillaries are loaded and initialized. Finally, all datasets being
        used are loaded and initialized.

        Criteria:
        1. At least one dataset of the datasets in DatasetsPaths is provided.
        2. For all dataset paths provided the location to their dependencies
        are provided in auxillary_paths.

        Args:
            dataset_paths (DatasetPaths): Paths for all datasets to be used.
            auxillary_paths (AuxillaryPaths): Paths for all auxillaries to be loaded.
        """

        self.dataset_paths = dataset_paths
        self.auxillary_paths = auxillary_paths

        self.provided_datasets = self._get_provided_datasets(dataset_paths)
        self.provided_auxillaries = \
            self._get_provided_auxillaries(auxillary_paths, self.provided_datasets)

        for auxillary in self.provided_auxillaries:
            self._initialize_auxillary(auxillary)

        self.datasets: dict[str, ModalDataset] = {}
        for dataset in self.provided_datasets:
            self._initialize_dataset(dataset)

        self.device = device

    def __getitem__(self,
                    idx: int | slice) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        x, masks = {}, {}
        for ds in MimicDataset.Dataset_Order:
            if ds not in self.provided_datasets:
                continue
            res = self.datasets[ds][idx]
            if isinstance(res, tuple):
                x[ds] = torch.tensor(res[0], dtype=torch.float32, device=self.device)
                masks[ds] = torch.tensor(res[1], dtype=torch.bool, device=self.device)
            else:
                x[ds] = torch.tensor(res, dtype=torch.float32, device=self.device)
        return x, masks

    def __len__(self) -> int:
        return len(self.pat_ids)

    def _initialize_dataset(self, name: str):
        """
        Intialize a dataset. The dataset is loaded from its specified path and then
        initialized under the appropriate modal-specific dataset.

        Args:
            name (str): The dataset to be loaded.

        Raises:
            RuntimeWarning: Attempt to load a dataset not supported.
        """

        match name:
            case 'demographic':
                self.datasets[name] = DemographicDataset(self.dataset_paths[name],
                                                         self.pat_ids)
            case 'vitals':
                self.datasets[name] = VitalsDataset(self.dataset_paths[name],
                                                    self.pat_ids,
                                                    VITALS_SEQ_LEN)
            case 'interventions':
                self.datasets[name] = InterventionsDataset(self.dataset_paths[name],
                                                           self.pat_ids,
                                                           INTERVENTIONS_SEQ_LEN)
            case 'static_notes':
                self.datasets[name] = StaticNotesDataset(self.dataset_paths[name],
                                                         self.pat_ids,
                                                         self.vocab)
            case 'ts_notes':
                self.datasets[name] = TSNotesDataset(self.dataset_paths[name],
                                                     self.pat_ids,
                                                     self.vocab,
                                                     TS_NOTES_TIME_DIM)
            case _:
                raise RuntimeWarning(f"MimicDataset: Dataset {name} not supported.")

    def _initialize_auxillary(self, name: str):
        """
        Intialize an auxillary. The file is loaded from its specified path and then
        initialized under the appropriate class/operation.

        Args:
            name (str): The auxillary to be loaded.

        Raises:
            RuntimeWarning: Attempt to load an auxillary not supported.
        """

        match name:
            case 'pat_ids':
                self.pat_ids = np.load(self.auxillary_paths[name])
            case 'vocab':
                self.vocab = Vocab.from_json(self.auxillary_paths[name])
            case _:
                raise RuntimeWarning(f"MimicDataset: Auxillary {name} not supported.")

    @classmethod
    def _get_provided_datasets(cls, dataset_paths: DatasetPaths) -> set[str]:
        """
        Identify all provided datasets. Before identifying provided datasets the
        provided paths are validated. Next is checking that at least one dataset was
        provided.

        Args:
            dataset_paths (DatasetPaths): A dictionary of the paths to dataset files.

        Raises:
            RuntimeError: No datasets were provided.

        Returns:
            set[str]: A set of all datasets provided.
        """

        provided_datasets = cls._validate_paths(dataset_paths)

        if len(provided_datasets) == 0:
            raise RuntimeError("MimicDataset: No datasets provided.")
        return provided_datasets

    @classmethod
    def _get_provided_auxillaries(cls,
                                  auxillary_paths: AuxillaryPaths,
                                  provided_datasets: set[str]) -> set[str]:
        """
        Identify all provided auxillaries. Before identifying provided auxillaries
        the provided paths are validated. Next is verifying that all required
        auxillaries were provided.

        Args:
            auxillary_paths (AuxillaryPaths): A dictionary of the paths to auxillary
                files.
            provided_datasets (set[str]): All datasets being used.

        Raises:
            RuntimeError: One or more required auxillaries aren't provided.

        Returns:
            set[str]: A set of all auxillaries provided.
        """

        # Required auxillaries
        required_auxillaries: set[str] = set()
        for dataset in provided_datasets:
            required_auxillaries |= MimicDataset.Dependencies[dataset]

        provided_auxillaries = cls._validate_paths(auxillary_paths)
        required_auxillaries -= provided_auxillaries

        if len(required_auxillaries):
            raise RuntimeError(
                f"MimicDataset: Missing required auxillaries: {required_auxillaries}."
            )

        return provided_auxillaries

    @classmethod
    def _validate_paths(cls, paths: dict[str, Optional[PathLike]]) -> set[str]:
        """
        Validate the provided dictionary of paths.

        Criteria:
        1. If a provided entry is not None then it must be a PathLike (str or Path).
        2. All none empty paths must point to an existing file.

        Args:
            paths dict[str, Optional[PathLike]]: Dictionary listing paths to files.

        Raises:
            FileNotFoundError: Provided path points to a nonexistant file.
            TypeError: Provided path is not None or PathLike.

        Returns:
            set[str]: A set of the keys describing all provided paths.
        """

        provided_paths: set[str] = set()

        for key, path in paths.items():
            if path is not None:
                if isinstance(path, PathLike):
                    if not Path(path).exists():
                        raise FileNotFoundError(
                            f"MimicDataset: Provided path does not exist: {path}."
                        )
                    else:
                        provided_paths.add(key)
                else:
                    raise TypeError(
                        f"""IMICDataset: Provided path type is `{type(path)}`, requires
                        PathLike."""
                    )

        return provided_paths
