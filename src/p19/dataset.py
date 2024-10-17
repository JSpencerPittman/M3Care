from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch


class P19Dataset(Dataset):
    def __init__(self,
                 ts_inp: Tensor,
                 times: Tensor,
                 lengths: Tensor,
                 static_inp: Tensor,
                 labels: Tensor,
                 device: str):
        self.ts_inp = ts_inp
        self.times = times
        self.lengths = lengths
        self.static_inp = static_inp
        self.labels = labels
        self.device = device

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, loc: int | slice):
        ts_inp = self.ts_inp[loc]

        # Create mask
        mask = torch.zeros(ts_inp.shape[:-1], dtype=bool, device=self.device)
        if isinstance(loc, int):
            mask[:self.lengths[loc]] = 1
        else:
            for idx, length in enumerate(self.lengths[loc]):
                mask[idx, :length] = 1

        return ts_inp, self.times[loc], mask, self.static_inp[loc], self.labels[loc]
