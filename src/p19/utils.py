from pathlib import Path

import numpy as np
import torch

from src.p19.dataset import P19Dataset


def load_p19_data(data_path: Path, device: str = 'cpu'):
    # static_feat_names = np.load(data_path / 'labels_demogr.npy')
    # ts_feat_names = np.load(data_path / 'labels_ts.npy')
    inputs = np.load(data_path / 'PT_dict_list_6.npy', allow_pickle=True)
    labels = np.load(data_path / 'arr_outcomes_6.npy').squeeze()

    ts_inputs = np.array([inp['arr'] for inp in inputs])[:, :, :, np.newaxis]
    static_inputs = np.array([inp['extended_static'] for inp in inputs])
    times = np.array([inp['time'] for inp in inputs]).squeeze()
    lengths = np.array([inp['length'] for inp in inputs])

    ts_inputs = torch.tensor(ts_inputs, dtype=torch.float32).to(device)
    static_inputs = torch.tensor(static_inputs, dtype=torch.float32).to(device)
    times = torch.tensor(times, dtype=torch.float32).to(device)
    lengths = torch.tensor(lengths).to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)

    return ts_inputs, static_inputs, times, lengths, labels

def split_p19_data(ts_inputs,
                   static_inputs,
                   times,
                   lengths,
                   labels,
                   device,
                   summary: bool = False):
    np.random.seed(42)

    num_samples = ts_inputs.shape[0]
    idxs = np.arange(num_samples)
    np.random.shuffle(idxs)

    train_idxs, val_idxs, test_idxs = idxs[:(s1 := int(num_samples*0.8))], \
                                      idxs[s1: (s2 := int(num_samples*0.9))], \
                                      idxs[s2:]

    train_ts_inp, val_ts_inp, test_ts_inp = ts_inputs[train_idxs], \
                                            ts_inputs[val_idxs], \
                                            ts_inputs[test_idxs]
    train_static_inp, val_static_inp, test_static_inp = static_inputs[train_idxs], \
                                                        static_inputs[val_idxs], \
                                                        static_inputs[test_idxs]
    train_times, val_times, test_times = times[train_idxs], \
                                         times[val_idxs], \
                                         times[test_idxs]
    train_lengths, val_lengths, test_lengths = lengths[train_idxs], \
                                               lengths[val_idxs], \
                                               lengths[test_idxs]
    train_lbls, val_lbls, test_lbls = labels[train_idxs], \
                                      labels[val_idxs], \
                                      labels[test_idxs]
    
    train_ds = P19Dataset(train_ts_inp,
                          train_times,
                          train_lengths,
                          train_static_inp,
                          train_lbls,
                          device)
    val_ds = P19Dataset(val_ts_inp,
                        val_times,
                        val_lengths,
                        val_static_inp,
                        val_lbls,
                        device)
    test_ds = P19Dataset(test_ts_inp,
                         test_times,
                         test_lengths,
                         test_static_inp,
                         test_lbls,
                         device)
    
    if summary:
        names = ['Training', 'Validation', 'Testing']
        s = "P19 Summary\n"

        def prop(lbls: torch.Tensor) -> str:
            pos = lbls.sum().item()
            count = lbls.shape[0]
            return f"{(100 * pos/count):.2f}%"

        num_samples = f"\tTotal samples {len(ts_inputs)}\n"
        class_props = f"\tClasses {prop(labels)} positive\n"

        for lbl, name in zip([train_lbls, val_lbls, test_lbls],
                             names):
            num_samples += f"\t\t{name}: {lbl.shape[0]}\n"
            class_props += f"\t\t{name}: {prop(lbl)}\n"

        s += num_samples
        s += class_props

        print(s)

    return (train_idxs, val_idxs, test_idxs), (train_ds, val_ds, test_ds)
