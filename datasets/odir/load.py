from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

class DataSplit:
    def __init__(self, idx_split):
        self._idx_split = idx_split

    def idx_split(self):
        return self._idx_split

    def split(self, data):
        return {
            "train": data[self._idx_split["train"]],
            "val": data[self._idx_split["val"]],
            "test": data[self._idx_split["test"]]
        }

def load_idxsplit():
    with open("odir_prep/idx_split.pkl", "rb") as f:
        idx_split = pickle.load(f)
    return DataSplit(idx_split)

def load_x(splitter):
    with open("odir_prep/x_prep.pkl", "rb") as f:
        x_prep = pickle.load(f)

    x_split = splitter.split(x_prep)

    age_scaler = StandardScaler()

    x_split["train"][:, 0] = age_scaler.fit_transform(x_split["train"][:, 0].astype(np.float32).reshape(-1,1)).ravel()
    x_split["val"][:, 0] = age_scaler.transform(x_split["val"][:, 0].astype(np.float32).reshape(-1,1)).ravel()
    x_split["test"][:, 0] = age_scaler.transform(x_split["test"][:, 0].astype(np.float32).reshape(-1,1)).ravel()

    return x_split

def load_y(splitter):
    with open("odir_prep/y_prep.pkl", "rb") as f:
        y_prep = pickle.load(f)

    return splitter.split(y_prep)

def load_diagnoses(splitter):
    with open("odir_prep/left_diag.pkl", "rb") as f:
        left_diag = pickle.load(f)
    with open("odir_prep/right_diag.pkl", "rb") as f:
        right_diag = pickle.load(f)

    left_diag = np.array(left_diag, dtype="object")
    right_diag = np.array(right_diag, dtype="object")

    return {
        "left": splitter.split(left_diag),
        "right": splitter.split(right_diag)
    }

def load_diagnosis_masks(splitter):
    with open("odir_prep/left_diag_mask.pkl", "rb") as f:
        left_diag_mask = pickle.load(f)
    with open("odir_prep/right_diag_mask.pkl", "rb") as f:
        right_diag_mask = pickle.load(f)

    left_diag_mask = np.array(left_diag_mask)
    right_diag_mask = np.array(right_diag_mask)

    return {
        "left": splitter.split(left_diag_mask),
        "right": splitter.split(right_diag_mask)
    }

def load_images(splitter):
    results = {
        "left": dict(),
        "right": dict()
    }

    for section in ["train", "val", "test"]:
        left_fundus_images = list()
        right_fundus_images = list()

        for idx in splitter.idx_split()[section]:
            fundus_images = np.load(f"odir_prep/Transform/{section.capitalize()}/{idx}.npy")
            left_fundus_images.append(fundus_images[0])
            right_fundus_images.append(fundus_images[1])

        results["left"][section] = left_fundus_images
        results["right"][section] = right_fundus_images
    
    return results
