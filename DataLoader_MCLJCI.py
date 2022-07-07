import os
import torch
import numpy as np
from torch.utils.data import Dataset


class mcl_jci(Dataset):
    def __init__(self, jnd_info, root_dir):
        # Initialization of file names and dis-similarity measure acquired from MCL-JCI dataset.
        self.ref_name = jnd_info[:, 0]
        self.test_name = jnd_info[:, 1]
        self.root_dir = str(root_dir)
        self.gt = jnd_info[:, 2]

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt = float(self.gt[idx])
        # Loading pre-processed Achromatic responses which are stored as npy file.
        ref = np.load(os.path.join(self.root_dir, str(self.ref_name[idx])))
        ref = np.expand_dims(ref, 0)
        test = np.load(os.path.join(self.root_dir, str(self.test_name[idx])))
        test = np.expand_dims(test, 0)
        return ref, test, gt


class ToTensor(object):
    # Convert ndarrays in sample to Tensors
    def __call__(self, sample):
        ref_f, test_f, gt = sample[0], sample[1], sample[2]
        return torch.from_numpy(ref_f), torch.from_numpy(test_f), torch.from_numpy(gt)
