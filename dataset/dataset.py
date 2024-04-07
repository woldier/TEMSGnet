from torch.utils.data import Dataset
import numpy as np
import torch
dim = 8
ddim = 64


class MyDataset(Dataset):
    def __init__(self, s_path, s_n_path, repeat=None):
        s = np.load(s_path)[:, 50:]
        s_n = np.load(s_n_path)[:, 50:]
        self.max = -3.608072142064736
        self.min = -9.360870256811259
        mean = self.max + self.min
        std = self.max - self.min
        s = (s - mean / 2) / std * 2
        s_n = (s_n - mean / 2) / std * 2
        if repeat is not None:
            self.s = np.repeat(s, repeat, axis=0)
            self.s_n = np.repeat(s_n, repeat, axis=0)
        else:
            self.s = s
            self.s_n = s_n

    def __getitem__(self, item):
        s = np.array(self.s[item])
        s_n = np.array((self.s_n[item]))
        s_n = torch.Tensor(s_n)
        s = torch.Tensor(s)
        return s_n.unsqueeze(0), s.unsqueeze(0)

    def __len__(self):
        return len(self.s)

    def resume(self, x):
        mean = self.max + self.min
        std = self.max - self.min
        x = x * std / 2 + mean / 2
        return x


def SNR(data_s, data_n):
    index = 0
    if data_n.min() < 0:
        data_s = data_s - 2 * data_n.min()
        data_n = data_n - 2 * data_n.min()
    s = np.square(data_s)
    s = np.sum(s, axis=1)
    r = np.square(data_n - data_s)
    r = np.sum(r, axis=1)
    return 10 * np.log10(s / r)


if __name__ == '__main__':
    pass
