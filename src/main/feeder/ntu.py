import pickle
from typing import List

import numpy as np
from torch.utils.data import Dataset

from . import util


class NtuFeeder(Dataset):
    def __init__(
        self,
        path_data,
        path_label,
        ls_class=set(range(0, 120)),
        random_speed=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=True,
    ):
        """
        :param path_data:
        :param path_label:
        :param ls_class: The list of class [0-59] with ntu rgbd v1 (default: all)
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.random_speed: List[float] = random_speed
        self.debug: bool = debug
        self.path_data: str = path_data
        self.path_label: str = path_label
        self.random_choose: bool = random_choose
        self.random_shift: bool = random_shift
        self.random_move: bool = random_move
        self.window_size: int = window_size
        self.normalization: bool = normalization
        self.use_mmap: bool = use_mmap
        self.ls_class: List[int] = ls_class

        self.label: List[int]
        self.sample_name: List[int]
        self.data: np.ndarray

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C T V M

        try:
            with open(self.path_label) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.path_label, "rb") as f:
                self.sample_name, self.label = pickle.load(f, encoding="latin1")

        # load data
        if self.use_mmap:
            self.data = np.load(self.path_data, mmap_mode="r")
        else:
            self.data = np.load(self.path_data)
        if self.debug:
            self.label = self.label[0:10]
            self.data = self.data[0:10]
            self.sample_name = self.sample_name[0:10]

        idx_filter = [
            idx for idx in range(self.__len__()) if self.label[idx] in self.ls_class
        ]

        self.label = [self.label[idx] for idx in idx_filter]
        self.sample_name = [self.sample_name[idx] for idx in idx_filter]

        self.data = self.data[idx_filter]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def get_num_label(self):
        return len(set(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # data: N C T V M
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = util.random_shift(data_numpy)
        if self.random_speed:
            speed: int = np.clip(np.random.normal(loc=1, scale=0.5, size=1), a_max=2, a_min=0.2)[0]
            data_numpy=util.change_speed(data_numpy, speed)
        if self.random_choose:
            data_numpy = util.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = util.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = util.random_move(data_numpy)

        return data_numpy, label, index

