from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap

phases = {
    'train', 'val'
}
# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'ntu/xview', 'ntu/xsub'
}

parts = {
    'joint', 'bone'
}

for dataset in datasets:
    for phase in phases:
        for part in parts:
            print(dataset, phase, part)
            data = np.load(
                'data/{}/{}_data_{}.npy'.format(dataset, phase, part))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                'data/{}/{}_data_{}_motion.npy'.format(dataset, phase, part),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))
            for t in tqdm(range(T - 1)):
                fp_sp[:, :, t, :, :] = data[:, :,
                                            t + 1, :, :] - data[:, :, t, :, :]
            fp_sp[:, :, T - 1, :, :] = 0
