from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap

phases = {
    'train', 'val'
}

datasets = {
    'ntu/xview', 'ntu/xsub',
}

parts = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
        (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
        (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ),
}

if __name__ == "__main__":
    for phase in phases:
        print(dataset, phase)
        data = np.load('data/{}/{}_data_joint.npy'.format(dataset, phase))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            'data/{}/{}_data_bone.npy'.format(dataset, phase),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))
        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(parts[dataset]):
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
