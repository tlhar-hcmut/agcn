from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap

phases = {
    'train', 'val'
}

datasets = {
    'ntu/xview', 'ntu/xsub'
}

parts = {
    'joint', 'bone'
}

if __name__ == "__main__":
    for dataset in datasets:
        for phase in phases:
            for part in parts:
                print(dataset, phase, part)
                data = np.load('data/{}/{}_data_{}.npy' %
                               (dataset, phase, part))

                N, C, T, V, M = data.shape

                fp_sp = open_memmap(
                    'data/{}/{}_data_{}_motion.npy' % (dataset, phase, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))

                for t in tqdm(range(T - 1)):
                    frame_next = data[:, :, t + 1, :, :]
                    frame_prev = data[:, :, t, :, :]
                    fp_sp[:, :, t, :, :] = frame_next - frame_prev

                fp_sp[:, :, T - 1, :, :] = 0
