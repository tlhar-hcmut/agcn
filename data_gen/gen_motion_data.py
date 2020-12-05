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
                file_dataset = 'data/%s/%s_data_%s.npy' % (dataset, phase, part)
                file_motion = 'data/%s/%s_data_%s_motion.npy' % (dataset, phase, part)
                data = np.load(file_dataset)
                N, C, T, V, M = data.shape

                fp_sp = open_memmap(
                    filename=file_motion,
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))

                for t in tqdm(range(T - 1)):
                    frame_next = data[:, :, t + 1, :, :]
                    frame_prev = data[:, :, t, :, :]
                    fp_sp[:, :, t, :, :] = frame_next - frame_prev

                fp_sp[:, :, T - 1, :, :] = 0

                del fp_sp
