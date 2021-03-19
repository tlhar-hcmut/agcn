import yaml

with open("agcn/config/general-config/general_config.yaml", "r") as f:
    default_arg = yaml.load(f, Loader=yaml.FullLoader)

from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap

phases = {"train", "val"}

datasets = {"xview", "xsub"}

parts = {"joint", "bone"}


def gen_motion(dataset, phase, part):
    """
    Sub coordinates of the same joint through frames . Run this after gen joint data.
    """
    print(dataset, phase, part)
    file_dataset = default_arg["path_data_preprocess"] + "/%s/%s/%s_%s.npy" % (
        dataset,
        phase,
        phase,
        part,
    )
    file_motion = default_arg["path_data_preprocess"] + "/%s/%s/%s_%s_motion.npy" % (
        dataset,
        phase,
        phase,
        part,
    )
    data = np.load(file_dataset)
    N, C, T, V, M = data.shape

    fp_sp = open_memmap(
        filename=file_motion, dtype="float32", mode="w+", shape=(N, 3, T, V, M)
    )

    for t in tqdm(range(T - 1)):
        frame_next = data[:, :, t + 1, :, :]
        frame_prev = data[:, :, t, :, :]
        fp_sp[:, :, t, :, :] = frame_next - frame_prev

    fp_sp[:, :, T - 1, :, :] = 0

    del fp_sp


if __name__ == "__main__":
    for dataset in datasets:
        for phase in phases:
            for part in parts:
                gen_motion(dataset, phase, part)
