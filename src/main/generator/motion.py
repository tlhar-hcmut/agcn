import numpy as np
from numpy.lib.format import open_memmap
from src.main.config import cfg_ds
from tqdm import tqdm
from xcommon import xconsole

phases = {"train", "val"}

datasets = {"xview", "xsub"}

parts = {"joint", "bone"}


def gen_motion(file_join: str, file_motion: str) -> None:
    """
    Sub coordinates of the same joint through frames . Run this after gen joint data.
    """

    data = np.load(file_join)
    N, C, T, V, M = data.shape

    fp_sp = open_memmap(filename=file_motion, dtype="float32", mode="w+", shape=(N, 3, T, V, M))

    for t in tqdm(range(T - 1)):
        frame_next = data[:, :, t + 1, :, :]
        frame_prev = data[:, :, t, :, :]
        fp_sp[:, :, t, :, :] = frame_next - frame_prev

    fp_sp[:, :, T - 1, :, :] = 0

    del fp_sp


if __name__ == "__main__":
    for benmark in cfg_ds.ls_benmark:
        xconsole.info(benmark.name)
        for phase in ["train", "val"]:

            file_join = "%s/%s_%s_joint.npy" % (
                cfg_ds.path_data_preprocess,
                phase,
                benmark.name,
            )
            file_motion = "%s/%s_%s_motion.npy" % (
                cfg_ds.path_data_preprocess,
                phase,
                benmark.name,
            )

            gen_motion(file_join, file_motion)
