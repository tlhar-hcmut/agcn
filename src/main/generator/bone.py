import numpy as np
from numpy.lib.format import open_memmap
from src.main.config import cfg_ds
from tqdm import tqdm
from xcommon import xconsole

egdes = (
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20),
    (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0),
    (17, 16), (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)
)


def gen_bone(file_join: str, file_bone: str) -> None:
    """
    Link joint from data joint. Run this after gen joint data.
    """
    data = np.load(file_join)
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(
        filename=file_bone, dtype="float32", mode="w+", shape=(N, 3, T, V, M)
    )
    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(egdes):
        joint_1 = data[:, :, :, v1, :]
        joint_2 = data[:, :, :, v2, :]
        fp_sp[:, :, :, v1, :] = joint_1 - joint_2

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
            file_bone = "%s/%s_%s_bone.npy" % (
                cfg_ds.path_data_preprocess,
                phase,
                benmark.name,
            )

            gen_bone(file_join, file_bone)
