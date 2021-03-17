from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import yaml
with open("agcn/config/general-config/general_config.yaml", 'r') as f:
    default_arg = yaml.load(f, Loader=yaml.FullLoader)
            
phases = {
    'train', 'val'
}

datasets = {
    'ntu/xview', 'ntu/xsub',
}

egdes = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
         (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
         (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
         (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))


def gen_bone(dataset, phase):
    print(dataset, phase)
    file_join = default_arg['output_data']+'data/%s/%s_data_joint.npy' % (dataset, phase)
    file_bone = 'data/%s/%s_data_bone.npy' % (dataset, phase)
    data = np.load(file_join)
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(
        filename=file_bone,
        dtype='float32',
        mode='w+',
        shape=(N, 3, T, V, M))
    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(egdes):
        joint_1 = data[:, :, :, v1, :]
        joint_2 = data[:, :, :, v2, :]
        fp_sp[:, :, :, v1, :] = joint_1 - joint_2

    del fp_sp


if __name__ == "__main__":
    for dataset in datasets:
        for phase in phases:
            gen_bone(dataset, phase)
