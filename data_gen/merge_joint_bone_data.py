import os
import numpy as np

phases = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub', 'kinetics'
datasets = {
    'ntu/xview', 'ntu/xsub'
}

for dataset in datasets:
    for phase in phases:
        print(dataset, phase)
        data_jpt = np.load('data/{}/{}_data_joint.npy'.format(dataset, phase))
        data_bone = np.load('data/{}/{}_data_bone.npy'.format(dataset, phase))
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
        np.save('data/{}/{}_data_joint_bone.npy'.format(dataset, phase), data_jpt_bone)
