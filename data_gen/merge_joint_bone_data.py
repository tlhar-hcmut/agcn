import os
import numpy as np

phases = {
    'train', 'val'
}

datasets = {
    'ntu/xview', 'ntu/xsub'
}


def gen_joint_motion(dataset, phase):
    print(dataset, phase)
    file_joint = 'data/%s/%s_data_joint.npy' % (dataset, phase)
    file_bone = 'data/%s/%s_data_bone.npy' % (dataset, phase)
    file_joint_bone = 'data/%s/%s_data_joint_bone.npy' % (dataset, phase)
    data_jpt = np.load(file_joint)
    data_bone = np.load(file_bone)
    data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)

    np.save(file_joint_bone, data_jpt_bone)


if __name__ == "__main__":
    for dataset in datasets:
        for phase in phases:
            gen_joint_motion(dataset, phase)
