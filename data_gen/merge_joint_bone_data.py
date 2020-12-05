import os
import numpy as np

phases = {
    'train', 'val'
}

datasets = {
    'ntu/xview', 'ntu/xsub'
}

if __name__ == "__main__":
    for dataset in datasets:
        for phase in phases:
            print(dataset, phase)
            data_jpt = np.load('data/{}/{}_data_joint.npy' % (dataset, phase))
            data_bone = np.load('data/{}/{}_data_bone.npy' % (dataset, phase))
            data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)

            np.save('data/{}/{}_data_joint_bone.npy' %
                    (dataset, phase), data_jpt_bone)
            np.save('data/{}/{}_data_joint_bone.npy' %
                    (dataset, phase), data_jpt_bone)
